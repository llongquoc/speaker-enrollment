"""End-to-end evaluation pipeline: manifest, embeddings, threshold, the experiments of
both papers, operating-point and design-choice analysis, and report.

Every step reads the configuration and writes into a workspace directory; the
expensive embedding step is cached and recomputed only when the utterance list or the
checkpoint changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import Config
from .data import build_manifest, index_corpus, read_relabel_csv, read_speaker_list, read_trials, read_utterance_list
from .data.protocol import read_transcripts, write_list
from .embedding.extract import EmbeddingSet

log = logging.getLogger(__name__)

QUICK_TEST_SPEAKERS = 6
QUICK_VALIDATION_SPEAKERS = 6
QUICK_REPETITIONS = 3


# --------------------------------------------------------------------------- workspace
@dataclass
class Workspace:
    root: Path

    @classmethod
    def create(cls, config: Config, quick: bool = False) -> Workspace:
        root = Path(config.output_dir + ("_quick" if quick else ""))
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    def path(self, *parts: str) -> Path:
        """File path inside the workspace (parent directories are created)."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def directory(self, *parts: str, clean: bool = False) -> Path:
        """Directory inside the workspace (created).  ``clean`` first removes an existing
        directory, so that a step's outputs never mix with those of an earlier run."""
        p = self.root.joinpath(*parts)
        if clean and p.is_dir():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def manifest(self) -> Path:
        return self.path("manifest.csv")

    def embeddings(self, subset: str) -> Path:
        return self.path("embeddings", f"{subset}.npz")

    @property
    def threshold(self) -> Path:
        return self.path("threshold.json")

    @property
    def phonemes(self) -> Path:
        return self.path("phonemes", "test_phonemes.csv")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the preceding pipeline step first")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(config: Config) -> str:
    """SHA-256 of the checkpoint, verified against ``model.sha256`` when that is set."""
    from .embedding.ecapa_tdnn import check_not_lfs_pointer

    path = Path(config.model.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path} (see checkpoints/README.md)")
    check_not_lfs_pointer(path)
    digest = file_sha256(path)
    expected = config.model.sha256
    if expected and digest != expected.strip().lower():
        raise ValueError(
            f"{path} has SHA-256 {digest}, but model.sha256 is {expected}: use the checkpoint of this "
            "repository, or set model.sha256 to null to evaluate another model"
        )
    return digest


def _manifest(ws: Workspace) -> pd.DataFrame:
    if not ws.manifest.exists():
        raise FileNotFoundError(f"{ws.manifest} not found; run `speaker-enrollment prepare` first")
    return pd.read_csv(ws.manifest, dtype={"utterance": str, "speaker": str, "locator": str, "subset": str})


def load_embeddings(ws: Workspace, subset: str) -> EmbeddingSet:
    path = ws.embeddings(subset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `speaker-enrollment embed` first")
    return EmbeddingSet.load(path)


def load_threshold(ws: Workspace) -> float:
    return float(_read_json(ws.threshold)["threshold"])


# --------------------------------------------------------------------------- steps
def verify_protocol(config: Config) -> None:
    """Check the protocol files against the ``SHA256SUMS`` file next to the test list,
    if there is one."""
    directory = Path(config.protocol.test_utterances).parent
    sums = directory / "SHA256SUMS"
    if not sums.exists():
        return
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        path = directory / name.strip().lstrip("*")
        if not path.exists():
            continue
        if file_sha256(path) != digest.lower():
            raise ValueError(
                f"{path} differs from the released protocol ({sums}); restore the file, or update {sums.name} "
                "if the change is intended"
            )
        checked += 1
    log.info("%d protocol files match %s", checked, sums)


def prepare(config: Config, ws: Workspace, quick: bool = False) -> pd.DataFrame:
    """Index the corpus and assign test / validation / other subsets."""
    verify_protocol(config)
    index = index_corpus(config.corpus.path, config.corpus.audio_subdir)
    test_utterances = read_utterance_list(config.protocol.test_utterances)
    validation = read_speaker_list(config.protocol.validation_speakers)
    relabel = read_relabel_csv(config.protocol.relabel) if config.protocol.relabel else None
    manifest = build_manifest(index, test_utterances, validation, relabel)
    if quick:
        manifest = _restrict_for_quick_run(manifest)
    manifest.to_csv(ws.manifest, index=False)
    for subset, group in manifest.groupby("subset"):
        log.info("%-10s %4d speakers %6d utterances", subset, group.speaker.nunique(), len(group))
    return manifest


def _restrict_for_quick_run(manifest: pd.DataFrame) -> pd.DataFrame:
    """Smallest test speakers and largest validation speakers, for a fast end-to-end check."""
    m = manifest.copy()
    test_sizes = m[m.subset == "test"].speaker.value_counts().sort_values(kind="stable")
    keep_test = set(test_sizes.index[:QUICK_TEST_SPEAKERS])
    val_sizes = m[m.subset == "validation"].speaker.value_counts()
    keep_val = set(val_sizes[val_sizes >= 2].index[:QUICK_VALIDATION_SPEAKERS])
    m.loc[(m.subset == "test") & ~m.speaker.isin(keep_test), "subset"] = "excluded"
    m.loc[(m.subset == "validation") & ~m.speaker.isin(keep_val), "subset"] = "excluded"
    return m


def _cached_embeddings(path: Path, rows: pd.DataFrame, stamp: dict) -> bool:
    """Whether ``path`` holds embeddings of exactly ``rows`` (utterances and labels)
    computed with the checkpoint and settings in ``stamp``."""
    meta = path.with_suffix(".json")
    if not (path.exists() and meta.exists()):
        return False
    recorded = _read_json(meta)
    if any(recorded.get(key) != value for key, value in stamp.items()):
        return False
    cached = EmbeddingSet.load(path)
    return cached.utterances == list(rows.utterance) and cached.speakers == list(rows.speaker)


def embed(
    config: Config,
    ws: Workspace,
    subsets=("validation", "test"),
    force: bool = False,
    reuse_from: str | Path | None = None,
) -> None:
    """Whole-utterance embeddings of the validation and test utterances.

    Embeddings are cached in the workspace.  ``reuse_from`` names another output
    directory whose embeddings are copied when they match (same utterances, labels,
    checkpoint and settings), e.g. to evaluate a configuration variant without
    recomputing them.
    """
    from .embedding.ecapa_tdnn import load_checkpoint
    from .embedding.extract import extract_embeddings, select_device, verify_device

    stamp = {"checkpoint_sha256": checkpoint_sha256(config), "max_seconds": config.model.max_seconds}
    manifest = _manifest(ws)
    todo = []
    for subset in subsets:
        rows = manifest[manifest.subset == subset]
        if rows.empty:
            raise ValueError(f"no '{subset}' utterances in the manifest")
        path = ws.embeddings(subset)
        if not force and _cached_embeddings(path, rows, stamp):
            log.info("embeddings for %s are up to date (%s)", subset, path)
            continue
        source = Path(reuse_from) / "embeddings" / f"{subset}.npz" if reuse_from else None
        if not force and source is not None and source.resolve() != path.resolve():
            if _cached_embeddings(source, rows, stamp):
                shutil.copyfile(source, path)
                shutil.copyfile(source.with_suffix(".json"), path.with_suffix(".json"))
                log.info("embeddings for %s copied from %s", subset, source)
                continue
            log.warning("%s does not match the %s utterances of this run; recomputing", source, subset)
        todo.append((subset, rows, path))
    if not todo:
        return
    model, _ = load_checkpoint(config.model.checkpoint)
    device = verify_device(model, select_device(config.model.device), todo[0][1].locator.iloc[0])
    for subset, rows, path in todo:
        log.info("embedding %d %s utterances on %s", len(rows), subset, device)
        start = time.time()
        vectors = extract_embeddings(model, list(rows.locator), device, config.model.max_seconds)
        EmbeddingSet(list(rows.utterance), list(rows.speaker), vectors).save(path)
        seconds = time.time() - start
        _write_json(path.with_suffix(".json"), {**stamp, "device": device, "seconds": round(seconds, 1)})
        log.info("%s embeddings written to %s (%.0f s)", subset, path, seconds)


def calibrate(config: Config, ws: Workspace) -> dict:
    """Decision threshold from the validation speakers; EERs of the test set for reference."""
    from .verification import all_pair_scores, calibrate_threshold, eer_point, operating_point, trial_scores

    validation = load_embeddings(ws, "validation")
    point = calibrate_threshold(
        validation.vectors,
        validation.labels,
        config.threshold.rule,
        config.threshold.target_far,
        config.threshold.value,
    )
    result = {
        "rule": config.threshold.rule,
        "threshold": point.threshold,
        "validation": {"speakers": validation.n_speakers, "utterances": len(validation), **point.as_dict()},
    }
    test = load_embeddings(ws, "test")
    target, impostor = all_pair_scores(test.vectors, test.labels)
    result["test_all_pairs"] = {
        "eer": eer_point(target, impostor).as_dict(),
        "at_threshold": operating_point(target, impostor, point.threshold).as_dict(),
    }
    if config.protocol.test_trials:
        trials = read_trials(config.protocol.test_trials)
        target, impostor, n = trial_scores(
            trials.label.to_numpy(), list(trials.a), list(trials.b), test.utterances, test.vectors
        )
        if len(target) and len(impostor):
            result["test_trials"] = {
                "file": config.protocol.test_trials,
                "trials_scored": n,
                "trials_total": len(trials),
                "eer": eer_point(target, impostor).as_dict(),
            }
    _write_json(ws.threshold, result)
    log.info(
        "threshold %.4f (%s rule; validation EER %.2f%%, FAR %.2f%%, FRR %.2f%%; %d speakers)",
        point.threshold,
        config.threshold.rule,
        100 * point.eer,
        100 * point.far,
        100 * point.frr,
        validation.n_speakers,
    )
    return result


def phonemize(config: Config, ws: Workspace) -> pd.DataFrame:
    """Phoneme units and coverage of the test transcripts."""
    from .phonetics import make_g2p, phonemize_transcripts

    if not config.protocol.transcripts:
        raise ValueError("protocol.transcripts is not set")
    manifest = _manifest(ws)
    needed = set(manifest.utterance[manifest.subset == "test"])
    transcripts = read_transcripts(config.protocol.transcripts)
    missing = needed - set(transcripts.utterance)
    if missing:
        raise KeyError(f"{len(missing)} test utterances have no transcript, e.g. {sorted(missing)[:3]}")
    g2p = make_g2p(
        config.phoneme_selection.g2p, config.phoneme_selection.compound_vowels, config.phoneme_selection.tones
    )
    table = phonemize_transcripts(transcripts[transcripts.utterance.isin(needed)], g2p)
    table.to_csv(ws.phonemes, index=False)
    inventory = set(" ".join(table.units).split())
    log.info(
        "phonemized %d transcripts with %s: %d distinct units, mean coverage %.1f, %d with non-Vietnamese tokens",
        len(table),
        config.phoneme_selection.g2p,
        len(inventory),
        table.coverage.mean(),
        (table.unparsed != "").sum(),
    )
    return table


def _repetitions(config: Config, quick: bool) -> int:
    return (
        min(QUICK_REPETITIONS, config.similarity_selection.repetitions)
        if quick
        else config.similarity_selection.repetitions
    )


def evaluate_icpram2025(config: Config, ws: Workspace, quick: bool = False) -> pd.DataFrame:
    """ICPRAM 2025: bad / random / optimal enrollment for k = 1..max_size."""
    from .experiments import plots
    from .experiments.evaluation import SpeakerSimilarities, run_similarity_selection, summarize

    test, tau = load_embeddings(ws, "test"), load_threshold(ws)
    cfg = config.similarity_selection
    sims = SpeakerSimilarities(test, tau)
    log.info(
        "ICPRAM 2025: %d test utterances, %d speakers, threshold %.4f, candidate rule %s, scoring %s",
        len(test),
        test.n_speakers,
        tau,
        cfg.candidate_rule,
        config.identification.scoring,
    )
    runs = run_similarity_selection(
        test,
        tau,
        sizes=range(1, cfg.max_size + 1),
        repetitions=_repetitions(config, quick),
        candidate_rules=(cfg.candidate_rule,),
        scorings=(config.identification.scoring,),
        reject=config.identification.reject,
        seed=cfg.seed,
        similarities=sims,
    )
    out = ws.directory("icpram2025", clean=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(out / "summary.csv", index=False)
    _write_json(out / "candidates.json", sims.candidate_statistics(cfg.candidate_rule))
    plots.metrics_vs_size(summary, ("optimal", "random", "bad"), ("accuracy", "f1"), out / "metrics_vs_k.png")
    plots.embedding_pca(test.vectors, test.labels, test.speaker_ids, out / "embedding_pca.png")
    return summary


def evaluate_pricai2025(config: Config, ws: Workspace, quick: bool = False) -> pd.DataFrame:
    """PRICAI 2025 experiment: coverage-ordered High-P and Low-P enrollment, plus the
    Random and Optimal configurations used as Baseline and Similarity."""
    from .experiments import plots
    from .experiments.evaluation import SpeakerSimilarities, run_phoneme_selection, run_similarity_selection, summarize
    from .phonetics import load_phoneme_sets

    test, tau = load_embeddings(ws, "test"), load_threshold(ws)
    coverage, phoneme_sets = load_phoneme_sets(ws.phonemes, test.utterances)
    cfg, scoring = config.similarity_selection, config.identification.scoring
    sims = SpeakerSimilarities(test, tau)
    sizes = range(1, cfg.max_size + 1)
    runs = run_phoneme_selection(
        test,
        tau,
        coverage,
        phoneme_sets,
        sizes,
        (cfg.candidate_rule,),
        (config.phoneme_selection.quality_check,),
        (scoring,),
        config.identification.reject,
        sims,
    )
    baseline = run_similarity_selection(
        test,
        tau,
        ("random", "optimal"),
        sizes,
        _repetitions(config, quick),
        (cfg.candidate_rule,),
        (scoring,),
        config.identification.reject,
        cfg.seed,
        phoneme_sets,
        sims,
    )
    out = ws.directory("pricai2025", clean=True)
    all_runs = pd.concat([baseline, runs], ignore_index=True)
    all_runs.to_csv(out / "runs.csv", index=False)
    summary = summarize(all_runs)
    summary.to_csv(out / "summary.csv", index=False)
    plots.metrics_vs_coverage(summary, out / "metrics_vs_coverage.png")
    plots.metrics_vs_size(
        summary, ("optimal", "random", "high_p", "low_p"), ("accuracy", "f1"), out / "metrics_vs_k.png"
    )
    return summary


def analyze(config: Config, ws: Workspace, quick: bool = False) -> None:
    """Operating-point sweep and design-choice comparison."""
    from .experiments import plots
    from .experiments.analysis import design_choices, threshold_sweep
    from .phonetics import load_phoneme_sets

    test, validation, tau = load_embeddings(ws, "test"), load_embeddings(ws, "validation"), load_threshold(ws)
    coverage = phoneme_sets = None
    if ws.phonemes.exists():
        coverage, phoneme_sets = load_phoneme_sets(ws.phonemes, test.utterances)
    a = config.analysis
    thresholds = sorted(set(a.thresholds[::5] if quick else a.thresholds) | {tau})
    repetitions = 2 if quick else a.repetitions
    sizes = range(1, config.similarity_selection.max_size + 1)
    k = min(a.report_size, config.similarity_selection.max_size)
    out = ws.directory("analysis", clean=True)
    _write_json(
        out / "settings.json",
        {
            "thresholds": thresholds,
            "repetitions": repetitions,
            "report_size": k,
            "candidate_rules": a.candidate_rules,
            "scorings": a.scorings,
            "quality_checks": a.quality_checks,
        },
    )
    log.info("operating-point analysis: %d thresholds, %d random draws", len(thresholds), repetitions)
    sweep, points = threshold_sweep(
        test,
        thresholds,
        coverage,
        phoneme_sets,
        config.similarity_selection.candidate_rule,
        config.identification.scoring,
        config.phoneme_selection.quality_check,
        sizes,
        repetitions,
        config.similarity_selection.seed,
        validation,
    )
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    points.to_csv(out / "operating_points.csv", index=False)
    plots.operating_point(sweep, out / "operating_point.png", k, tau)
    log.info("design-choice analysis at τ = %.4f", tau)
    icpram, pricai = design_choices(
        test,
        tau,
        coverage,
        phoneme_sets,
        a.candidate_rules,
        a.scorings,
        a.quality_checks,
        sizes,
        repetitions,
        config.similarity_selection.seed,
    )
    icpram.to_csv(out / "design_choices_icpram2025.csv", index=False)
    if pricai is not None:
        pricai.to_csv(out / "design_choices_pricai2025.csv", index=False)


def write_run_info(config: Config, ws: Workspace, started: float, command: str) -> None:
    import torch

    info = {
        "package_version": __version__,
        "command": command,
        "config_file": getattr(config, "source", None),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "duration_seconds": round(time.time() - started, 1),
        "checkpoint_sha256": checkpoint_sha256(config),
        "config": config.to_dict(),
    }
    _write_json(ws.path("run_info.json"), info)


def reproduce(
    config: Config, ws: Workspace, quick: bool = False, force: bool = False, reuse_from: str | Path | None = None
) -> Path:
    """All steps; returns the path of the report.  ``force`` recomputes the embeddings;
    ``reuse_from`` copies matching embeddings from another output directory."""
    from .experiments.report import write_report

    started = time.time()
    prepare(config, ws, quick)
    embed(config, ws, force=force, reuse_from=reuse_from)
    calibrate(config, ws)
    evaluate_icpram2025(config, ws, quick)
    if config.protocol.transcripts:
        phonemize(config, ws)
        evaluate_pricai2025(config, ws, quick)
    analyze(config, ws, quick)
    write_run_info(config, ws, started, " ".join(["speaker-enrollment", *sys.argv[1:]]))
    report = write_report(config, ws)
    log.info("report written to %s", report)
    return report


# --------------------------------------------------------------------------- split recovery
def recover_split(
    config: Config,
    ws: Workspace,
    per_speaker: int = 3,
    min_similarity: float = 0.25,
    write_protocol_dir: str | None = None,
) -> dict:
    """Recover the checkpoint's training speakers and derive the validation speakers
    (see :mod:`speaker_enrollment.split_recovery`)."""
    from .embedding.ecapa_tdnn import load_checkpoint
    from .embedding.extract import extract_embeddings, select_device, verify_device
    from .split_recovery import choose_reference_utterances
    from .split_recovery import recover_split as run_recovery
    from .verification import l2_normalize

    index = index_corpus(config.corpus.path, config.corpus.audio_subdir)
    test_speakers = sorted({u.split("/")[0] for u in read_utterance_list(config.protocol.test_utterances)})
    references = choose_reference_utterances(index, per_speaker)
    checkpoint_sha256(config)
    model, weights = load_checkpoint(config.model.checkpoint)
    if weights is None:
        raise ValueError(f"{config.model.checkpoint} has no classifier weights (speaker_loss.weight)")
    path = ws.path("split_recovery", "reference_embeddings.npz")
    if path.exists() and EmbeddingSet.load(path).utterances == list(references.utterance):
        embeddings = EmbeddingSet.load(path)
    else:
        device = verify_device(model, select_device(config.model.device), references.locator.iloc[0])
        log.info(
            "embedding %d reference utterances of %d speakers on %s",
            len(references),
            references.speaker.nunique(),
            device,
        )
        vectors = extract_embeddings(model, list(references.locator), device, max_seconds=10.0)
        embeddings = EmbeddingSet(list(references.utterance), list(references.speaker), vectors)
        embeddings.save(path)
    speakers = embeddings.speaker_ids
    speaker_vectors = l2_normalize(np.stack([embeddings.vectors[g].mean(0) for g in embeddings.groups()]))
    result = run_recovery(speakers, speaker_vectors, weights.numpy(), test_speakers, min_similarity)
    counts = index.speaker.value_counts()
    summary = dict(result.summary)
    for role in ("train", "validation", "test"):
        summary[f"{role}_utterances"] = int(sum(counts[s] for s in result.speakers(role)))
    result.table.to_csv(ws.path("split_recovery", "speakers.csv"), index=False)
    _write_json(ws.path("split_recovery", "summary.json"), summary)
    log.info(
        "training speakers: %d (%d utterances); validation: %d (%d); test: %d; classes without speaker: %s",
        summary["n_train"],
        summary["train_utterances"],
        summary["n_validation"],
        summary["validation_utterances"],
        summary["n_test"],
        summary["classes_without_speaker"],
    )
    if write_protocol_dir:
        target = Path(write_protocol_dir)
        target.mkdir(parents=True, exist_ok=True)
        note = (
            f"Derived by `speaker-enrollment recover-split` from {Path(config.model.checkpoint).name} "
            f"(minimum class similarity {min_similarity})."
        )
        write_list(target / "training_speakers.txt", result.speakers("train"), note)
        write_list(target / "validation_speakers.txt", result.speakers("validation"), note)
        write_list(target / "test_speakers.txt", result.speakers("test"))
    return summary
