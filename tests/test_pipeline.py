"""End-to-end runs of the command-line interface on a miniature corpus."""

import json

import pandas as pd
import pytest

from speaker_enrollment.cli import main
from tests.conftest import SENTENCES


def test_reproduce_end_to_end(mini_corpus):
    main(["reproduce", "--config", str(mini_corpus["config"])])
    out = mini_corpus["root"] / "outputs"
    for name in (
        "manifest.csv",
        "embeddings/test.npz",
        "embeddings/validation.npz",
        "threshold.json",
        "phonemes/test_phonemes.csv",
        "icpram2025/summary.csv",
        "pricai2025/summary.csv",
        "analysis/operating_points.csv",
        "analysis/operating_point.png",
        "analysis/design_choices_icpram2025.csv",
        "analysis/design_choices_pricai2025.csv",
        "icpram2025/metrics_vs_k.png",
        "pricai2025/metrics_vs_k.png",
        "pricai2025/metrics_vs_coverage.png",
        "run_info.json",
        "REPORT.md",
    ):
        assert (out / name).exists(), name
    threshold = json.loads((out / "threshold.json").read_text())
    assert threshold["rule"] == "eer" and threshold["validation"]["speakers"] == 4
    assert threshold["test_trials"]["trials_scored"] == threshold["test_trials"]["trials_total"]
    summary = pd.read_csv(out / "icpram2025" / "summary.csv")
    assert set(summary.case) == {"bad", "random", "optimal"} and summary.k.max() == 4
    report = (out / "REPORT.md").read_text()
    for heading in ("# Reproduced results", "## Setup", "### Optimal case", "### Low-P and High-P"):
        assert heading in report
    assert "## Operating point" in report and "## Design choices" in report
    for heading in ("### Accuracy at k = ", "### F1 at k = ", "#### Accuracy", "#### F1"):
        assert heading in report
    assert "| Ordering | Accuracy | F1 |" in report
    assert "](icpram2025/metrics_vs_k.png)" in report
    # cached embeddings are reused; step outputs are rewritten without stale files
    stamp = (out / "embeddings" / "test.npz").stat().st_mtime
    (out / "analysis" / "stale.csv").write_text("x\n")
    main(["reproduce", "--config", str(mini_corpus["config"])])
    assert (out / "embeddings" / "test.npz").stat().st_mtime == stamp
    assert not (out / "analysis" / "stale.csv").exists()
    assert json.loads((out / "embeddings" / "test.json").read_text())["device"] == "cpu"
    # the report can be written elsewhere together with its figures
    docs = mini_corpus["root"] / "docs" / "results.md"
    main(["report", "--config", str(mini_corpus["config"]), "--output", str(docs)])
    assert "](figures/icpram2025_metrics_vs_k.png)" in docs.read_text()
    assert (docs.parent / "figures" / "analysis_operating_point.png").exists()
    # a variant in another directory copies the matching embeddings
    variant = mini_corpus["root"] / "variant"
    config = str(mini_corpus["config"])
    main(
        ["reproduce", "--config", config, "--output-dir", str(variant), "--reuse-embeddings", str(out)]
        + ["--set", "threshold.rule=fixed", "--set", "threshold.value=0.3"]
    )
    for subset in ("test", "validation"):
        assert (variant / "embeddings" / f"{subset}.npz").read_bytes() == (
            out / "embeddings" / f"{subset}.npz"
        ).read_bytes()
    assert json.loads((variant / "threshold.json").read_text())["threshold"] == 0.3


def test_quick_run_and_split_recovery(mini_corpus, tmp_path):
    main(["reproduce", "--config", str(mini_corpus["config"]), "--quick"])
    quick = mini_corpus["root"] / "outputs_quick"
    manifest = pd.read_csv(quick / "manifest.csv")
    assert manifest[manifest.subset == "test"].speaker.nunique() == 6
    assert (quick / "REPORT.md").exists()
    protocol = tmp_path / "derived"
    main(
        [
            "recover-split",
            "--config",
            str(mini_corpus["config"]),
            "--write-protocol",
            str(protocol),
            "--references-per-speaker",
            "2",
        ]
    )
    test_speakers = (protocol / "test_speakers.txt").read_text().split()
    assert test_speakers == [f"t{i:02d}" for i in range(7)]
    assert (protocol / "validation_speakers.txt").exists() and (protocol / "training_speakers.txt").exists()


def test_cli_reports_errors_without_traceback(mini_corpus, capsys):
    config = str(mini_corpus["config"])
    for command, override in (("prepare", "identification.scoring=median"), ("embed", "model.sha256=0123abcd")):
        with pytest.raises(SystemExit) as exit_info:
            main([command, "--config", config, "--set", override])
        assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "identification.scoring" in error and "model.sha256 is 0123abcd" in error


def test_protocol_files_are_verified(mini_corpus, capsys):
    import hashlib

    protocol, config = mini_corpus["protocol"], str(mini_corpus["config"])
    digest = hashlib.sha256((protocol / "test_utterances.txt").read_bytes()).hexdigest()
    (protocol / "SHA256SUMS").write_text(f"{digest}  test_utterances.txt\n{'0' * 64}  validation_speakers.txt\n")
    with pytest.raises(SystemExit):
        main(["prepare", "--config", config])
    assert "validation_speakers.txt differs from the released protocol" in capsys.readouterr().err
    (protocol / "SHA256SUMS").write_text(f"{digest}  test_utterances.txt\n")
    main(["prepare", "--config", config])


def test_make_prompts(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(SENTENCES), encoding="utf-8")
    main(
        [
            "make-prompts",
            "--corpus",
            str(corpus),
            "--sentences",
            "3",
            "--words",
            "4",
            "--g2p",
            "rules",
            "--output",
            str(tmp_path / "prompts.txt"),
        ]
    )
    lines = (tmp_path / "prompts.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(len(line.split()) == 4 for line in lines)


def test_relabelling_invalidates_cached_embeddings(mini_corpus, tmp_path):
    from speaker_enrollment.embedding import EmbeddingSet

    config = str(mini_corpus["config"])
    main(["prepare", "--config", config])
    main(["embed", "--config", config, "--subsets", "test"])
    first = mini_corpus["test_utterances"][0]
    relabel = tmp_path / "relabel.csv"
    relabel.write_text(f"utterance,speaker\n{first},t06\n")
    override = ["--set", f"protocol.relabel={relabel}"]
    main(["prepare", "--config", config, *override])
    main(["embed", "--config", config, "--subsets", "test", *override])
    embeddings = EmbeddingSet.load(mini_corpus["root"] / "outputs" / "embeddings" / "test.npz")
    assert embeddings.speakers[embeddings.utterances.index(first)] == "t06"
