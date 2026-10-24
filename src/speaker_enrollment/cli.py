"""Command-line interface.

    speaker-enrollment reproduce --config configs/vietnam_celeb.yaml [--quick]

runs every step of the evaluation; the individual steps are available as separate
commands.  All commands read the same configuration file; values can be overridden
with ``--set section.key=value``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import __version__  # noqa: E402

DEFAULT_CONFIG = "configs/vietnam_celeb.yaml"
log = logging.getLogger("speaker_enrollment")
FORMAT = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def _setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(FORMAT)
    root.addHandler(console)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _log_to_file(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(FORMAT)
    logging.getLogger().addHandler(handler)


def _context(args, quick: bool = False):
    from .config import load_config
    from .pipeline import Workspace

    overrides = list(args.set or [])
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    path = args.config
    if path == DEFAULT_CONFIG and not Path(path).exists():
        log.warning("%s not found (run from the repository root); using built-in defaults", path)
        path = None
    config = load_config(path, overrides)
    ws = Workspace.create(config, quick)
    _log_to_file(ws.root / "run.log")
    return config, ws


# --------------------------------------------------------------------------- commands
def cmd_reproduce(args):
    from .pipeline import reproduce

    config, ws = _context(args, args.quick)
    if args.quick:
        log.info("quick run: %s", ws.root)
    reproduce(config, ws, quick=args.quick, force=args.force, reuse_from=args.reuse_embeddings)


def cmd_prepare(args):
    from .pipeline import prepare

    config, ws = _context(args, args.quick)
    prepare(config, ws, quick=args.quick)


def cmd_embed(args):
    from .pipeline import embed

    config, ws = _context(args, args.quick)
    embed(config, ws, subsets=tuple(args.subsets), force=args.force, reuse_from=args.reuse_embeddings)


def cmd_calibrate(args):
    from .pipeline import calibrate

    config, ws = _context(args, args.quick)
    calibrate(config, ws)


def cmd_phonemize(args):
    from .pipeline import phonemize

    config, ws = _context(args, args.quick)
    phonemize(config, ws)


def cmd_icpram2025(args):
    from .pipeline import evaluate_icpram2025

    config, ws = _context(args, args.quick)
    evaluate_icpram2025(config, ws, quick=args.quick)


def cmd_pricai2025(args):
    from .pipeline import evaluate_pricai2025

    config, ws = _context(args, args.quick)
    evaluate_pricai2025(config, ws, quick=args.quick)


def cmd_analyze(args):
    from .pipeline import analyze

    config, ws = _context(args, args.quick)
    analyze(config, ws, quick=args.quick)


def cmd_report(args):
    from .experiments.report import write_report

    config, ws = _context(args, args.quick)
    log.info("report written to %s", write_report(config, ws, args.output))


def cmd_recover_split(args):
    from .pipeline import recover_split

    config, ws = _context(args)
    recover_split(config, ws, args.references_per_speaker, args.min_similarity, args.write_protocol)


def cmd_audit_labels(args):
    from .data.label_audit import label_consistency
    from .pipeline import load_embeddings

    config, ws = _context(args, args.quick)
    table = label_consistency(load_embeddings(ws, args.subset))
    path = ws.path(f"label_audit_{args.subset}.csv")
    table.to_csv(path, index=False)
    log.info("%d utterances with margin < %g -> %s", int((table.margin < args.margin).sum()), args.margin, path)


def cmd_make_prompts(args):
    from .phonetics import generate_prompts, make_g2p, prompt_coverage

    if args.corpus:
        text = Path(args.corpus).read_text(encoding="utf-8")
    else:
        from importlib import resources

        text = resources.files("speaker_enrollment").joinpath("resources/sample_corpus_vi.txt").read_text("utf-8")
    g2p = make_g2p(args.g2p)
    prompts = generate_prompts(text, args.sentences, args.words, g2p)
    Path(args.output).write_text("\n".join(" ".join(p) for p in prompts) + "\n", encoding="utf-8")
    report = prompt_coverage(prompts, g2p, text)
    report.to_csv(Path(args.output).with_suffix(".csv"), index=False)
    print(report.to_string(index=False))


def cmd_transcribe(args):
    import pandas as pd

    from .phonetics.transcription import transcribe

    config, ws = _context(args, args.quick)
    manifest = pd.read_csv(ws.manifest)
    rows = manifest[manifest.subset == args.subset]
    transcribe(
        zip(rows.utterance, rows.locator),
        ws.path(f"transcripts_{args.subset}.csv"),
        args.model,
        args.device,
        args.batch_size,
        args.language,
    )


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speaker-enrollment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML configuration (default: {DEFAULT_CONFIG})")
    common.add_argument("--output-dir", help="override output_dir of the configuration")
    common.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a configuration value, e.g. --set threshold.rule=far (repeatable)",
    )
    common.add_argument(
        "--quick",
        action="store_true",
        help="6 test and 6 validation speakers, 3 random draws (output in <output_dir>_quick)",
    )
    common.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, func, help_text, parents=(common,)):
        p = sub.add_parser(name, help=help_text, description=help_text, parents=list(parents))
        p.set_defaults(func=func)
        return p

    reuse_help = "copy matching embeddings from another output directory instead of recomputing them"
    p = add("reproduce", cmd_reproduce, "run every step and write REPORT.md")
    p.add_argument("--force", action="store_true", help="recompute cached embeddings")
    p.add_argument("--reuse-embeddings", metavar="DIR", help=reuse_help)
    add("prepare", cmd_prepare, "index the corpus and assign test / validation subsets")
    p = add("embed", cmd_embed, "extract validation and test embeddings")
    p.add_argument("--subsets", nargs="+", default=["validation", "test"])
    p.add_argument("--force", action="store_true", help="recompute cached embeddings")
    p.add_argument("--reuse-embeddings", metavar="DIR", help=reuse_help)
    add("calibrate", cmd_calibrate, "decision threshold from the validation speakers")
    add("phonemize", cmd_phonemize, "phoneme units and coverage of the test transcripts")
    add("icpram2025", cmd_icpram2025, "ICPRAM 2025 experiment: bad / random / optimal enrollment")
    add("pricai2025", cmd_pricai2025, "PRICAI 2025 experiment: phoneme-coverage-ordered enrollment")
    add("analyze", cmd_analyze, "operating-point and design-choice analysis")
    p = add("report", cmd_report, "write REPORT.md from existing results")
    p.add_argument(
        "--output",
        metavar="FILE",
        help="write the report to FILE (e.g. docs/results.md) and copy its figures to figures/ next to it",
    )
    p = add(
        "recover-split",
        cmd_recover_split,
        "recover the checkpoint's training speakers and derive the validation speakers",
    )
    p.add_argument("--references-per-speaker", type=int, default=3)
    p.add_argument("--min-similarity", type=float, default=0.25)
    p.add_argument("--write-protocol", metavar="DIR", help="write training/validation/test speaker lists to DIR")
    p = add("audit-labels", cmd_audit_labels, "rank utterances by label consistency for manual review")
    p.add_argument("--subset", default="test")
    p.add_argument("--margin", type=float, default=0.0)
    p = add("make-prompts", cmd_make_prompts, "phoneme-rich enrollment prompts (PRICAI 2025, Algorithm 1)", parents=())
    p.add_argument("--corpus", help="UTF-8 text corpus (default: bundled sample)")
    p.add_argument("--sentences", type=int, default=10, help="number of sentences N")
    p.add_argument("--words", type=int, default=15, help="maximum words per sentence W")
    p.add_argument("--g2p", choices=("viphoneme", "rules"), default="viphoneme")
    p.add_argument("--output", default="prompts.txt")
    p.add_argument("-v", "--verbose", action="store_true")
    p = add("transcribe", cmd_transcribe, "transcribe utterances with PhoWhisper (requires the asr extra)")
    p.add_argument("--subset", default="test")
    p.add_argument("--model", default="vinai/PhoWhisper-large")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--language")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        if getattr(args, "verbose", False):
            raise
        log.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
