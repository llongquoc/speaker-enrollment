"""Markdown report of an evaluation run.

:func:`write_report` writes ``REPORT.md`` into the run's output directory.  With
``output``, the report is written to that file instead (for example
``docs/results.md``) and the figures it shows are copied into ``figures/`` next to it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from .. import __version__
from .analysis import mean_over_sizes
from .evaluation import METRICS, coverage_correlation, recommended_size
from .tables import (
    CASE_NAMES,
    EXPERIMENT_CASES,
    METRIC_NAMES,
    icpram_table,
    markdown_table,
    ordering_holds,
    ordering_table,
    pricai_coverage_table,
    pricai_methods_table,
    ranges,
)

# Metrics of the findings and of the operating-point and design-choice tables, in the order shown.
SUMMARY_METRICS = ("accuracy", "f1")

RULE_DESCRIPTIONS = {
    "eer": "equal-error-rate point of the validation pairs",
    "far": "lowest threshold meeting the target false-acceptance rate on the validation pairs",
    "fixed": "fixed value",
}


def _pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def _metric(metric: str) -> str:
    """Metric name inside a sentence: ``F1``, ``accuracy``, ``precision``, ``recall``."""
    return METRIC_NAMES[metric] if metric == "f1" else METRIC_NAMES[metric].lower()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class _Figures:
    """Markdown image links; copies the images when the report is written elsewhere."""

    def __init__(self, root: Path, output: Path, external: bool):
        self.root, self.output, self.external = root, output, external
        self.directory = output.parent / "figures"

    def __call__(self, relative: str, alt: str) -> list[str]:
        source = self.root / relative
        if not source.exists():
            return []
        if self.external:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / relative.replace("/", "_")
            shutil.copyfile(source, target)
            link = os.path.relpath(target, self.output.parent).replace(os.sep, "/")
        else:
            link = relative
        return [f"![{alt}]({link})", ""]


# --------------------------------------------------------------------------- sections
def _setup(config, root: Path, threshold: dict, info: dict) -> list[str]:
    manifest = pd.read_csv(root / "manifest.csv", usecols=["speaker", "subset"])

    def subset(name: str) -> str:
        g = manifest[manifest.subset == name]
        return f"{g.speaker.nunique():,} speakers, {len(g):,} utterances"

    validation = threshold["validation"]
    rule = threshold["rule"]
    tau_text = f"{threshold['threshold']:.4f}: {RULE_DESCRIPTIONS.get(rule, rule)}"
    if rule == "eer":
        tau_text += f" (EER {_pct(validation['eer'])})"
    else:
        tau_text += f" (validation FAR {_pct(validation['far'])}, FRR {_pct(validation['frr'])})"
    test_eer = f"{_pct(threshold['test_all_pairs']['eer']['eer'])} over all utterance pairs"
    if "test_trials" in threshold:
        trials = threshold["test_trials"]
        test_eer += f"; {_pct(trials['eer']['eer'])} over the trial list ({trials['trials_scored']:,} trials)"
    checkpoint = f"`{Path(config.model.checkpoint).name}`"
    if info.get("checkpoint_sha256"):
        checkpoint += f" (SHA-256 `{info['checkpoint_sha256'][:12]}…`)"
    device = _read_json(root / "embeddings" / "test.json").get("device")
    embeddings = "whole utterances, one forward pass each" + (f"; computed on `{device}`" if device else "")
    selection = (
        f"candidate rule `{config.similarity_selection.candidate_rule}`, scoring "
        f"`{config.identification.scoring}`, quality check `{config.phoneme_selection.quality_check}`; "
        f"{_repetitions_used(root)} random draws per configuration and k"
    )
    rows = [
        ["Test set", subset("test")],
        ["Validation set", subset("validation")],
        ["Speaker encoder", f"ECAPA-TDNN, {checkpoint}"],
        ["Embeddings", embeddings],
        ["Decision threshold τ", tau_text],
        ["Test EER (reference)", test_eer],
        ["Selection", selection],
        [
            "Metrics",
            "accuracy; precision, recall and F1 per test speaker, macro-averaged; rejected queries count as errors",
        ],
    ]
    if info:
        rows.append(
            [
                "Software",
                f"speaker-enrollment {info.get('package_version', __version__)}, Python {info.get('python')}, "
                f"PyTorch {info.get('torch')}, {info.get('platform')}",
            ]
        )
    return ["## Setup", "", markdown_table(["Item", "Value"], rows, "left"), ""]


def _icpram(root: Path, figures: _Figures, n_speakers: int) -> list[str]:
    path = root / "icpram2025" / "summary.csv"
    if not path.exists():
        return []
    s = pd.read_csv(path)
    ks = sorted(s.k.unique())
    optimal = s[s.case == "optimal"].set_index("k")
    lines = [
        "## Similarity-based enrollment selection (ICPRAM 2025 experiment)",
        "",
        f"Identification of the {n_speakers} test speakers with k enrollment utterances per speaker. Optimal is "
        "the method proposed in ICPRAM 2025, Random the baseline and Bad the control; mean ± standard "
        f"deviation over {_repetitions_used(root)} random draws.",
        "",
        *figures("icpram2025/metrics_vs_k.png", "Accuracy and F1 versus enrollment size, ICPRAM 2025 configurations"),
    ]
    for case in ("bad", "random", "optimal"):
        lines += [f"### {CASE_NAMES[case]} case", "", icpram_table(s, case), ""]
    lines += [
        "### Orderings",
        "",
        "Enrollment sizes k at which the first configuration has the higher mean.",
        "",
        ordering_table(s, "icpram2025"),
        "",
    ]
    if {1, 5, 10} <= set(ks):

        def change(first: int, last: int) -> str:
            return " and ".join(
                f"{_metric(m)} {optimal.loc[last, f'{m}_mean'] - optimal.loc[first, f'{m}_mean']:+.3f}"
                for m in SUMMARY_METRICS
            )

        lines.append(f"- Optimal case: {change(1, 5)} from k = 1 to 5; {change(5, 10)} from k = 5 to 10.")
    lines.append(_smallest_size(s, "optimal"))
    lines += _selection_diagnostics(s, EXPERIMENT_CASES["icpram2025"], n_speakers)
    lines += [
        "",
        *figures("icpram2025/embedding_pca.png", "Embeddings of seven test speakers, first three principal components"),
    ]
    return lines


def _pricai(root: Path, figures: _Figures, n_speakers: int) -> list[str]:
    path = root / "pricai2025" / "summary.csv"
    if not path.exists():
        return []
    s = pd.read_csv(path)
    correlation = coverage_correlation(s)
    lines = [
        "## Phoneme-based enrollment selection (PRICAI 2025 experiment)",
        "",
        "High-P is the method proposed in PRICAI 2025 and Low-P its control; both are deterministic. Baseline "
        "and Similarity are the Random and Optimal configurations of the ICPRAM 2025 experiment.",
        "",
        *figures("pricai2025/metrics_vs_k.png", "Accuracy and F1 versus enrollment size, PRICAI 2025 configurations"),
        "### Selection methods",
        "",
        pricai_methods_table(s),
        "",
        "### Low-P and High-P",
        "",
        "P: mean number of distinct phoneme units in a speaker's enrollment set.",
        "",
        pricai_coverage_table(s),
        "",
        "### Orderings",
        "",
        "Enrollment sizes k at which the first configuration has the higher mean.",
        "",
        ordering_table(s, "pricai2025"),
        "",
        "- Pearson correlation between P and each metric over the Low-P and High-P configurations: "
        + ", ".join(f"{_metric(m)} {correlation[m]:.2f}" for m in METRICS)
        + ".",
        _smallest_size(s, "high_p"),
    ]
    high_p, optimal = s[s.case == "high_p"].set_index("k"), s[s.case == "optimal"].set_index("k")
    if len(high_p) and len(optimal):
        k_max = int(optimal.index.max())
        parts = []
        for m in SUMMARY_METRICS:
            target = optimal.loc[k_max, f"{m}_mean"]
            reached = [int(k) for k in high_p.index if high_p.loc[k, f"{m}_mean"] >= target]
            if reached and reached[0] < k_max:
                k = reached[0]
                parts.append(f"{_metric(m)} with k = {k} ({high_p.loc[k, f'{m}_mean']:.3f} against {target:.3f})")
        if parts:
            lines.append(
                f"- High-P matches Optimal at k = {k_max} with fewer enrollment utterances: {', '.join(parts)}."
            )
    lines += _selection_diagnostics(s, EXPERIMENT_CASES["pricai2025"], n_speakers)
    lines += ["", *figures("pricai2025/metrics_vs_coverage.png", "Metrics versus phoneme coverage")]
    return lines


def _tau(value: float, calibrated: float) -> str:
    return f"{value:.4f}" if abs(value - calibrated) < 5e-5 else f"{value:.2f}"


def _operating_point(root: Path, figures: _Figures, calibrated: float) -> list[str]:
    sweep_path, points_path = root / "analysis" / "threshold_sweep.csv", root / "analysis" / "operating_points.csv"
    if not (sweep_path.exists() and points_path.exists()):
        return []
    sweep, points = pd.read_csv(sweep_path), pd.read_csv(points_path)
    settings = _read_json(root / "analysis" / "settings.json")
    k = int(settings.get("report_size", 5))
    cases = [c for c in ("bad", "random", "optimal", "low_p", "high_p") if (sweep.case == c).any()]
    rates = [(c, h) for c, h in (("validation_far", "Val. FAR"), ("validation_frr", "Val. FRR")) if c in points]
    header = ["τ"] + [h for _, h in rates] + [CASE_NAMES[c] for c in cases]
    lines = [
        "## Operating point",
        "",
        f"Both experiments repeated at decision thresholds from {points.threshold.min():.2f} to "
        f"{points.threshold.max():.2f} ({settings.get('repetitions', '?')} random draws per configuration). "
        f"FAR and FRR are computed over all validation utterance pairs; accuracy and F1 are shown at k = {k}.",
        "",
        *figures("analysis/operating_point.png", "Accuracy, F1 and rejection rate versus the decision threshold"),
    ]
    for metric in SUMMARY_METRICS:
        values = sweep[sweep.k == k].pivot_table(index="threshold", columns="case", values=f"{metric}_mean")
        rows = []
        for _, p in points.sort_values("threshold").iterrows():
            tau = p.threshold
            is_calibrated = abs(tau - calibrated) < 5e-5
            if not (is_calibrated or abs(tau * 10 - round(tau * 10)) < 1e-6):
                continue
            nearest = values.index[abs(values.index - tau).argmin()]
            row = [f"{tau:.3f}" + (" (calibrated)" if is_calibrated else "")]
            row += [_pct(p[c]) for c, _ in rates] + [f"{values.loc[nearest, c]:.3f}" for c in cases]
            rows.append(row)
        lines += [
            f"### {METRIC_NAMES[metric]} at k = {k}",
            "",
            markdown_table(header, rows, ["right"] * len(header)),
            "",
        ]
    lines += _threshold_orderings(sweep, cases, calibrated)
    return lines


def _threshold_orderings(sweep: pd.DataFrame, cases, calibrated: float) -> list[str]:
    """Thresholds at which each ordering holds at every k, per summary metric."""
    rows, failed_sizes, margins = [], [], []
    for better, worse in (("optimal", "random"), ("random", "bad"), ("high_p", "low_p")):
        if better not in cases or worse not in cases:
            continue
        cells = []
        for metric in SUMMARY_METRICS:
            failures = []
            for tau in sorted(sweep.threshold.unique()):
                group = sweep[sweep.threshold == tau]
                missing = sorted(
                    set(int(x) for x in group.k.unique()) - set(ordering_holds(group, better, worse, metric))
                )
                if missing:
                    failures.append(_tau(tau, calibrated))
                    means = group.pivot_table(index="k", columns="case", values=f"{metric}_mean")
                    failed_sizes += missing
                    margins += [means.loc[size, worse] - means.loc[size, better] for size in missing]
            cells.append("all τ" if not failures else "all τ except " + ", ".join(failures))
        rows.append([f"{CASE_NAMES[better]} > {CASE_NAMES[worse]}"] + cells)
    if not rows:
        return []
    lines = [
        "### Orderings",
        "",
        "Decision thresholds at which the first configuration has the higher mean at every k.",
        "",
        markdown_table(["Ordering"] + [METRIC_NAMES[m] for m in SUMMARY_METRICS], rows, "left"),
        "",
    ]
    if failed_sizes:
        lines += [
            f"- Where an ordering fails, it fails at k ≥ {min(failed_sizes)} only and by at most {max(margins):.3f}.",
            "",
        ]
    return lines


def _ordering_label(group: pd.DataFrame, orderings, metric: str) -> str:
    ks = [int(x) for x in sorted(group.k.unique())]
    holds = set(ks)
    for better, worse in orderings:
        holds &= set(ordering_holds(group, better, worse, metric))
    return "all k" if sorted(holds) == ks else "none" if not holds else f"k = {ranges(holds)}"


def _design_table(summary: pd.DataFrame, keys, cases, proposed, orderings, ordering_name, metric, defaults) -> str:
    """Mean ``metric`` over the enrollment sizes for every design, best first."""
    means = mean_over_sizes(summary, keys)
    values = means.pivot_table(index=keys, columns="case", values=f"{metric}_mean")
    rejected = means[means.case == proposed].set_index(keys).rejection_rate_mean
    rows = []
    for key in values.sort_values(proposed, ascending=False).index:
        key = key if isinstance(key, tuple) else (key,)
        group = summary
        for column, value in zip(keys, key):
            group = group[group[column] == value]
        cells = [f"`{v}`" for v in key] + [f"{values.loc[key, c]:.4f}" for c in cases]
        cells += [_pct(rejected.loc[key]), _ordering_label(group, orderings, metric)]
        if all(v == defaults[column] for column, v in zip(keys, key)):
            cells = [f"**{c}**" for c in cells]
        rows.append(cells)
    header = [k.replace("_", " ").capitalize() for k in keys] + [CASE_NAMES[c] for c in cases]
    header += [f"Rejected ({CASE_NAMES[proposed]})", ordering_name]
    return markdown_table(header, rows, ["left"] * len(keys) + ["right"] * (len(cases) + 1) + ["left"])


def _design_choices(root: Path, config) -> list[str]:
    defaults = {
        "candidate_rule": config.similarity_selection.candidate_rule,
        "scoring": config.identification.scoring,
        "quality_check": config.phoneme_selection.quality_check,
    }
    settings = _read_json(root / "analysis" / "settings.json")
    keys = ["candidate_rule", "scoring"]
    lines, sizes = [], set()
    path = root / "analysis" / "design_choices_icpram2025.csv"
    if path.exists():
        summary = pd.read_csv(path)
        sizes |= set(summary.k)
        lines += ["### ICPRAM 2025 experiment", ""]
        for metric in SUMMARY_METRICS:
            table = _design_table(
                summary,
                keys,
                ("bad", "random", "optimal"),
                "optimal",
                (("optimal", "random"), ("random", "bad")),
                "Optimal > Random > Bad",
                metric,
                defaults,
            )
            lines += [f"#### {METRIC_NAMES[metric]}", "", table, ""]
    path = root / "analysis" / "design_choices_pricai2025.csv"
    if path.exists():
        summary = pd.read_csv(path)
        sizes |= set(summary.k)
        checks = sorted(summary.quality_check.unique())
        selected = defaults["quality_check"] if defaults["quality_check"] in checks else checks[0]
        lines += ["### PRICAI 2025 experiment", ""]
        for metric in SUMMARY_METRICS:
            table = _design_table(
                summary[summary.quality_check == selected],
                keys,
                ("low_p", "high_p"),
                "high_p",
                (("high_p", "low_p"),),
                "High-P > Low-P",
                metric,
                defaults,
            )
            lines += [f"#### {METRIC_NAMES[metric]}", "", table, ""]
        if len(checks) > 1:
            high_p = summary[summary.case == "high_p"]
            spreads = {
                m: high_p.groupby(["candidate_rule", "scoring", "k"])[f"{m}_mean"].agg(lambda x: x.max() - x.min())
                for m in SUMMARY_METRICS
            }
            others = ", ".join(f"`{c}`" for c in checks if c != selected)
            text = (
                f"Shown with the `{selected}` quality check. With {others}, the High-P mean differs by at most "
                + " and ".join(f"{spreads[m].max():.4f} in {_metric(m)}" for m in SUMMARY_METRICS)
                + " at any k"
            )
            rules = sorted(set(high_p.candidate_rule))
            same = [
                f"`{rule}`"
                for rule in rules
                if all(spreads[m].groupby(level="candidate_rule").max()[rule] == 0 for m in SUMMARY_METRICS)
            ]
            if same:
                text += f", and not at all with the {', '.join(same)} candidate rule"
            lines += [text + ".", ""]
    if not lines:
        return []
    intro = (
        f"Mean accuracy and F1 over k = {ranges(sizes)} at the calibrated threshold for every combination of "
        f"candidate rule and speaker scoring rule ({settings.get('repetitions', '?')} random draws per "
        "configuration), best first; the configuration in use is in bold. Rejected: share of queries rejected "
        "as unknown, mean over k. The last column lists the enrollment sizes k at which the ordering holds. "
        "Centroid and maximum scores are higher than the mean pairwise cosine, so fewer queries fall below the "
        "threshold and are rejected."
    )
    return ["## Design choices", "", intro, "", *lines]


def _smallest_size(summary: pd.DataFrame, case: str) -> str:
    """Finding: smallest k within 0.005 of the best mean of ``case``, per summary metric."""
    sizes = " and ".join(f"k = {recommended_size(summary, case, m)} in {_metric(m)}" for m in SUMMARY_METRICS)
    return f"- {CASE_NAMES[case]} comes within 0.005 of its best mean at {sizes}."


def _selection_diagnostics(summary: pd.DataFrame, cases, n_speakers: int) -> list[str]:
    """Rejection rate and share of enrollment utterances added by the fallback rule."""
    rejection, fallback = [], []
    for case in cases:
        c = summary[summary.case == case]
        if c.empty:
            continue
        rejection.append(f"{CASE_NAMES[case]} {_pct(c.rejection_rate_mean.mean())}")
        if "fallback_mean" in c and case != "random":  # random draws have no constraint
            share = c.fallback_mean.sum() / max(n_speakers * c.k.sum(), 1)
            fallback.append(f"{CASE_NAMES[case]} {_pct(share, 1)}")
    lines = ["- Queries rejected as unknown, mean over k: " + ", ".join(rejection) + "."]
    if fallback and any(not part.endswith(" 0.0%") for part in fallback):
        lines.append(
            "- Enrollment utterances added by the fallback rule because no subset satisfies the pairwise "
            "constraint: " + ", ".join(fallback) + "."
        )
    return lines


def _repetitions_used(root: Path) -> int:
    path = root / "icpram2025" / "runs.csv"
    return int(pd.read_csv(path, usecols=["repetition"]).repetition.max() + 1) if path.exists() else 0


# --------------------------------------------------------------------------- report
def write_report(config, ws, output: str | Path | None = None) -> Path:
    """Write the report of the run in ``ws``; returns its path."""
    root = ws.root
    threshold = json.loads((root / "threshold.json").read_text(encoding="utf-8"))
    info = _read_json(root / "run_info.json")
    path = Path(output) if output else root / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = _Figures(root, path, external=output is not None)
    manifest = pd.read_csv(root / "manifest.csv", usecols=["speaker", "subset"])
    n_speakers = manifest.speaker[manifest.subset == "test"].nunique()
    generated = time.strftime("%Y-%m-%d")
    lines = [
        "# Reproduced results",
        "",
        f"Results reproduced with this repository (speaker-enrollment {__version__}) using "
        f"`{getattr(config, 'source', None) or 'the default configuration'}`; generated on {generated}.",
        "",
        *_setup(config, root, threshold, info),
        *_icpram(root, figures, n_speakers),
        *_pricai(root, figures, n_speakers),
        *_operating_point(root, figures, float(threshold["threshold"])),
        *_design_choices(root, config),
    ]
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
