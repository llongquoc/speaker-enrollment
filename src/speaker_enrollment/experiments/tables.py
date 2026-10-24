"""Markdown tables in the layout of the papers, and checks of the orderings they report."""

from __future__ import annotations

import pandas as pd

from .evaluation import METRICS

CASE_NAMES = {
    "bad": "Bad",
    "random": "Random",
    "optimal": "Optimal",
    "low_p": "Low-P",
    "high_p": "High-P",
}
METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
METRIC_ABBREVIATIONS = {"accuracy": "Acc.", "precision": "Prec.", "recall": "Rec.", "f1": "F1"}
EXPERIMENT_CASES = {"icpram2025": ("bad", "random", "optimal"), "pricai2025": ("low_p", "high_p")}

# Orderings examined for each experiment, as (better, worse) configurations.
ORDERINGS = {
    "icpram2025": (("optimal", "random"), ("random", "bad")),
    "pricai2025": (("high_p", "low_p"), ("high_p", "optimal"), ("high_p", "random")),
}


def _numeric(cell: str) -> bool:
    try:
        float(cell.split()[0].replace("%", "").replace(",", ""))
        return True
    except (ValueError, IndexError):
        return False


def markdown_table(header: list[str], rows: list[list[str]], align="auto") -> str:
    """Markdown table.  ``align`` is ``"auto"`` (numeric columns right-aligned, text
    left-aligned), ``"left"``, ``"right"``, or one of ``"left"``/``"right"`` per column."""
    if isinstance(align, str) and align == "auto":
        marks = ["---:" if rows and all(_numeric(r[i]) for r in rows) else "---" for i in range(len(header))]
    else:
        per_column = [align] * len(header) if isinstance(align, str) else list(align)
        marks = ["---:" if a == "right" else "---" for a in per_column]
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "|".join(marks) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]
    )


def ranges(values) -> str:
    """``[1, 2, 3, 5]`` -> ``"1–3, 5"``."""
    values = sorted(int(v) for v in values)
    parts, start = [], None
    for i, v in enumerate(values):
        if start is None:
            start = v
        if i + 1 == len(values) or values[i + 1] != v + 1:
            parts.append(f"{start}" if start == v else f"{start}–{v}")
            start = None
    return ", ".join(parts)


# --------------------------------------------------------------------------- paper layouts
def _mean_std(row, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"


def icpram_table(summary: pd.DataFrame, case: str) -> str:
    """Layout of ICPRAM 2025, Tables 1–3: mean ± standard deviation over the random draws."""
    s = summary[summary.case == case].sort_values("k")
    rows = [[str(int(r.k))] + [_mean_std(r, m) for m in METRICS] for _, r in s.iterrows()]
    return markdown_table(["k"] + [METRIC_NAMES[m] for m in METRICS], rows, ["right"] * (1 + len(METRICS)))


def pricai_methods_table(summary: pd.DataFrame) -> str:
    """Layout of PRICAI 2025, Table 2: Baseline (Random), Similarity (Optimal) and
    Similarity + Phoneme (High-P)."""
    methods = {"random": "Baseline", "optimal": "Similarity", "high_p": "Similarity + Phoneme"}
    header = ["k"] + [f"{name} {METRIC_ABBREVIATIONS[m]}" for name in methods.values() for m in METRICS]
    rows = []
    for k in sorted(summary.k.unique()):
        row = [str(int(k))]
        for case in methods:
            r = summary[(summary.case == case) & (summary.k == k)]
            row += [f"{r[f'{m}_mean'].iloc[0]:.3f}" if len(r) else "–" for m in METRICS]
        rows.append(row)
    return markdown_table(header, rows, ["right"] * len(header))


def pricai_coverage_table(summary: pd.DataFrame) -> str:
    """Layout of PRICAI 2025, Table 3: Low-P and High-P with the mean phoneme coverage P."""
    header = ["k"] + [f"{CASE_NAMES[c]} {h}" for c in ("low_p", "high_p") for h in ("P", "Acc.", "Prec.", "Rec.", "F1")]
    rows = []
    for k in sorted(summary.k.unique()):
        row = [str(int(k))]
        for case in ("low_p", "high_p"):
            r = summary[(summary.case == case) & (summary.k == k)].iloc[0]
            row += [f"{r.coverage_mean:.1f}"] + [f"{r[f'{m}_mean']:.3f}" for m in METRICS]
        rows.append(row)
    return markdown_table(header, rows, ["right"] * len(header))


# --------------------------------------------------------------------------- orderings
def ordering_holds(summary: pd.DataFrame, better: str, worse: str, metric: str = "accuracy") -> list[int]:
    """Enrollment sizes k at which ``better`` has a strictly higher mean ``metric``
    than ``worse``."""
    s = summary.pivot_table(index="k", columns="case", values=f"{metric}_mean")
    if better not in s or worse not in s:
        return []
    return [int(k) for k in s.index[s[better] > s[worse]]]


def ordering_table(summary: pd.DataFrame, experiment: str) -> str:
    """For each ordering of the experiment and each metric, the enrollment sizes k at
    which it holds."""
    ks = sorted(int(k) for k in summary.k.unique())
    rows = []
    for better, worse in ORDERINGS[experiment]:
        if not {better, worse} <= set(summary.case):
            continue
        cells = []
        for metric in METRICS:
            holds = ordering_holds(summary, better, worse, metric)
            cells.append("all k" if holds == ks else "none" if not holds else f"k = {ranges(holds)}")
        rows.append([f"{CASE_NAMES[better]} > {CASE_NAMES[worse]}"] + cells)
    return markdown_table(["Ordering"] + [METRIC_NAMES[m] for m in METRICS], rows, "left")
