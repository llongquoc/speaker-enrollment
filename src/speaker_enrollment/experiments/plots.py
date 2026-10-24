"""Figures: metrics versus enrollment size, metrics versus phoneme coverage (PRICAI
2025, Fig. 3), speaker embeddings (ICPRAM 2025, Fig. 1) and the operating-point analysis.

Each configuration keeps one colour and one marker in every figure (validated categorical
palette; the marker is a second, colour-independent encoding)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .tables import CASE_NAMES  # noqa: E402

CASE_STYLE = {
    "optimal": ("#2a78d6", "o"),
    "random": ("#eb6834", "s"),
    "bad": ("#1baf7a", "^"),
    "high_p": ("#eda100", "D"),
    "low_p": ("#e87ba4", "v"),
}
# Categorical order of the palette, with a marker per slot for series that are not adjacent.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _legend_above(fig, handles, labels, ncol: int) -> None:
    """Legend centred above the axes, filled row by row."""
    order = [i for column in range(ncol) for i in range(column, len(handles), ncol)]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT,
        columnspacing=1.4,
        handlelength=2.4,
    )


def _save(fig, path, top: float = 0.9) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def metrics_vs_size(summary: pd.DataFrame, cases, metrics, path: str | Path) -> None:
    """One panel per metric, on a common scale: mean (band: ± one standard deviation over
    the random draws) against the enrollment size k."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.9), sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        _style(ax)
        for case in cases:
            s = summary[summary.case == case].sort_values("k")
            if s.empty:
                continue
            color, marker = CASE_STYLE[case]
            mean = s[f"{metric}_mean"].to_numpy()
            ax.plot(s.k, mean, color=color, marker=marker, markersize=5, linewidth=2, label=CASE_NAMES[case])
            std = s.get(f"{metric}_std")
            if std is not None and np.nanmax(std.to_numpy()) > 0:
                ax.fill_between(s.k, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_xticks(sorted(summary.k.unique()))
        ax.set_xlabel("Enrollment utterances per speaker (k)", color=TEXT, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], color=TEXT, fontsize=10, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=len(labels))
    _save(fig, path, top=0.9)


def metrics_vs_coverage(summary: pd.DataFrame, path: str | Path) -> None:
    """Each metric against the mean phoneme coverage P of the enrollment sets (PRICAI
    2025, Fig. 3): one panel per metric, one line per configuration in order of k, and
    the Pearson correlation over both configurations."""
    cases = [c for c in ("high_p", "low_p") if (summary.case == c).any()]
    points = summary[summary.case.isin(cases)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        _style(ax)
        for case in cases:
            s = summary[summary.case == case].sort_values("k")
            color, marker = CASE_STYLE[case]
            ax.plot(
                s.coverage_mean,
                s[f"{metric}_mean"],
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
                label=CASE_NAMES[case],
            )
        name = METRIC_LABELS[metric]
        if len(points) > 2:
            r = np.corrcoef(points.coverage_mean, points[f"{metric}_mean"])[0, 1]
            name += f"  (r = {r:.2f})"
        ax.set_title(name, color=TEXT, fontsize=9.5, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Mean phoneme coverage of the enrollment set (P)", color=TEXT, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=len(cases))
    _save(fig, path, top=0.94)


def embedding_pca(
    vectors: np.ndarray, labels: np.ndarray, names: list[str], path: str | Path, n_speakers: int = 7, seed: int = 0
) -> None:
    """3-D principal-component view of the embeddings of randomly chosen speakers
    (ICPRAM 2025, Fig. 1)."""
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(names), size=min(n_speakers, len(names), len(SERIES_COLORS)), replace=False))
    mask = np.isin(labels, chosen)
    x = vectors[mask] - vectors[mask].mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[:3].T
    fig = plt.figure(figsize=(7.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    for slot, s in enumerate(chosen):
        sel = labels[mask] == s
        ax.scatter(
            *z[sel].T,
            s=9,
            color=SERIES_COLORS[slot],
            marker=SERIES_MARKERS[slot],
            alpha=0.85,
            linewidths=0,
            label=names[s],
        )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for setter, label in ((ax.set_xlabel, "PC 1"), (ax.set_ylabel, "PC 2"), (ax.set_zlabel, "PC 3")):
        setter(label, fontsize=8.5, color=TEXT)
    fig.legend(
        loc="center right",
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        markerscale=1.6,
        title="Speaker",
        title_fontsize=8.5,
    )
    fig.subplots_adjust(left=0.0, right=0.82, bottom=0.04, top=0.98)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def operating_point(sweep: pd.DataFrame, path: str | Path, k: int, threshold: float | None = None) -> None:
    """Accuracy, F1 and rejection rate at enrollment size ``k`` against the decision
    threshold, for every configuration; ``threshold`` marks the calibrated value."""
    s = sweep[sweep.k == k]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.4), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]})
    for ax, column, label in (
        (axes[0], "accuracy_mean", f"Accuracy at k = {k}"),
        (axes[1], "f1_mean", f"F1 at k = {k}"),
        (axes[2], "rejection_rate_mean", "Queries rejected"),
    ):
        _style(ax)
        for case in ("optimal", "random", "bad", "high_p", "low_p"):
            c = s[s.case == case].sort_values("threshold")
            if c.empty:
                continue
            color, marker = CASE_STYLE[case]
            ax.plot(
                c.threshold, c[column], color=color, marker=marker, markersize=4, linewidth=2, label=CASE_NAMES[case]
            )
        if threshold is not None:
            ax.axvline(
                threshold,
                color=MUTED,
                linewidth=1.2,
                linestyle=(0, (5, 3)),
                zorder=1,
                label=f"calibrated τ = {threshold:.3f}",
            )
        ax.set_ylabel(label, color=TEXT)
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[2].set_xlabel("Decision threshold τ (cosine)", color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=3)
    _save(fig, path, top=0.94)
