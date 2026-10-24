"""Experiment runners for both papers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..embedding.extract import EmbeddingSet
from ..identification import evaluate_enrollment
from ..selection.phoneme import select_by_coverage
from ..selection.similarity import CASES, candidate_sets, select_enrollment

log = logging.getLogger(__name__)

METRICS = ("accuracy", "precision", "recall", "f1")
CONFIG_COLUMNS = ("candidate_rule", "quality_check", "scoring", "threshold")


class SpeakerSimilarities:
    """Per-speaker similarity matrices and candidate sets for one threshold."""

    def __init__(self, embeddings: EmbeddingSet, threshold: float):
        self.threshold = threshold
        self.groups = embeddings.groups()
        self.matrices = [embeddings.vectors[g] @ embeddings.vectors[g].T for g in self.groups]
        self._candidates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def candidates(self, rule: str) -> list[tuple[np.ndarray, np.ndarray]]:
        if rule not in self._candidates:
            self._candidates[rule] = [candidate_sets(m, self.threshold, rule) for m in self.matrices]
        return self._candidates[rule]

    def candidate_statistics(self, rule: str) -> dict:
        high = np.array([len(c[0]) for c in self.candidates(rule)])
        sizes = np.array([len(g) for g in self.groups])
        return {
            "high_quality_size_min": int(high.min()),
            "high_quality_size_median": float(np.median(high)),
            "high_quality_fraction_mean": float(np.mean(high / sizes)),
            "speakers_with_fewer_than_10_high_quality": int(np.sum(high < 10)),
        }


def _coverage_of(enrollment: list[np.ndarray], phoneme_sets: list[set] | None) -> float:
    if phoneme_sets is None:
        return float("nan")
    return float(np.mean([len(set().union(*(phoneme_sets[i] for i in rows))) for rows in enrollment]))


def run_similarity_selection(
    embeddings: EmbeddingSet,
    threshold: float,
    cases=CASES,
    sizes=range(1, 11),
    repetitions: int = 30,
    candidate_rules=("clique",),
    scorings=("mean",),
    reject: bool = True,
    seed: int = 0,
    phoneme_sets: list[set] | None = None,
    similarities: SpeakerSimilarities | None = None,
) -> pd.DataFrame:
    """ICPRAM 2025 (Sec. 4.2-4.3): bad / random / optimal enrollment for k in
    ``sizes``, ``repetitions`` random draws each.

    Every draw is evaluated with each scoring rule; random draws do not depend on the
    candidate rule and are shared by all rules.  Draws are seeded by
    ``(seed, case, k, repetition)``, so results are reproducible.
    """
    sims = similarities or SpeakerSimilarities(embeddings, threshold)
    rows = []
    for case_index, case in enumerate(CASES):
        if case not in cases:
            continue
        rules = candidate_rules[:1] if case == "random" else candidate_rules
        for rule in rules:
            candidates = None if case == "random" else sims.candidates(rule)
            for k in sizes:
                for repetition in range(repetitions):
                    rng = np.random.default_rng([seed, case_index, k, repetition])
                    enrollment, fallback = [], 0
                    for s, group in enumerate(sims.groups):
                        selection = select_enrollment(
                            sims.matrices[s], k, case, threshold, rng, candidates[s] if candidates else None
                        )
                        enrollment.append(group[selection.indices])
                        fallback += selection.fallback
                    coverage = _coverage_of(enrollment, phoneme_sets)
                    results = evaluate_enrollment(
                        embeddings.vectors, embeddings.labels, enrollment, threshold, scorings, reject
                    )
                    for scoring, metrics in results.items():
                        for rule_out in candidate_rules if case == "random" else (rule,):
                            rows.append(
                                {
                                    "case": case,
                                    "k": k,
                                    "repetition": repetition,
                                    "candidate_rule": rule_out,
                                    "scoring": scoring,
                                    "threshold": threshold,
                                    **metrics,
                                    "fallback": fallback,
                                    "coverage": coverage,
                                }
                            )
            log.debug("%s (%s) done", case, rule)
    return pd.DataFrame(rows)


def run_phoneme_selection(
    embeddings: EmbeddingSet,
    threshold: float,
    coverage: np.ndarray,
    phoneme_sets: list[set],
    sizes=range(1, 11),
    candidate_rules=("clique",),
    quality_checks=("pairwise",),
    scorings=("mean",),
    reject: bool = True,
    similarities: SpeakerSimilarities | None = None,
) -> pd.DataFrame:
    """PRICAI 2025 (Sec. 4): deterministic coverage-ordered selection, descending
    (``high_p``, proposed) and ascending (``low_p``), from each speaker's
    high-quality candidate set."""
    sims = similarities or SpeakerSimilarities(embeddings, threshold)
    rows = []
    for rule in candidate_rules:
        candidates = sims.candidates(rule)
        for quality_check in quality_checks:
            for case, order in (("high_p", "descending"), ("low_p", "ascending")):
                for k in sizes:
                    enrollment, fallback = [], 0
                    for s, group in enumerate(sims.groups):
                        selection = select_by_coverage(
                            sims.matrices[s], coverage[group], k, threshold, order, candidates[s][0], quality_check
                        )
                        enrollment.append(group[selection.indices])
                        fallback += selection.fallback
                    mean_coverage = _coverage_of(enrollment, phoneme_sets)
                    results = evaluate_enrollment(
                        embeddings.vectors, embeddings.labels, enrollment, threshold, scorings, reject
                    )
                    for scoring, metrics in results.items():
                        rows.append(
                            {
                                "case": case,
                                "k": k,
                                "repetition": 0,
                                "candidate_rule": rule,
                                "quality_check": quality_check,
                                "scoring": scoring,
                                "threshold": threshold,
                                **metrics,
                                "fallback": fallback,
                                "coverage": mean_coverage,
                            }
                        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (population, ddof=0) over repetitions."""
    keys = [c for c in CONFIG_COLUMNS if c in runs] + ["case", "k"]
    values = [c for c in (*METRICS, "rejection_rate", "coverage", "fallback") if c in runs]
    grouped = runs.groupby(keys, dropna=False)[values]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std"))
    return summary.reset_index()


def select_config(summary: pd.DataFrame, **config) -> pd.DataFrame:
    """Rows of a summary that match the given configuration values, e.g.
    ``select_config(summary, candidate_rule="clique", scoring="mean", threshold=0.14)``;
    ``None`` values are ignored and thresholds are compared with a tolerance."""
    mask = np.ones(len(summary), dtype=bool)
    for key, value in config.items():
        if key in summary and value is not None:
            column = summary[key]
            mask &= np.isclose(column, value) if key == "threshold" else (column == value).to_numpy()
    return summary[mask]


def recommended_size(summary: pd.DataFrame, case: str, metric: str = "accuracy", tolerance: float = 0.005) -> int:
    """Enrollment-size selection (PRICAI 2025 Sec. 3.3, ICPRAM 2025 Sec. 4.4): the
    smallest k whose mean ``metric`` is within ``tolerance`` of the best k."""
    s = summary[summary.case == case].sort_values("k")
    best = s[f"{metric}_mean"].max()
    return int(s[s[f"{metric}_mean"] >= best - tolerance].k.iloc[0])


def coverage_correlation(summary: pd.DataFrame) -> dict:
    """Pearson correlation between mean phoneme coverage and each metric over the
    High-P and Low-P configurations (PRICAI 2025, Fig. 3)."""
    points = summary[summary.case.isin(["high_p", "low_p"])]
    return {m: float(np.corrcoef(points.coverage_mean, points[f"{m}_mean"])[0, 1]) for m in METRICS}
