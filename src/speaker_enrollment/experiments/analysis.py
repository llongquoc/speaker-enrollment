"""Operating-point and design-choice analysis of both experiments.

* **Operating point** — both experiments repeated over a range of decision thresholds,
  with the false-acceptance and false-rejection rates of all validation and test
  utterance pairs at each threshold.
* **Design choices** — every combination of candidate rule, speaker scoring rule and
  quality check at the calibrated threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..embedding.extract import EmbeddingSet
from ..verification import all_pair_scores, far_frr
from .evaluation import METRICS, SpeakerSimilarities, run_phoneme_selection, run_similarity_selection, summarize

log = logging.getLogger(__name__)


def threshold_sweep(
    test: EmbeddingSet,
    thresholds,
    coverage: np.ndarray | None,
    phoneme_sets: list[set] | None,
    candidate_rule: str = "clique",
    scoring: str = "mean",
    quality_check: str = "pairwise",
    sizes=range(1, 11),
    repetitions: int = 10,
    seed: int = 0,
    validation: EmbeddingSet | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both experiments at every threshold in ``thresholds``.

    Returns the summary of all configurations (with a ``threshold`` column) and the
    operating points: FAR and FRR over all validation and test utterance pairs.
    """
    pair_scores = {}
    for name, emb in (("validation", validation), ("test", test)):
        if emb is not None:
            target, impostor = all_pair_scores(emb.vectors, emb.labels)
            pair_scores[name] = (np.sort(target), np.sort(impostor))
    summaries, points = [], []
    for tau in thresholds:
        tau = float(tau)
        sims = SpeakerSimilarities(test, tau)
        runs = [
            run_similarity_selection(
                test,
                tau,
                sizes=sizes,
                repetitions=repetitions,
                candidate_rules=(candidate_rule,),
                scorings=(scoring,),
                seed=seed,
                similarities=sims,
            )
        ]
        if coverage is not None and phoneme_sets is not None:
            runs.append(
                run_phoneme_selection(
                    test,
                    tau,
                    coverage,
                    phoneme_sets,
                    sizes,
                    (candidate_rule,),
                    (quality_check,),
                    (scoring,),
                    similarities=sims,
                )
            )
        summaries.append(summarize(pd.concat(runs, ignore_index=True)))
        point = {"threshold": tau}
        for name, (target, impostor) in pair_scores.items():
            point[f"{name}_far"], point[f"{name}_frr"] = far_frr(target, impostor, tau)
        points.append(point)
        log.info("operating point τ = %.3f evaluated", tau)
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(points)


def design_choices(
    test: EmbeddingSet,
    threshold: float,
    coverage: np.ndarray | None,
    phoneme_sets: list[set] | None,
    candidate_rules,
    scorings,
    quality_checks,
    sizes=range(1, 11),
    repetitions: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Summaries of both experiments for every combination of candidate rule and
    scoring rule (and, for PRICAI 2025, quality check) at a fixed threshold."""
    sims = SpeakerSimilarities(test, threshold)
    icpram = summarize(
        run_similarity_selection(
            test,
            threshold,
            sizes=sizes,
            repetitions=repetitions,
            candidate_rules=tuple(candidate_rules),
            scorings=tuple(scorings),
            seed=seed,
            similarities=sims,
        )
    )
    pricai = None
    if coverage is not None and phoneme_sets is not None:
        pricai = summarize(
            run_phoneme_selection(
                test,
                threshold,
                coverage,
                phoneme_sets,
                sizes,
                tuple(candidate_rules),
                tuple(quality_checks),
                tuple(scorings),
                similarities=sims,
            )
        )
    return icpram, pricai


def mean_over_sizes(summary: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean of each metric and of the rejection rate over the enrollment sizes, one
    row per configuration in ``by`` and case."""
    values = [f"{m}_mean" for m in METRICS] + ["rejection_rate_mean"]
    return summary.groupby([*by, "case"], dropna=False)[values].mean().reset_index()
