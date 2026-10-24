"""Experiment runners, operating-point and design-choice analysis, tables, figures and reports."""

from .analysis import design_choices, mean_over_sizes, threshold_sweep
from .evaluation import (
    METRICS,
    SpeakerSimilarities,
    coverage_correlation,
    recommended_size,
    run_phoneme_selection,
    run_similarity_selection,
    select_config,
    summarize,
)
from .tables import ordering_holds, ordering_table

__all__ = [
    "METRICS",
    "SpeakerSimilarities",
    "coverage_correlation",
    "design_choices",
    "mean_over_sizes",
    "ordering_holds",
    "ordering_table",
    "recommended_size",
    "run_phoneme_selection",
    "run_similarity_selection",
    "select_config",
    "summarize",
    "threshold_sweep",
]
