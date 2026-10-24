"""Enrollment-utterance selection.

* :mod:`.similarity` — data-centric selection by pairwise similarity (ICPRAM 2025)
* :mod:`.phoneme` — phoneme-coverage ordering with a similarity quality check (PRICAI 2025)

All functions operate on one speaker: ``sim`` is that speaker's
utterance-by-utterance cosine-similarity matrix, and returned indices refer to its rows.
"""

from .base import Selection
from .phoneme import ORDERS, QUALITY_CHECKS, select_by_coverage
from .similarity import (
    CANDIDATE_RULES,
    CASES,
    candidate_sets,
    constrained_sample,
    dominant_clique,
    mean_similarity,
    select_enrollment,
    violating_pairs,
)

__all__ = [
    "CANDIDATE_RULES",
    "CASES",
    "ORDERS",
    "QUALITY_CHECKS",
    "Selection",
    "candidate_sets",
    "constrained_sample",
    "dominant_clique",
    "mean_similarity",
    "select_by_coverage",
    "select_enrollment",
    "violating_pairs",
]
