"""Phoneme-based enrollment selection (PRICAI 2025, Fig. 2).

A speaker's candidate utterances are sorted by phoneme coverage (number of distinct
phoneme units in the transcript) and taken in that order; an utterance enters the
enrollment set only if it passes a similarity quality check against the utterances
already enrolled.  ``descending`` order is the proposed method (High-P);
``ascending`` order is the Low-P control.

Quality checks for a candidate ``c`` and enrolled set ``E``:

``pairwise``  cos(c, e) >= tau for every e in E (as in ICPRAM 2025; default)
``mean``      mean cos(c, e) over E >= tau
"""

from __future__ import annotations

import numpy as np

from .base import Selection

QUALITY_CHECKS = ("pairwise", "mean")
ORDERS = ("descending", "ascending")


def select_by_coverage(
    sim: np.ndarray,
    coverage: np.ndarray,
    k: int,
    tau: float,
    order: str = "descending",
    candidates: np.ndarray | None = None,
    quality_check: str = "pairwise",
) -> Selection:
    """Deterministic coverage-ordered selection for one speaker.

    ``candidates`` restricts the utterances considered (default: all).  If fewer than
    k candidates pass the quality check, the set is completed with the remaining
    utterances in coverage order (reported in ``Selection.fallback``).
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; choose from {ORDERS}")
    if quality_check not in QUALITY_CHECKS:
        raise ValueError(f"unknown quality check {quality_check!r}; choose from {QUALITY_CHECKS}")
    n = len(sim)
    k = min(k, n)
    coverage = np.asarray(coverage)
    sign = -1 if order == "descending" else 1
    pool = np.arange(n) if candidates is None else np.asarray(candidates, dtype=int)
    ranked = pool[np.argsort(sign * coverage[pool], kind="stable")]
    chosen: list[int] = []
    for c in ranked:
        if chosen:
            s = sim[c, chosen]
            if not (np.all(s >= tau) if quality_check == "pairwise" else s.mean() >= tau):
                continue
        chosen.append(int(c))
        if len(chosen) == k:
            return Selection(np.array(chosen))
    remaining = [int(i) for i in np.argsort(sign * coverage, kind="stable") if int(i) not in chosen]
    fallback = k - len(chosen)
    chosen += remaining[:fallback]
    return Selection(np.array(chosen), fallback)
