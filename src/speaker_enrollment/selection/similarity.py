"""Data-centric enrollment selection (ICPRAM 2025).

Three enrollment configurations are compared:

``bad``      every pair of selected utterances has cosine similarity < tau
``random``   k utterances drawn uniformly (baseline)
``optimal``  every pair of selected utterances has cosine similarity >= tau

At k = 1 no pair exists, so the configurations are also distinguished by where
utterances are drawn from.  A *candidate rule* splits each speaker's utterances into a
high-quality set (``optimal`` draws from it) and a low-quality set (``bad`` draws from
it):

``clique``  the speaker's dominant clique of the graph whose edges join pairs with
            cosine >= tau — the high-similarity cluster of the paper's Fig. 2 (default)
``mean``    utterances whose mean cosine to the speaker's other utterances is >= tau
``all``     no split; only the pairwise constraint applies (k = 1 then equals random)
"""

from __future__ import annotations

import numpy as np

from .base import Selection

CASES = ("bad", "random", "optimal")
CANDIDATE_RULES = ("clique", "mean", "all")


def adjacency(sim: np.ndarray, tau: float) -> np.ndarray:
    adj = sim >= tau
    np.fill_diagonal(adj, False)
    return adj


def dominant_clique(sim: np.ndarray, tau: float) -> np.ndarray:
    """Large clique of the ``cos >= tau`` graph, found greedily.

    Each step adds the candidate with the most neighbours among the remaining
    candidates (ties: highest summed similarity) and keeps only its neighbours.
    The result is maximal: no other utterance is similar to all of its members.
    """
    adj = adjacency(sim, tau)
    candidates = np.ones(len(sim), dtype=bool)
    clique: list[int] = []
    while candidates.any():
        idx = np.flatnonzero(candidates)
        degree = adj[np.ix_(idx, idx)].sum(1)
        strength = sim[np.ix_(idx, idx)].sum(1)
        v = int(idx[np.lexsort((-strength, -degree))[0]])
        clique.append(v)
        candidates &= adj[v]
        candidates[v] = False
    return np.array(sorted(clique), dtype=int)


def mean_similarity(sim: np.ndarray) -> np.ndarray:
    """Mean cosine of each utterance to the speaker's other utterances."""
    n = len(sim)
    return (sim.sum(1) - np.diag(sim)) / max(n - 1, 1)


def candidate_sets(sim: np.ndarray, tau: float, rule: str = "clique") -> tuple[np.ndarray, np.ndarray]:
    """``(high_quality, low_quality)`` row indices of one speaker."""
    n = len(sim)
    if rule == "clique":
        high = dominant_clique(sim, tau)
        return high, np.setdiff1d(np.arange(n), high)
    if rule == "mean":
        q = mean_similarity(sim)
        return np.flatnonzero(q >= tau), np.flatnonzero(q < tau)
    if rule == "all":
        return np.arange(n), np.arange(n)
    raise ValueError(f"unknown candidate rule {rule!r}; choose from {CANDIDATE_RULES}")


def _compatible(row: np.ndarray, chosen: list[int], tau: float, mode: str) -> bool:
    if not chosen:
        return True
    s = row[chosen]
    return bool(np.all(s >= tau)) if mode == "high" else bool(np.all(s < tau))


def constrained_sample(
    sim: np.ndarray, k: int, tau: float, rng: np.random.Generator, mode: str, candidates: np.ndarray, restarts: int = 50
) -> Selection:
    """Random k-subset of ``candidates`` whose pairs all satisfy the constraint
    (``mode="high"``: cos >= tau, ``"low"``: cos < tau).

    Randomised greedy search: a random permutation of the candidates is scanned and an
    utterance is accepted if it satisfies the constraint with every accepted one.  If
    no restart reaches k, the largest set found is completed with the utterances that
    violate the constraint least (reported in ``Selection.fallback``).
    """
    n = len(sim)
    k = min(k, n)
    best: list[int] = []
    for _ in range(restarts if len(candidates) else 0):
        chosen: list[int] = []
        for c in rng.permutation(candidates):
            if _compatible(sim[c], chosen, tau, mode):
                chosen.append(int(c))
                if len(chosen) == k:
                    return Selection(np.array(chosen))
        if len(chosen) > len(best):
            best = chosen
    chosen = list(best)
    fallback = 0
    while len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), chosen)
        if not chosen:
            pick = rest[rng.integers(len(rest))]
        elif mode == "high":
            pick = rest[np.argmax(sim[np.ix_(rest, chosen)].min(1))]
        else:
            pick = rest[np.argmin(sim[np.ix_(rest, chosen)].max(1))]
        chosen.append(int(pick))
        fallback += 1
    return Selection(np.array(chosen), fallback)


def select_enrollment(
    sim: np.ndarray,
    k: int,
    case: str,
    tau: float,
    rng: np.random.Generator,
    candidates: tuple[np.ndarray, np.ndarray] | None = None,
    rule: str = "clique",
) -> Selection:
    """Enrollment set of one speaker for an ICPRAM 2025 configuration.

    ``candidates`` are precomputed ``(high, low)`` sets; they are derived with
    ``rule`` when omitted.
    """
    n = len(sim)
    if case == "random":
        return Selection(rng.choice(n, size=min(k, n), replace=False))
    if case not in ("bad", "optimal"):
        raise ValueError(f"unknown case {case!r}; choose from {CASES}")
    high, low = candidates if candidates is not None else candidate_sets(sim, tau, rule)
    if case == "optimal":
        return constrained_sample(sim, k, tau, rng, "high", high)
    return constrained_sample(sim, k, tau, rng, "low", low)


def violating_pairs(sim: np.ndarray, indices: np.ndarray, tau: float, mode: str) -> int:
    """Number of selected pairs that break the constraint (diagnostics)."""
    sub = sim[np.ix_(indices, indices)]
    iu = np.triu_indices(len(indices), 1)
    return int(np.sum(sub[iu] < tau)) if mode == "high" else int(np.sum(sub[iu] >= tau))
