"""Pairwise cosine scoring and the positive (decision) threshold.

Two utterances are accepted as the same speaker when their cosine similarity is at
least the threshold, so ``FAR(t) = P(impostor score >= t)`` and
``FRR(t) = P(target score < t)``.  The papers place the threshold at the equal error
rate (FAR = FRR) of a validation set of speakers unseen in training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

THRESHOLD_RULES = ("eer", "far", "fixed")


@dataclass
class OperatingPoint:
    threshold: float
    far: float
    frr: float
    n_target: int
    n_impostor: int

    @property
    def eer(self) -> float:
        return 0.5 * (self.far + self.frr)

    def as_dict(self) -> dict:
        return {**asdict(self), "eer": self.eer}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def far_frr(target: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """FAR and FRR at ``threshold``; both score arrays must be sorted ascending."""
    frr = np.searchsorted(target, threshold, side="left") / len(target)
    far = (len(impostor) - np.searchsorted(impostor, threshold, side="left")) / len(impostor)
    return float(far), float(frr)


def _sorted_scores(target, impostor) -> tuple[np.ndarray, np.ndarray]:
    target = np.sort(np.asarray(target, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    if not len(target) or not len(impostor):
        raise ValueError(
            f"need target and impostor scores (got {len(target)} target, {len(impostor)} impostor): "
            "use at least two speakers with at least two utterances each"
        )
    return target, impostor


def eer_point(target, impostor, iterations: int = 100) -> OperatingPoint:
    """Equal-error-rate operating point, by bisection on the threshold."""
    target, impostor = _sorted_scores(target, impostor)
    lo, hi = min(target[0], impostor[0]), max(target[-1], impostor[-1]) + 1e-9
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        far, frr = far_frr(target, impostor, mid)
        if far > frr:
            lo = mid
        else:
            hi = mid
    candidates = [(t, *far_frr(target, impostor, t)) for t in (lo, hi)]
    t, far, frr = min(candidates, key=lambda c: abs(c[1] - c[2]))
    return OperatingPoint(float(t), far, frr, len(target), len(impostor))


def far_point(target, impostor, target_far: float) -> OperatingPoint:
    """Lowest threshold whose false-acceptance rate does not exceed ``target_far``."""
    target, impostor = _sorted_scores(target, impostor)
    if not 0 < target_far < 1:
        raise ValueError("target_far must be in (0, 1)")
    k = int(np.floor(target_far * len(impostor)))  # impostor scores allowed at or above t
    t = float(np.nextafter(impostor[len(impostor) - k - 1], np.inf)) if k < len(impostor) else float(impostor[0])
    far, frr = far_frr(target, impostor, t)
    return OperatingPoint(t, far, frr, len(target), len(impostor))


def operating_point(target, impostor, threshold: float) -> OperatingPoint:
    target, impostor = _sorted_scores(target, impostor)
    far, frr = far_frr(target, impostor, threshold)
    return OperatingPoint(float(threshold), far, frr, len(target), len(impostor))


def all_pair_scores(vectors: np.ndarray, labels, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Cosine scores of every utterance pair, split into target and impostor scores."""
    x = l2_normalize(vectors)
    labels = np.asarray(labels)
    n = len(x)
    target, impostor = [], []
    for s in range(0, n, block):
        e = min(n, s + block)
        scores = x[s:e] @ x[s:].T
        rows, cols = np.triu_indices(e - s, 1, m=n - s)
        values = scores[rows, cols]
        same = labels[s + rows] == labels[s + cols]
        target.append(values[same])
        impostor.append(values[~same])
    return np.concatenate(target), np.concatenate(impostor)


def trial_scores(
    labels, a, b, utterances: list[str], vectors: np.ndarray, chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Cosine scores of a trial list; trials with an utterance lacking an embedding are
    skipped.  Returns ``(target, impostor, n_scored)``."""
    index = {u: i for i, u in enumerate(utterances)}
    ia = np.fromiter((index.get(u, -1) for u in a), dtype=np.int64, count=len(a))
    ib = np.fromiter((index.get(u, -1) for u in b), dtype=np.int64, count=len(b))
    keep = (ia >= 0) & (ib >= 0)
    ia, ib, labels = ia[keep], ib[keep], np.asarray(labels)[keep]
    x = l2_normalize(vectors)
    scores = (
        np.concatenate(
            [np.einsum("ij,ij->i", x[ia[i : i + chunk]], x[ib[i : i + chunk]]) for i in range(0, len(ia), chunk)]
        )
        if len(ia)
        else np.zeros(0, np.float32)
    )
    return scores[labels == 1], scores[labels == 0], int(keep.sum())


def calibrate_threshold(
    vectors: np.ndarray, labels, rule: str = "eer", target_far: float | None = None, value: float | None = None
) -> OperatingPoint:
    """Positive threshold from validation embeddings.

    ``eer``    equal-error-rate point (as in both papers)
    ``far``    lowest threshold with FAR <= ``target_far``
    ``fixed``  ``value``; the returned point reports FAR/FRR at that value
    """
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"unknown threshold rule {rule!r}; choose from {THRESHOLD_RULES}")
    target, impostor = all_pair_scores(vectors, labels)
    if rule == "eer":
        return eer_point(target, impostor)
    if rule == "far":
        if target_far is None:
            raise ValueError("threshold rule 'far' needs target_far")
        return far_point(target, impostor, target_far)
    if value is None:
        raise ValueError("threshold rule 'fixed' needs value")
    return operating_point(target, impostor, value)
