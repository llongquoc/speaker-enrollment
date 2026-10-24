"""Closed-set speaker identification with rejection, and its metrics.

Every test utterance that is not part of an enrollment set is a query.  It is scored
against each enrolled speaker and assigned to the best-scoring speaker; if that score
is below the positive threshold, the query is rejected as unknown.  A rejection is a
miss for the true speaker (it lowers recall) but a false alarm for no speaker (it does
not lower precision).

Scoring rules for a query ``q`` and enrollment embeddings ``e_1..e_k``:

``mean``      mean of cos(q, e_i) — on the same scale as the pairwise threshold
              (default)
``centroid``  cos(q, normalised mean of e_i)
``max``       max of cos(q, e_i)
"""

from __future__ import annotations

import numpy as np

UNKNOWN = -1
SCORING_RULES = ("mean", "centroid", "max")


def speaker_scores(
    queries: np.ndarray, vectors: np.ndarray, enrollment: list[np.ndarray], rule: str = "mean"
) -> np.ndarray:
    """(n_queries, n_speakers) scores; ``enrollment[s]`` are row indices of ``vectors``.
    All vectors must be L2-normalised."""
    if rule == "mean":
        return queries @ np.stack([vectors[i].mean(0) for i in enrollment]).T
    if rule == "centroid":
        centroids = np.stack([vectors[i].mean(0) for i in enrollment])
        return queries @ (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)).T
    if rule == "max":
        return np.stack([(queries @ vectors[i].T).max(1) for i in enrollment], axis=1)
    raise ValueError(f"unknown scoring rule {rule!r}; choose from {SCORING_RULES}")


def decide(scores: np.ndarray, threshold: float, reject: bool = True) -> np.ndarray:
    """Best-scoring speaker per query, ``UNKNOWN`` where the best score < threshold."""
    best = scores.argmax(1)
    if reject:
        best = np.where(scores[np.arange(len(best)), best] >= threshold, best, UNKNOWN)
    return best


def identification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_speakers: int) -> dict:
    """Accuracy (rejections count as errors), macro-averaged precision, recall and F1
    over the enrolled speakers (unknown is not a class; empty denominators give 0),
    and the rejection rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    hit = y_true == y_pred
    tp = np.bincount(y_true[hit], minlength=n_speakers)[:n_speakers].astype(float)
    predicted = np.bincount(y_pred[y_pred >= 0], minlength=n_speakers)[:n_speakers].astype(float)
    actual = np.bincount(y_true, minlength=n_speakers)[:n_speakers].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    n = len(y_true)
    return {
        "accuracy": float(hit.mean()) if n else 0.0,
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "rejection_rate": float(np.mean(y_pred == UNKNOWN)) if n else 0.0,
        "n_queries": int(n),
    }


def evaluate_enrollment(
    vectors: np.ndarray,
    labels: np.ndarray,
    enrollment: list[np.ndarray],
    threshold: float,
    rules=("mean",),
    reject: bool = True,
) -> dict[str, dict]:
    """Identification metrics of one enrollment configuration for each scoring rule.

    ``labels`` are speaker codes ``0..S-1``; ``enrollment[s]`` are the enrollment rows
    of speaker ``s``; all other rows are queries.
    """
    enrollment = [np.asarray(i) for i in enrollment]
    enrolled = np.zeros(len(vectors), dtype=bool)
    for rows in enrollment:
        enrolled[rows] = True
    queries, truth = vectors[~enrolled], np.asarray(labels)[~enrolled]
    return {
        rule: identification_metrics(
            truth, decide(speaker_scores(queries, vectors, enrollment, rule), threshold, reject), len(enrollment)
        )
        for rule in rules
    }
