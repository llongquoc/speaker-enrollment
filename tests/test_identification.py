import numpy as np
import pytest

from speaker_enrollment.enrollment import Enroller, SpeakerIdentifier
from speaker_enrollment.identification import (
    UNKNOWN,
    decide,
    evaluate_enrollment,
    identification_metrics,
    speaker_scores,
)


def test_rejection_lowers_recall_not_precision():
    m = identification_metrics(np.array([0, 0, 1, 1]), np.array([0, UNKNOWN, 1, 1]), 2)
    assert m["precision"] == 1.0 and m["recall"] == 0.75 and m["accuracy"] == 0.75 and m["rejection_rate"] == 0.25


def test_metrics_match_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 7, 300)
        pred = np.where(rng.random(300) < 0.6, y, rng.integers(-1, 7, 300))
        p, r, f, _ = metrics.precision_recall_fscore_support(
            y, pred, labels=np.arange(8), average="macro", zero_division=0
        )
        m = identification_metrics(y, pred, 8)
        assert np.allclose([m["precision"], m["recall"], m["f1"]], [p, r, f])


def test_scoring_rules():
    x = np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)
    q = np.array([[1, 0]], dtype=np.float32)
    enroll = [np.array([0, 1]), np.array([2])]
    assert np.allclose(speaker_scores(q, x, enroll, "mean"), [[0.8, 0.0]])
    assert np.allclose(speaker_scores(q, x, enroll, "max"), [[1.0, 0.0]])
    c = np.array([0.8, 0.4]) / np.linalg.norm([0.8, 0.4])
    assert np.allclose(speaker_scores(q, x, enroll, "centroid"), [[c[0], 0.0]])
    assert decide(np.array([[0.1, 0.3], [0.9, 0.2]]), 0.5).tolist() == [UNKNOWN, 0]
    with pytest.raises(ValueError):
        speaker_scores(q, x, enroll, "median")


def test_evaluate_enrollment():
    x = np.array([[1, 0], [1, 0.05], [0.6, 0.8], [0, 1], [0.05, 1]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([0, 0, 0, 1, 1])
    results = evaluate_enrollment(x, y, [np.array([0]), np.array([3])], 0.7, rules=("mean", "centroid", "max"))
    # queries: row 1 -> speaker 0; row 2 scores 0.6 (speaker 0) vs 0.8 (speaker 1) -> wrong; row 4 -> speaker 1
    assert all(r["n_queries"] == 3 and abs(r["accuracy"] - 2 / 3) < 1e-9 for r in results.values())


def test_online_enrollment_and_identification():
    enroller = Enroller(threshold=0.8, target_size=2)
    assert enroller.offer([1, 0]) and not enroller.offer([0, 1]) and enroller.offer([1, 0.1]) and enroller.complete
    identifier = SpeakerIdentifier(threshold=0.8)
    identifier.enroll("a", enroller.profile())
    identifier.enroll("b", [[0, 1]])
    assert identifier.identify([1, 0.05])[0] == "a"
    assert identifier.identify([0.7, 0.7])[0] is None
    prompts = ["p1", "p2", "p3"]
    recorded = iter([[1, 0], [0, 1], [0.99, 0.1], [1, 0.05]])
    profile = Enroller(0.8, target_size=2).run(prompts, record=lambda p: next(recorded), embed=lambda w: w)
    assert profile.shape == (2, 2)
