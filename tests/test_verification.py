import numpy as np
import pytest
from scipy.stats import norm

from speaker_enrollment.verification import (
    all_pair_scores,
    calibrate_threshold,
    eer_point,
    far_frr,
    far_point,
    trial_scores,
)


def test_eer_of_gaussian_scores():
    rng = np.random.default_rng(0)
    point = eer_point(rng.normal(1, 1, 200_000), rng.normal(-1, 1, 200_000))
    assert abs(point.eer - norm.cdf(-1)) < 0.005 and abs(point.threshold) < 0.02
    assert abs(point.far - point.frr) < 0.001


def test_far_point_is_lowest_threshold_meeting_target():
    rng = np.random.default_rng(1)
    target, impostor = np.sort(rng.normal(1, 1, 5000)), np.sort(rng.normal(-1, 1, 20000))
    point = far_point(target, impostor, 0.01)
    assert point.far <= 0.01
    assert far_frr(target, impostor, np.nextafter(point.threshold, -np.inf) - 1e-9)[0] > 0.01


def test_all_pair_counts():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), [3, 4, 5, 6])
    target, impostor = all_pair_scores(rng.normal(size=(18, 8)), labels, block=5)
    assert len(target) == 3 + 6 + 10 + 15 and len(target) + len(impostor) == 18 * 17 // 2


def test_trial_scores_skip_unknown_utterances():
    x = np.eye(3, dtype=np.float32)
    target, impostor, n = trial_scores(np.array([1, 0, 1]), ["a", "a", "z"], ["a", "b", "a"], ["a", "b", "c"], x)
    assert n == 2 and target.tolist() == [1.0] and impostor.tolist() == [0.0]


def test_calibration_rules(embeddings):
    eer = calibrate_threshold(embeddings.vectors, embeddings.labels, "eer")
    far = calibrate_threshold(embeddings.vectors, embeddings.labels, "far", target_far=0.001)
    fixed = calibrate_threshold(embeddings.vectors, embeddings.labels, "fixed", value=0.5)
    assert far.threshold > eer.threshold and far.far <= 0.001 and fixed.threshold == 0.5
    with pytest.raises(ValueError):
        calibrate_threshold(embeddings.vectors, embeddings.labels, "far")
    with pytest.raises(ValueError):
        eer_point([], [0.1])
