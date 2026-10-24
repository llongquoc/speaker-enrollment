import numpy as np
import pandas as pd

from speaker_enrollment.split_recovery import assign_classes, choose_reference_utterances, recover_split


def _setup(n_speakers=60, n_classes=50, extra_classes=3, dim=192, seed=0):
    """Speakers ``train`` own one class each (in shuffled order); ``extra_classes``
    classes belong to no corpus speaker."""
    rng = np.random.default_rng(seed)
    speakers = [f"id{i:05d}" for i in range(n_speakers)]
    unseen = sorted(rng.choice(n_speakers, n_speakers - n_classes, replace=False).tolist())
    train = [i for i in range(n_speakers) if i not in unseen]
    centres = rng.normal(size=(n_speakers, dim))
    order = rng.permutation(len(train))
    weights = np.vstack(
        [
            centres[np.asarray(train)[order]] + 0.3 * rng.normal(size=(n_classes, dim)),
            rng.normal(size=(extra_classes, dim)),
        ]
    )
    emb = centres + 0.6 * rng.normal(size=(n_speakers, dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return speakers, emb, weights, train, unseen


def test_assignment_recovers_training_speakers():
    speakers, emb, weights, train, unseen = _setup(extra_classes=0)
    w = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    assert np.flatnonzero(assign_classes(emb @ w.T) >= 0).tolist() == train


def test_recover_split_rejects_chance_level_assignments():
    speakers, emb, weights, train, unseen = _setup()
    test = {speakers[i] for i in unseen[:4]}
    result = recover_split(speakers, emb, weights, test)
    assert result.speakers("train") == [speakers[i] for i in train]
    assert set(result.speakers("validation")) == {speakers[i] for i in unseen} - test
    assert result.speakers("test") == sorted(test)
    assert result.summary["classes_without_speaker"] == [50, 51, 52]
    assert result.summary["accepted_similarity"]["min"] > 0.25 > result.summary["rejected_assignment_similarity_max"]
    assert result.summary["test_speakers_matching_a_class"] == []


def test_test_speaker_matching_a_class_is_reported():
    speakers, emb, weights, train, unseen = _setup()
    result = recover_split(speakers, emb, weights, {speakers[train[0]]})
    assert result.summary["test_speakers_matching_a_class"] == [speakers[train[0]]]
    assert speakers[train[0]] not in result.speakers("train")


def test_reference_utterances():
    index = pd.DataFrame(
        {
            "utterance": [f"a/{i}.wav" for i in range(5)] + ["b/0.wav"],
            "speaker": ["a"] * 5 + ["b"],
            "approx_seconds": [1.0, 5.5, 6.2, 30.0, 7.0, 1.5],
        }
    )
    chosen = choose_reference_utterances(index, 2)
    assert chosen[chosen.speaker == "a"].utterance.tolist() == ["a/2.wav", "a/1.wav"]
    assert chosen[chosen.speaker == "b"].utterance.tolist() == ["b/0.wav"]
