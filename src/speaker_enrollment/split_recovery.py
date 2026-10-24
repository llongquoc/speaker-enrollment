"""Recovery of a speaker encoder's training speakers from its classifier weights.

A model trained with an AAM-softmax (ArcFace) objective keeps one weight vector per
training speaker.  Embeddings of a training speaker's utterances lie close to that
speaker's weight vector, whereas speakers absent from training have no such
counterpart.  Matching every class to a distinct corpus speaker (maximum-similarity
assignment, Hungarian algorithm) therefore identifies the training speakers;
assignments with a cosine below ``min_similarity`` are chance-level and rejected.

The speakers that are neither training speakers nor test speakers form the
validation set used to place the decision threshold.  This module documents how the
validation list of the released protocol was derived; it is not needed to run the
evaluation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def choose_reference_utterances(
    index: pd.DataFrame, per_speaker: int = 3, target_seconds: float = 6.0, min_seconds: float = 2.0
) -> pd.DataFrame:
    """Per speaker, the ``per_speaker`` utterances whose duration is closest to
    ``target_seconds``, preferring utterances of at least ``min_seconds``."""
    d = index.assign(_long=index.approx_seconds >= min_seconds, _distance=(index.approx_seconds - target_seconds).abs())
    d = d.sort_values(["speaker", "_long", "_distance", "utterance"], ascending=[True, False, True, True])
    return d.groupby("speaker", sort=True).head(per_speaker).drop(columns=["_long", "_distance"])


def assign_classes(similarity: np.ndarray) -> np.ndarray:
    """One-to-one assignment of classes (columns) to speakers (rows) maximising the
    summed similarity.  Returns, per speaker, the class index or -1."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(similarity, maximize=True)
    assignment = np.full(similarity.shape[0], -1)
    assignment[rows] = cols
    return assignment


@dataclass
class SplitRecovery:
    table: pd.DataFrame  # one row per speaker
    summary: dict

    def speakers(self, role: str) -> list[str]:
        return sorted(self.table.speaker[self.table.role == role])


def recover_split(
    speakers: list[str],
    speaker_embeddings: np.ndarray,
    class_weights: np.ndarray,
    test_speakers,
    min_similarity: float = 0.25,
) -> SplitRecovery:
    """Assign every speaker the role ``train``, ``validation`` or ``test``.

    ``speaker_embeddings[i]`` is the L2-normalised mean embedding of speaker
    ``speakers[i]``; ``class_weights`` are the classifier weights (n_classes, dim).
    Test speakers keep the role ``test``; ``table.trained`` records whether a test
    speaker also matched a class.
    """
    weights = class_weights / np.linalg.norm(class_weights, axis=1, keepdims=True)
    sim = np.asarray(speaker_embeddings) @ weights.T
    best_class, best_similarity = sim.argmax(1), sim.max(1)
    assignment = assign_classes(sim)
    assigned = assignment >= 0
    assigned_similarity = np.where(assigned, sim[np.arange(len(speakers)), np.maximum(assignment, 0)], np.nan)
    trained = assigned & (assigned_similarity >= min_similarity)
    spk = np.asarray(speakers)
    is_test = np.isin(spk, list(test_speakers))
    role = np.where(is_test, "test", np.where(trained, "train", "validation"))
    table = pd.DataFrame(
        {
            "speaker": speakers,
            "role": role,
            "trained": trained,
            "class_index": np.where(trained, assignment, -1),
            "class_similarity": np.where(trained, assigned_similarity, np.nan),
            "nearest_class": best_class,
            "nearest_class_similarity": best_similarity,
        }
    )
    unmatched = sorted(set(range(weights.shape[0])) - set(assignment[trained].tolist()))
    rejected = np.sort(assigned_similarity[assigned & ~trained])
    accepted = assigned_similarity[trained]
    summary = {
        "n_speakers": len(speakers),
        "n_classes": int(weights.shape[0]),
        "min_similarity": min_similarity,
        "n_train": int((role == "train").sum()),
        "n_validation": int((role == "validation").sum()),
        "n_test": int(is_test.sum()),
        "classes_without_speaker": unmatched,
        "accepted_similarity": {"min": float(accepted.min()), "median": float(np.median(accepted))}
        if trained.any()
        else None,
        "rejected_assignment_similarity_max": float(rejected.max()) if len(rejected) else None,
        "argmax_agreement": float(np.mean(best_class[trained] == assignment[trained])) if trained.any() else None,
        "test_speakers_matching_a_class": sorted(spk[is_test & trained].tolist()),
    }
    return SplitRecovery(table, summary)
