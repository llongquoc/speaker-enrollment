"""Label-consistency ranking to support manual re-labelling (ICPRAM 2025, Sec. 4.1.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_consistency(embeddings) -> pd.DataFrame:
    """Rank utterances by how much closer they are to their own speaker
    (leave-one-out centroid) than to the nearest other speaker.

    ``embeddings`` is a :class:`speaker_enrollment.embedding.EmbeddingSet`.  Utterances
    with a negative margin are the first candidates for manual review.
    """
    groups = embeddings.groups()
    sums = np.stack([embeddings.vectors[g].sum(0) for g in groups])
    centroids = sums / np.linalg.norm(sums, axis=1, keepdims=True)
    other = embeddings.vectors @ centroids.T
    rows = np.arange(len(embeddings))
    other[rows, embeddings.labels] = -np.inf
    nearest = other.argmax(1)
    leave_one_out = sums[embeddings.labels] - embeddings.vectors
    norms = np.clip(np.linalg.norm(leave_one_out, axis=1, keepdims=True), 1e-12, None)
    own = np.einsum("ij,ij->i", embeddings.vectors, leave_one_out / norms)
    table = pd.DataFrame(
        {
            "utterance": embeddings.utterances,
            "speaker": embeddings.speakers,
            "own_score": own,
            "nearest_other_speaker": [embeddings.speaker_ids[i] for i in nearest],
            "nearest_other_score": other[rows, nearest],
        }
    )
    table["margin"] = table.own_score - table.nearest_other_score
    return table.sort_values("margin").reset_index(drop=True)
