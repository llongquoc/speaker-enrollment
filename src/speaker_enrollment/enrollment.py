"""Deployment-time enrollment and identification.

Implements the interactive procedure of PRICAI 2025 (Fig. 1) and the practical
recommendation of ICPRAM 2025 (Sec. 4.3): present a phoneme-rich prompt, record an
utterance, accept it only if it passes the similarity quality check against the
utterances already enrolled, and repeat until the target enrollment size is reached.

Example::

    enroller = Enroller(threshold=tau, target_size=5)
    profile = enroller.run(prompts, record=record_fn, embed=embed_fn)
    identifier = SpeakerIdentifier(threshold=tau)
    identifier.enroll("alice", profile)
    speaker, score = identifier.identify(embed_fn(record_fn(None)))
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .identification import SCORING_RULES
from .selection.phoneme import QUALITY_CHECKS


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-12)


class Enroller:
    """Quality-gated enrollment of one speaker."""

    def __init__(self, threshold: float, target_size: int = 5, quality_check: str = "pairwise", max_attempts: int = 3):
        if quality_check not in QUALITY_CHECKS:
            raise ValueError(f"quality_check must be one of {QUALITY_CHECKS}")
        self.threshold = threshold
        self.target_size = target_size
        self.quality_check = quality_check
        self.max_attempts = max_attempts
        self.embeddings: list[np.ndarray] = []

    @property
    def complete(self) -> bool:
        return len(self.embeddings) >= self.target_size

    def passes_quality_check(self, embedding) -> bool:
        if not self.embeddings:
            return True
        scores = np.stack(self.embeddings) @ _unit(embedding)
        if self.quality_check == "pairwise":
            return bool(scores.min() >= self.threshold)
        return bool(scores.mean() >= self.threshold)

    def offer(self, embedding) -> bool:
        """Add the utterance if it passes the quality check; return whether it did."""
        accepted = self.passes_quality_check(embedding)
        if accepted and not self.complete:
            self.embeddings.append(_unit(embedding))
        return accepted

    def run(
        self, prompts: list[str], record: Callable[[str], np.ndarray], embed: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Prompt, record and embed until complete; ``record(prompt)`` returns a
        waveform and ``embed(waveform)`` an embedding."""
        for prompt in prompts:
            for _ in range(self.max_attempts):
                if self.offer(embed(record(prompt))):
                    break
            if self.complete:
                break
        return self.profile()

    def profile(self) -> np.ndarray:
        if not self.embeddings:
            raise RuntimeError("no utterance has been enrolled")
        return np.stack(self.embeddings)


class SpeakerIdentifier:
    """Identification with rejection over enrolled speaker profiles."""

    def __init__(self, threshold: float, scoring: str = "mean"):
        if scoring not in SCORING_RULES:
            raise ValueError(f"scoring must be one of {SCORING_RULES}")
        self.threshold = threshold
        self.scoring = scoring
        self.profiles: dict[str, np.ndarray] = {}

    def enroll(self, speaker: str, embeddings) -> None:
        self.profiles[speaker] = np.stack([_unit(e) for e in embeddings])

    def score(self, embedding) -> dict[str, float]:
        q = _unit(embedding)
        scores = {}
        for speaker, profile in self.profiles.items():
            if self.scoring == "mean":
                scores[speaker] = float((profile @ q).mean())
            elif self.scoring == "centroid":
                scores[speaker] = float(_unit(profile.mean(0)) @ q)
            else:
                scores[speaker] = float((profile @ q).max())
        return scores

    def identify(self, embedding) -> tuple[str | None, float]:
        """Best-scoring speaker, or ``None`` if the best score is below the threshold."""
        if not self.profiles:
            raise RuntimeError("no speaker is enrolled")
        scores = self.score(embedding)
        best = max(scores, key=lambda speaker: scores[speaker])
        return (best if scores[best] >= self.threshold else None), scores[best]
