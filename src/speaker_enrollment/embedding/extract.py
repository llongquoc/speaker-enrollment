"""Embedding extraction and storage.

Embeddings are computed on whole utterances, one utterance per forward pass: the
encoder has no length masking, so padding utterances into a batch would change the
result.  Audio decoding runs in a background thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.audio import SAMPLE_RATE, read_audio
from .ecapa_tdnn import ECAPATDNN

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- storage
@dataclass
class EmbeddingSet:
    """L2-normalised utterance embeddings with their speaker labels."""

    utterances: list[str]
    speakers: list[str]
    vectors: np.ndarray

    def __post_init__(self):
        self.utterances = [str(u) for u in self.utterances]
        self.speakers = [str(s) for s in self.speakers]
        v = np.asarray(self.vectors, dtype=np.float32)
        if v.ndim != 2 or len(v) != len(self.utterances) or len(self.speakers) != len(self.utterances):
            raise ValueError("utterances, speakers and vectors must have the same length")
        self.vectors = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        self.speaker_ids = sorted(set(self.speakers))
        code = {s: i for i, s in enumerate(self.speaker_ids)}
        self.labels = np.array([code[s] for s in self.speakers], dtype=int)

    def __len__(self) -> int:
        return len(self.utterances)

    @property
    def n_speakers(self) -> int:
        return len(self.speaker_ids)

    def groups(self) -> list[np.ndarray]:
        """Row indices of each speaker, in the order of ``speaker_ids``."""
        return [np.flatnonzero(self.labels == s) for s in range(self.n_speakers)]

    def select(self, mask) -> EmbeddingSet:
        idx = np.flatnonzero(np.asarray(mask))
        return EmbeddingSet([self.utterances[i] for i in idx], [self.speakers[i] for i in idx], self.vectors[idx])

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            utterances=np.asarray(self.utterances, dtype=str),
            speakers=np.asarray(self.speakers, dtype=str),
            vectors=self.vectors,
        )

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingSet:
        with np.load(path, allow_pickle=False) as z:
            return cls(list(z["utterances"]), list(z["speakers"]), z["vectors"])


# --------------------------------------------------------------------------- devices
def select_device(preference: str = "auto") -> str:
    """``auto`` resolves to CUDA, then Apple MPS, then CPU."""
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def verify_device(model: ECAPATDNN, device: str, locator: str, min_cosine: float = 0.999) -> str:
    """Embed one utterance on ``device`` and on CPU; fall back to CPU if the device
    fails or disagrees (cosine below ``min_cosine``)."""
    if device == "cpu":
        return device
    try:
        reference = extract_embeddings(model.to("cpu"), [locator], "cpu")[0]
        candidate = extract_embeddings(model.to(device), [locator], device)[0]
        cosine = float(np.dot(reference, candidate))
    except Exception as error:  # unsupported operator, driver problem, ...
        log.warning("device %s failed (%s: %s); using CPU", device, type(error).__name__, error)
        model.to("cpu")
        return "cpu"
    if cosine < min_cosine:
        log.warning("device %s disagrees with CPU (cosine %.5f); using CPU", device, cosine)
        model.to("cpu")
        return "cpu"
    log.info("device %s verified against CPU (cosine %.6f)", device, cosine)
    return device


# --------------------------------------------------------------------------- extraction
def _prefetch(locators: list[str], max_seconds: float | None, depth: int = 8):
    """Yield ``(index, waveform, error)`` while audio is decoded in a background thread."""
    items: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        for i, locator in enumerate(locators):
            try:
                wav = read_audio(locator)
                if max_seconds and len(wav) > int(max_seconds * SAMPLE_RATE):
                    n = int(max_seconds * SAMPLE_RATE)
                    start = (len(wav) - n) // 2
                    wav = wav[start : start + n]
                items.put((i, wav, None))
            except Exception as error:
                items.put((i, None, error))
        items.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while (item := items.get()) is not None:
        yield item


MIN_SAMPLES = 400  # one analysis window (25 ms at 16 kHz)


@torch.inference_mode()
def embed_waveform(model: ECAPATDNN, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
    """L2-normalised embedding of one mono 16 kHz waveform (the model must be in eval mode
    on ``device``)."""
    wav = np.asarray(waveform, dtype=np.float32).ravel()
    if not len(wav):
        raise ValueError("empty waveform")
    if len(wav) < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)), mode="wrap")
    x = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0).to(device)
    return F.normalize(model(x).float(), dim=1).cpu().numpy()[0]


@torch.inference_mode()
def extract_embeddings(
    model: ECAPATDNN,
    locators: list[str],
    device: str = "cpu",
    max_seconds: float | None = None,
    log_every: int = 500,
) -> np.ndarray:
    """L2-normalised embeddings, (n_utterances, embedding_dim) float32.

    ``max_seconds`` centre-crops longer utterances (default: whole utterances).
    """
    model.eval()
    out = np.zeros((len(locators), model.embedding_dim), dtype=np.float32)
    for n, (i, wav, error) in enumerate(_prefetch(locators, max_seconds), 1):
        if error is not None:
            raise RuntimeError(f"cannot read {locators[i]}: {error}") from error
        if not len(wav):
            raise RuntimeError(f"empty audio: {locators[i]}")
        out[i] = embed_waveform(model, wav, device)
        if log_every and n % log_every == 0:
            log.info("embedded %d/%d utterances", n, len(locators))
    return out
