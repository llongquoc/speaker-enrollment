"""Audio access for corpora stored as a directory tree or as a ZIP archive.

Every utterance is addressed by a *locator*: a file path, or
``"<archive path>::<member name>"`` for a member of a ZIP archive.  Archive members
are decompressed on demand, so a distribution archive can be used without
extracting it.
"""

from __future__ import annotations

import io
import math
import threading
import zipfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
ARCHIVE_SEPARATOR = "::"

_local = threading.local()


def make_locator(archive: str | Path, member: str) -> str:
    """Locator of a member of a ZIP archive."""
    return f"{Path(archive).resolve()}{ARCHIVE_SEPARATOR}{member}"


def split_locator(locator: str) -> tuple[str, str] | None:
    """``(archive, member)`` for an archive locator, ``None`` for a plain file path."""
    if ARCHIVE_SEPARATOR in locator:
        archive, member = locator.split(ARCHIVE_SEPARATOR, 1)
        return archive, member
    return None


def _archive(path: str) -> zipfile.ZipFile:
    """Per-thread cached handle (ZipFile objects are not safe to share across threads)."""
    cache = getattr(_local, "archives", None)
    if cache is None:
        cache = _local.archives = {}
    if path not in cache:
        cache[path] = zipfile.ZipFile(path)
    return cache[path]


def read_audio(locator: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 waveform resampled to ``sample_rate`` if needed."""
    import soundfile as sf

    parts = split_locator(str(locator))
    source = io.BytesIO(_archive(parts[0]).read(parts[1])) if parts else str(locator)
    wav, rate = sf.read(source, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if rate != sample_rate:
        from scipy.signal import resample_poly

        g = math.gcd(rate, sample_rate)
        wav = resample_poly(wav, sample_rate // g, rate // g).astype(np.float32)
    return wav


def audio_duration(locator: str | Path) -> float:
    """Duration in seconds, read from the file header."""
    import soundfile as sf

    parts = split_locator(str(locator))
    if parts:
        with _archive(parts[0]).open(parts[1]) as handle:
            info = sf.info(handle)
    else:
        info = sf.info(str(locator))
    return info.frames / info.samplerate
