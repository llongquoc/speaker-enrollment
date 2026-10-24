"""Corpus indexing and evaluation manifests.

A corpus is a directory tree or a ZIP archive whose audio lies under
``<audio_subdir>/<speaker>/<file>`` (``data/id00000/00000.wav`` in Vietnam-Celeb).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .audio import SAMPLE_RATE, make_locator

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac")
SUBSETS = ("test", "validation", "other", "excluded")


def _approx_seconds(n_bytes: int) -> float:
    """Duration estimate for 16-bit mono PCM at 16 kHz (only used to rank utterances)."""
    return max(n_bytes - 44, 0) / (2 * SAMPLE_RATE)


def index_corpus(path: str | Path, audio_subdir: str = "data") -> pd.DataFrame:
    """Utterance index of a corpus directory or ZIP archive.

    Returns columns ``utterance`` (``<speaker>/<file>``), ``speaker``, ``locator``
    and ``approx_seconds``, sorted by utterance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    rows = _index_archive(path, audio_subdir) if zipfile.is_zipfile(path) else _index_directory(path, audio_subdir)
    if not rows:
        raise FileNotFoundError(f"no audio files found in {path} (audio_subdir={audio_subdir!r})")
    df = pd.DataFrame(rows).sort_values("utterance").reset_index(drop=True)
    duplicated = df.utterance[df.utterance.duplicated()]
    if len(duplicated):
        raise ValueError(f"duplicate utterance ids in {path}, e.g. {duplicated.iloc[0]}")
    log.info("indexed %s: %d speakers, %d utterances", path, df.speaker.nunique(), len(df))
    return df


def _index_archive(path: Path, audio_subdir: str) -> list[dict]:
    prefix = f"{audio_subdir.strip('/')}/" if audio_subdir else ""
    rows = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or not name.lower().endswith(AUDIO_EXTENSIONS) or not name.startswith(prefix):
                continue
            parts = name[len(prefix) :].split("/")
            if len(parts) != 2:
                continue
            rows.append(
                {
                    "utterance": "/".join(parts),
                    "speaker": parts[0],
                    "locator": make_locator(path, name),
                    "approx_seconds": _approx_seconds(info.file_size),
                }
            )
    return rows


def _index_directory(path: Path, audio_subdir: str) -> list[dict]:
    root = path / audio_subdir if audio_subdir and (path / audio_subdir).is_dir() else path
    rows = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(speaker_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append(
                    {
                        "utterance": f"{speaker_dir.name}/{f.name}",
                        "speaker": speaker_dir.name,
                        "locator": str(f.resolve()),
                        "approx_seconds": _approx_seconds(f.stat().st_size),
                    }
                )
    return rows


def build_manifest(
    index: pd.DataFrame,
    test_utterances: list[str],
    validation_speakers: list[str] | None = None,
    relabel: dict[str, str] | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Assign every utterance of the corpus to a subset.

    ``test``        utterances of the test list (label = speaker directory)
    ``validation``  all utterances of the validation speakers
    ``other``       everything else (e.g. training speakers)

    ``relabel`` maps utterance ids to corrected speaker labels (``"DROP"`` removes the
    utterance).  Missing test utterances raise unless ``allow_missing``.
    """
    df = index.copy()
    known = set(df.utterance)
    missing = [u for u in test_utterances if u not in known]
    if missing and not allow_missing:
        raise FileNotFoundError(f"{len(missing)} test utterances are not in the corpus, e.g. {missing[:3]}")
    if missing:
        log.warning("%d test utterances are not in the corpus and are skipped", len(missing))
    df["subset"] = "other"
    df.loc[df.utterance.isin(set(test_utterances)), "subset"] = "test"
    if validation_speakers:
        test_speakers = set(df.speaker[df.subset == "test"])
        overlap = sorted(set(validation_speakers) & test_speakers)
        if overlap:
            raise ValueError(f"validation and test speakers overlap: {overlap[:5]}")
        unknown = sorted(set(validation_speakers) - set(df.speaker))
        if unknown:
            raise ValueError(f"validation speakers not in the corpus: {unknown[:5]}")
        df.loc[df.speaker.isin(set(validation_speakers)), "subset"] = "validation"
    if relabel:
        unknown = sorted(set(relabel) - known)
        if unknown:
            raise ValueError(f"relabelled utterances not in the corpus: {unknown[:5]}")
        df["speaker"] = [relabel.get(u, s) for u, s in zip(df.utterance, df.speaker)]
        df = df[df.speaker != "DROP"]
    return df.reset_index(drop=True)


def read_relabel_csv(path: str | Path) -> dict[str, str]:
    """CSV with columns ``utterance,speaker`` (corrected labels; ``DROP`` removes)."""
    from .protocol import utterance_id

    df = pd.read_csv(path, dtype=str)
    if "utterance" not in df or "speaker" not in df:
        raise ValueError(f"{path} needs columns 'utterance' and 'speaker'")
    return {utterance_id(u): s for u, s in zip(df.utterance, df.speaker)}
