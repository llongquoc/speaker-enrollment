"""Readers for evaluation-protocol files.

Utterances are identified by their path relative to the corpus audio directory,
``<speaker>/<file>`` (e.g. ``id00760/00020.wav``), as in the Vietnam-Celeb lists.
Longer paths in input files (``testout/id00760/00020.wav``) are reduced to their last
two components.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd


def utterance_id(path: str) -> str:
    """``'any/prefix/id00760/00020.wav'`` -> ``'id00760/00020.wav'``."""
    parts = path.strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"utterance path needs a speaker directory: {path!r}")
    return f"{parts[-2]}/{parts[-1]}"


def speaker_of(utterance: str) -> str:
    return utterance.split("/", 1)[0]


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _lines(path: str | Path):
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_speaker_list(path: str | Path) -> list[str]:
    """One speaker identifier per line (``#`` starts a comment line)."""
    speakers = list(dict.fromkeys(_lines(path)))
    if not speakers:
        raise ValueError(f"empty speaker list: {path}")
    return speakers


def read_utterance_list(path: str | Path) -> list[str]:
    """Sorted unique utterance ids from a plain list (one path per line) or a trial
    list (``label path_a path_b``)."""
    ids: set[str] = set()
    for line in _lines(path):
        fields = line.split()
        if len(fields) == 3:
            ids.update((utterance_id(fields[1]), utterance_id(fields[2])))
        else:
            ids.add(utterance_id(fields[-1]))
    if not ids:
        raise ValueError(f"no utterances in {path}")
    return sorted(ids)


def read_trials(path: str | Path) -> pd.DataFrame:
    """Verification trials ``label path_a path_b`` (``.gz`` accepted).  Columns:
    ``label`` (1 = same speaker), ``a``, ``b`` (utterance ids)."""
    rows = []
    for line in _lines(path):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"malformed trial line in {path}: {line!r}")
        rows.append((int(fields[0]), utterance_id(fields[1]), utterance_id(fields[2])))
    return pd.DataFrame(rows, columns=["label", "a", "b"])


def read_transcripts(path: str | Path) -> pd.DataFrame:
    """Transcripts as a JSON object ``{"<speaker>/<file>": "text"}`` or a CSV with
    columns ``utterance,text``.  Returns columns ``utterance``, ``text``."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame({"utterance": [utterance_id(k) for k in data], "text": [str(v) for v in data.values()]})
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "utterance" not in df or "text" not in df:
            raise ValueError(f"{path} needs columns 'utterance' and 'text'")
        df["utterance"] = [utterance_id(u) for u in df.utterance]
    duplicated = df.utterance[df.utterance.duplicated()]
    if len(duplicated):
        raise ValueError(f"duplicate transcript ids in {path}, e.g. {duplicated.iloc[0]}")
    return df.reset_index(drop=True)


def write_list(path: str | Path, items, header: str | None = None) -> None:
    lines = [f"# {line}" for line in header.splitlines()] if header else []
    Path(path).write_text("\n".join(lines + list(items)) + "\n", encoding="utf-8")
