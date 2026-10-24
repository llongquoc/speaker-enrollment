"""Phoneme coverage of transcripts (PRICAI 2025, Sec. 4.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def phonemize_transcripts(transcripts: pd.DataFrame, g2p) -> pd.DataFrame:
    """``utterance, text`` -> ``utterance, text, units, coverage, n_syllables, unparsed``.

    ``units`` is the space-separated phoneme sequence; ``coverage`` the number of
    distinct units; ``unparsed`` lists tokens that are not Vietnamese syllables
    (e.g. English words), which do not contribute units.
    """
    rows = []
    for utterance, text in zip(transcripts.utterance, transcripts.text.fillna("")):
        units, n_syllables, unparsed = g2p.analyze(text)
        rows.append(
            {
                "utterance": utterance,
                "text": text,
                "units": " ".join(units),
                "coverage": len(set(units)),
                "n_syllables": n_syllables,
                "unparsed": " ".join(unparsed),
            }
        )
    return pd.DataFrame(rows)


def load_phoneme_sets(path: str | Path, utterances: list[str]) -> tuple[np.ndarray, list[set[str]]]:
    """Coverage array and phoneme sets aligned to ``utterances``."""
    df = pd.read_csv(path, dtype={"utterance": str, "units": str}, keep_default_na=False)
    sets = {u: set(x.split()) for u, x in zip(df.utterance, df.units)}
    missing = [u for u in utterances if u not in sets]
    if missing:
        raise KeyError(f"{len(missing)} utterances have no phoneme transcription, e.g. {missing[:3]}")
    aligned = [sets[u] for u in utterances]
    return np.array([len(s) for s in aligned]), aligned
