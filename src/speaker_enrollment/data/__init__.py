"""Corpus access, protocol files and manifests."""

from .audio import ARCHIVE_SEPARATOR, SAMPLE_RATE, audio_duration, make_locator, read_audio, split_locator
from .dataset import build_manifest, index_corpus, read_relabel_csv
from .protocol import (
    read_speaker_list,
    read_transcripts,
    read_trials,
    read_utterance_list,
    speaker_of,
    utterance_id,
    write_list,
)

__all__ = [
    "ARCHIVE_SEPARATOR",
    "SAMPLE_RATE",
    "audio_duration",
    "build_manifest",
    "index_corpus",
    "make_locator",
    "read_audio",
    "read_relabel_csv",
    "read_speaker_list",
    "read_transcripts",
    "read_trials",
    "read_utterance_list",
    "speaker_of",
    "split_locator",
    "utterance_id",
    "write_list",
]
