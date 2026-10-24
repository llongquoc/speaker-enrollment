import gzip
import json

import numpy as np
import pandas as pd
import pytest

from speaker_enrollment.data import (
    ARCHIVE_SEPARATOR,
    audio_duration,
    build_manifest,
    index_corpus,
    read_audio,
    read_speaker_list,
    read_transcripts,
    read_trials,
    read_utterance_list,
    utterance_id,
)


def test_utterance_id():
    assert utterance_id("testout/id00760/00020.wav") == "id00760/00020.wav"
    assert utterance_id("id00760/00020.wav") == "id00760/00020.wav"
    with pytest.raises(ValueError):
        utterance_id("00020.wav")


def test_index_archive_and_audio(mini_corpus):
    index = index_corpus(mini_corpus["archive"])
    assert len(index) == sum(mini_corpus["layout"].values())
    assert index.utterance.iloc[0] == "r00/00000.wav" and index.speaker.nunique() == 16
    locator = index.locator.iloc[0]
    assert ARCHIVE_SEPARATOR in locator
    wav = read_audio(locator)  # PCM_24 member
    assert wav.dtype == np.float32 and abs(audio_duration(locator) - len(wav) / 16000) < 1e-9


def test_index_directory(tmp_path):
    sf = pytest.importorskip("soundfile")
    for spk in ("a", "b"):
        (tmp_path / "data" / spk).mkdir(parents=True)
        sf.write(tmp_path / "data" / spk / "0.wav", np.zeros(800, np.float32), 8000)  # resampled on read
    index = index_corpus(tmp_path)
    assert index.utterance.tolist() == ["a/0.wav", "b/0.wav"]
    assert len(read_audio(index.locator.iloc[0])) == 1600


def test_build_manifest(mini_corpus):
    index = index_corpus(mini_corpus["archive"])
    test = read_utterance_list(mini_corpus["protocol"] / "test_utterances.txt")
    validation = read_speaker_list(mini_corpus["protocol"] / "validation_speakers.txt")
    m = build_manifest(index, test, validation)
    assert sorted(m.utterance[m.subset == "test"]) == sorted(mini_corpus["test_utterances"])
    assert set(m.speaker[m.subset == "validation"]) == set(validation)
    with pytest.raises(FileNotFoundError):
        build_manifest(index, test + ["t00/99999.wav"], validation)
    with pytest.raises(ValueError):
        build_manifest(index, test, validation + ["t00"])
    relabelled = build_manifest(index, test, validation, relabel={test[0]: "t01", test[1]: "DROP"})
    assert relabelled.set_index("utterance").speaker[test[0]] == "t01" and test[1] not in set(relabelled.utterance)


def test_protocol_readers(tmp_path):
    trials = tmp_path / "trials.txt.gz"
    with gzip.open(trials, "wt") as handle:
        handle.write("1 x/id1/a.wav x/id1/b.wav\n0 x/id1/a.wav x/id2/c.wav\n")
    t = read_trials(trials)
    assert t.label.tolist() == [1, 0] and t.a.tolist() == ["id1/a.wav", "id1/a.wav"]
    assert read_utterance_list(trials) == ["id1/a.wav", "id1/b.wav", "id2/c.wav"]
    js = tmp_path / "t.json"
    js.write_text(json.dumps({"id1/a.wav": "xin chào"}, ensure_ascii=False), encoding="utf-8")
    assert read_transcripts(js).to_dict("records") == [{"utterance": "id1/a.wav", "text": "xin chào"}]
    csv = tmp_path / "t.csv"
    pd.DataFrame({"utterance": ["id1/a.wav", "id1/a.wav"], "text": ["a", "b"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        read_transcripts(csv)
