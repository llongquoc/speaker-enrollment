"""Shared fixtures: synthetic embeddings, a miniature corpus archive with a
random-weight checkpoint, and a matching configuration."""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from speaker_enrollment.embedding.extract import EmbeddingSet

SENTENCES = [
    "sáng nay trời mưa nhỏ nên tôi đi làm muộn hơn mọi ngày",
    "mẹ tôi thường nấu canh chua cá lóc vào cuối tuần",
    "chúng ta cần hoàn thành báo cáo trước thứ sáu tuần sau",
    "quán cà phê đầu ngõ mở cửa từ sáu giờ sáng",
    "những chiếc thuyền nhỏ khuất dần sau rặng dừa xanh",
    "bạn có thể nói chậm lại một chút được không",
]


def synthetic_embeddings(
    n_speakers=12, utterances=(30, 50), dim=32, outlier_fraction=0.15, seed=0, prefix="spk"
) -> EmbeddingSet:
    """Speakers as noisy clusters around a shared direction, with outlier utterances."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)
    utts, spks, vecs = [], [], []
    for s in range(n_speakers):
        centre = rng.normal(size=dim)
        centre = centre / np.linalg.norm(centre) + 0.8 * common
        for u in range(rng.integers(*utterances)):
            v = centre + 0.35 * rng.normal(size=dim)
            if rng.random() < outlier_fraction:
                v += 1.2 * rng.normal(size=dim)
            utts.append(f"{prefix}{s:03d}/{u:05d}.wav")
            spks.append(f"{prefix}{s:03d}")
            vecs.append(v)
    return EmbeddingSet(utts, spks, np.array(vecs))


@pytest.fixture
def embeddings() -> EmbeddingSet:
    return synthetic_embeddings()


def _tone(rng, seconds, f0):
    t = np.arange(int(16000 * seconds)) / 16000
    wave = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2.5 * f0 * t)
    return (wave + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def mini_corpus(tmp_path) -> dict:
    """Archive ``data/<speaker>/<file>.wav`` with test, validation and training
    speakers, protocol files, a random-weight checkpoint and a configuration."""
    sf = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    from speaker_enrollment.embedding.ecapa_tdnn import ECAPATDNN

    rng = np.random.default_rng(0)
    layout = {
        **{f"t{i:02d}": 12 + i for i in range(7)},  # test speakers
        **{f"v{i:02d}": 8 for i in range(4)},  # validation speakers
        **{f"r{i:02d}": 6 for i in range(5)},  # training speakers
    }
    archive = tmp_path / "corpus.zip"
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    test_utterances = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "miniature corpus")
        for s, (speaker, n) in enumerate(layout.items()):
            for u in range(n):
                path = wav_dir / f"{speaker}_{u}.wav"
                sf.write(
                    path, _tone(rng, 0.8 + 0.1 * (u % 5), 110 + 23 * s), 16000, subtype="PCM_24" if u == 0 else "PCM_16"
                )
                z.write(path, f"data/{speaker}/{u:05d}.wav")
                if speaker.startswith("t") and u % 4 != 3:  # a subset of the test speakers' utterances
                    test_utterances.append(f"{speaker}/{u:05d}.wav")
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "test_utterances.txt").write_text("\n".join(f"testout/{u}" for u in test_utterances) + "\n")
    (protocol / "validation_speakers.txt").write_text(
        "# validation\n" + "\n".join(f"v{i:02d}" for i in range(4)) + "\n"
    )
    with gzip.open(protocol / "test_trials.txt.gz", "wt") as handle:
        for a, b in zip(test_utterances[:-1], test_utterances[1:]):
            handle.write(f"{int(a.split('/')[0] == b.split('/')[0])} {a} {b}\n")
    transcripts = {u: SENTENCES[i % len(SENTENCES)] for i, u in enumerate(test_utterances)}
    (protocol / "test_transcripts.json").write_text(json.dumps(transcripts, ensure_ascii=False))
    torch.manual_seed(0)
    model = ECAPATDNN(C=32)
    state = {f"speaker_encoder.{k}": v for k, v in model.state_dict().items()}
    state["speaker_loss.weight"] = torch.randn(5, 192)
    checkpoint = tmp_path / "model.model"
    torch.save(state, checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text(f"""
corpus: {{path: {archive}, audio_subdir: data}}
protocol:
  test_utterances: {protocol / "test_utterances.txt"}
  validation_speakers: {protocol / "validation_speakers.txt"}
  test_trials: {protocol / "test_trials.txt.gz"}
  transcripts: {protocol / "test_transcripts.json"}
model: {{checkpoint: {checkpoint}, sha256: null, device: cpu}}
phoneme_selection: {{g2p: rules}}
similarity_selection: {{max_size: 4, repetitions: 3}}
analysis: {{thresholds: [0.2, 0.5], repetitions: 2, report_size: 3}}
output_dir: {tmp_path / "outputs"}
""")
    return {
        "archive": archive,
        "protocol": protocol,
        "checkpoint": checkpoint,
        "config": config,
        "root": tmp_path,
        "test_utterances": test_utterances,
        "layout": layout,
    }


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]
