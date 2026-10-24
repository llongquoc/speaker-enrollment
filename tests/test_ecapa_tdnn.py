"""The ECAPA-TDNN port against an implementation with the original code structure."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from speaker_enrollment.data import index_corpus, read_audio  # noqa: E402
from speaker_enrollment.embedding import (  # noqa: E402
    ECAPATDNN,
    embed_waveform,
    extract_embeddings,
    load_checkpoint,
    zero_subnormal,
)
from speaker_enrollment.embedding.ecapa_tdnn import htk_mel_filterbank  # noqa: E402


class RefSE(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class RefBottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=pad) for _ in range(self.nums)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = RefSE(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]  # noqa: F821
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.se(self.bn3(self.relu(self.conv3(out))))
        out += residual
        return out


class RefECAPA(nn.Module):
    def __init__(self, C, fbank):
        super().__init__()
        self.torchfbank = fbank
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = RefBottle2neck(C, C, 3, 2)
        self.layer2 = RefBottle2neck(C, C, 3, 3)
        self.layer3 = RefBottle2neck(C, C, 3, 4)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, 1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
        x = self.bn1(self.relu(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        t = x.size()[-1]
        g = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )
        w = self.attention(g)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        return self.bn6(self.fc6(self.bn5(torch.cat((mu, sg), 1))))


def _checkpoint(path, C=48, n_classes=7):
    torch.manual_seed(0)
    model = ECAPATDNN(C=C)
    g = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.normal_(0, 0.1, generator=g)
            m.running_var.uniform_(0.5, 1.5, generator=g)
    state = {f"speaker_encoder.{k}": v for k, v in model.state_dict().items()}
    state["speaker_loss.weight"] = torch.randn(n_classes, 192)
    torch.save(state, path)


def test_matches_original_structure_and_front_end(tmp_path):
    torchaudio = pytest.importorskip("torchaudio")
    _checkpoint(tmp_path / "m.model")
    model, weights = load_checkpoint(tmp_path / "m.model")
    fbank = nn.Sequential(
        model.torchfbank[0],
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            window_fn=torch.hamming_window,
            n_mels=80,
        ),
    )
    reference = RefECAPA(48, fbank)
    state = torch.load(tmp_path / "m.model")
    reference.load_state_dict(
        {
            k[len("speaker_encoder.") :]: v
            for k, v in state.items()
            if k.startswith("speaker_encoder.") and "torchfbank" not in k
        },
        strict=False,
    )
    reference.eval()
    x = torch.randn(2, 32000) * 0.1
    with torch.no_grad():
        assert torch.allclose(model.torchfbank(x), fbank(x), rtol=1e-5, atol=1e-6)
        assert torch.allclose(model(x), reference(x), atol=1e-4, rtol=1e-4)
    assert weights.shape == (7, 192)
    expected = torchaudio.functional.melscale_fbanks(257, 20.0, 7600.0, 80, 16000, norm=None, mel_scale="htk")
    assert torch.allclose(htk_mel_filterbank(257, 20.0, 7600.0, 80, 16000), expected, atol=1e-6)


def test_load_checkpoint_rejects_other_files(tmp_path):
    torch.save({"weight": torch.zeros(3)}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "x.pt")
    pointer = tmp_path / "pointer.model"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 62743100\n")
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        load_checkpoint(pointer)


def test_zero_subnormal_keeps_outputs():
    model = ECAPATDNN(C=32).eval()
    with torch.no_grad():
        model.conv1.weight[0, 0, 0] = 1e-40
    x = torch.randn(1, 16000) * 0.1
    with torch.no_grad():
        before = model(x)
        assert zero_subnormal(model) == 1
        assert torch.equal(before, model(x))


def test_extract_embeddings_from_archive(mini_corpus):
    model, _ = load_checkpoint(mini_corpus["checkpoint"])
    index = index_corpus(mini_corpus["archive"])
    vectors = extract_embeddings(model, list(index.locator[:3]))
    assert vectors.shape == (3, 192) and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    with torch.no_grad():
        direct = F.normalize(model(torch.from_numpy(read_audio(index.locator[1])).unsqueeze(0)), dim=1)[0].numpy()
    assert np.allclose(vectors[1], direct, atol=1e-5)
    assert np.allclose(embed_waveform(model, read_audio(index.locator[1])), vectors[1], atol=1e-6)
    cropped = extract_embeddings(model, list(index.locator[:1]), max_seconds=0.5)
    assert cropped.shape == (1, 192)
    assert embed_waveform(model, np.ones(100, dtype=np.float32)).shape == (192,)  # shorter than one window
    with pytest.raises(ValueError):
        embed_waveform(model, np.zeros(0))
