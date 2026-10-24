"""ECAPA-TDNN speaker encoder (Desplanques et al., Interspeech 2020).

The module layout follows the open-source TaoRuijie/ECAPA-TDNN implementation, so its
checkpoints (``model_XXXX.model``: ``speaker_encoder.*`` parameters and the
AAM-softmax weights ``speaker_loss.weight``) load directly.  The front-end
(pre-emphasis 0.97, 80-band log-Mel spectrogram, 25 ms Hamming window, 10 ms hop,
512-point FFT, 20-7600 Hz, per-utterance mean normalisation) is written in plain
PyTorch and reads its window and Mel filterbank from the checkpoint, so it is
numerically identical to the original torchaudio front-end.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-excitation block."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class Bottle2neck(nn.Module):
    """SE-Res2Net block."""

    def __init__(self, inplanes: int, planes: int, kernel_size: int, dilation: int, scale: int = 8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        padding = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=padding) for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(self.nums))
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        splits = torch.split(out, self.width, 1)
        parts = []
        sp = splits[0]
        for i in range(self.nums):
            sp = splits[i] if i == 0 else sp + splits[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            parts.append(sp)
        parts.append(splits[self.nums])
        out = self.bn3(self.relu(self.conv3(torch.cat(parts, 1))))
        return self.se(out) + residual


class PreEmphasis(nn.Module):
    """First-order high-pass filter y[t] = x[t] - coefficient * x[t-1]."""

    flipped_filter: torch.Tensor

    def __init__(self, coefficient: float = 0.97):
        super().__init__()
        self.register_buffer("flipped_filter", torch.tensor([[[-coefficient, 1.0]]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class _Buffers(nn.Module):
    """Holds named buffers so that parameter names match the original checkpoints."""

    window: torch.Tensor
    fb: torch.Tensor


def htk_mel_filterbank(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int) -> torch.Tensor:
    """Triangular HTK Mel filterbank, (n_freqs, n_mels), as torchaudio.functional.melscale_fbanks."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min, m_max = (2595.0 * math.log10(1.0 + f / 700.0) for f in (f_min, f_max))
    f_pts = 700.0 * (10 ** (torch.linspace(m_min, m_max, n_mels + 2) / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.min(down, up), min=0.0)


class MelSpectrogram(nn.Module):
    """Power Mel spectrogram (centred frames, reflect padding)."""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        sample_rate: int = 16000,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.spectrogram = _Buffers()
        self.spectrogram.register_buffer("window", torch.hamming_window(win_length))
        self.mel_scale = _Buffers()
        self.mel_scale.register_buffer("fb", htk_mel_filterbank(n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = (
            torch.stft(
                x,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.spectrogram.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            .abs()
            .pow(2.0)
        )
        return torch.matmul(spec.transpose(-1, -2), self.mel_scale.fb).transpose(-1, -2)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN with channel width ``C`` and ``embedding_dim``-dimensional output."""

    def __init__(self, C: int = 1024, n_mels: int = 80, embedding_dim: int = 192):
        super().__init__()
        self.torchfbank = nn.Sequential(PreEmphasis(), MelSpectrogram(n_mels=n_mels))
        self.conv1 = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
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
        self.fc6 = nn.Linear(3072, embedding_dim)
        self.bn6 = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim

    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mean-normalised log-Mel features, (batch, n_mels, frames)."""
        with torch.no_grad():
            x = (self.torchfbank(waveform) + 1e-6).log()
            return x - torch.mean(x, dim=-1, keepdim=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) at 16 kHz -> (batch, embedding_dim) speaker embeddings."""
        x = self.bn1(self.relu(self.conv1(self.features(waveform))))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        t = x.size()[-1]
        context = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )
        w = self.attention(context)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        x = self.bn5(torch.cat((mu, sg), 1))
        return self.bn6(self.fc6(x))


@torch.no_grad()
def zero_subnormal(model: nn.Module) -> int:
    """Set subnormal float32 values (|w| < 1.18e-38, typically left by weight decay)
    to zero.  They do not change any output at float32 precision but slow down CPU
    matrix kernels by up to two orders of magnitude.  Returns the number changed."""
    tiny = torch.finfo(torch.float32).tiny
    changed = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point():
            mask = (t.abs() < tiny) & (t != 0)
            changed += int(mask.sum())
            t[mask] = 0.0
    return changed


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def check_not_lfs_pointer(path: str | Path) -> None:
    """Raise if ``path`` is a Git LFS pointer file instead of the file it stands for."""
    with open(path, "rb") as handle:
        if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise FileNotFoundError(
                f"{path} is a Git LFS pointer, not the checkpoint: install Git LFS and run "
                "`git lfs install` and `git lfs pull` in the repository"
            )


def load_checkpoint(path: str | Path) -> tuple[ECAPATDNN, torch.Tensor | None]:
    """Encoder (eval mode, on CPU) and AAM-softmax class weights (or ``None``)."""
    check_not_lfs_pointer(path)
    state = torch.load(str(path), map_location="cpu", weights_only=True)  # tensors only, no arbitrary objects
    if not isinstance(state, dict) or not any(k.startswith("speaker_encoder.") for k in state):
        raise ValueError(f"{path} is not an ECAPA-TDNN checkpoint with 'speaker_encoder.*' parameters")
    encoder_state = {k[len("speaker_encoder.") :]: v for k, v in state.items() if k.startswith("speaker_encoder.")}
    model = ECAPATDNN(
        C=encoder_state["conv1.weight"].shape[0],
        n_mels=encoder_state["conv1.weight"].shape[1],
        embedding_dim=encoder_state["fc6.weight"].shape[0],
    )
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    problems = [k for k in missing if not k.endswith("num_batches_tracked")] + list(unexpected)
    if problems:
        raise ValueError(f"{path} does not match the ECAPA-TDNN layout: {problems[:6]}")
    zero_subnormal(model)
    weights = state.get("speaker_loss.weight")
    return model.eval(), (weights.float() if weights is not None else None)
