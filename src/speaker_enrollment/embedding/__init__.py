"""Speaker encoder and embedding extraction."""

from .ecapa_tdnn import ECAPATDNN, load_checkpoint, zero_subnormal
from .extract import EmbeddingSet, embed_waveform, extract_embeddings, select_device, verify_device

__all__ = [
    "ECAPATDNN",
    "EmbeddingSet",
    "embed_waveform",
    "extract_embeddings",
    "load_checkpoint",
    "select_device",
    "verify_device",
    "zero_subnormal",
]
