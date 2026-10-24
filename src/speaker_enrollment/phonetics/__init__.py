"""Vietnamese phonetics: grapheme-to-phoneme conversion, phoneme coverage and
phoneme-rich prompt generation."""

from .coverage import load_phoneme_sets, phonemize_transcripts
from .g2p import VietnameseG2P, parse_syllable, read_number, tokenize
from .prompts import generate_prompts, prompt_coverage

G2P_BACKENDS = ("viphoneme", "rules")


def make_g2p(backend: str = "viphoneme", compound_vowels: bool = True, tones: str = "separate"):
    """``viphoneme`` (extractor cited by PRICAI 2025; evaluation default) or ``rules``
    (built-in, dependency-free)."""
    if backend == "viphoneme":
        from .viphoneme import ViphonemeG2P

        return ViphonemeG2P(compound_vowels, tones)
    if backend == "rules":
        return VietnameseG2P(compound_vowels, tones)
    raise ValueError(f"unknown G2P backend {backend!r}; choose from {G2P_BACKENDS}")


__all__ = [
    "G2P_BACKENDS",
    "VietnameseG2P",
    "generate_prompts",
    "load_phoneme_sets",
    "make_g2p",
    "parse_syllable",
    "phonemize_transcripts",
    "prompt_coverage",
    "read_number",
    "tokenize",
]
