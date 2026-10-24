"""Phoneme-based sentence generation for enrollment prompts (PRICAI 2025, Algorithm 1)."""

from __future__ import annotations

import pandas as pd

from .g2p import parse_syllable, tokenize


def generate_prompts(corpus: str, n_sentences: int, words_per_sentence: int, g2p) -> list[list[str]]:
    """Algorithm 1: rank the unique words of ``corpus`` by phoneme coverage
    c(w) = |phonemes(w)| (ties keep corpus order) and fill ``n_sentences`` sentences
    with up to ``words_per_sentence`` words each, in rank order."""
    first_seen: dict[str, int] = {}
    for token in tokenize(corpus):
        if token not in first_seen and parse_syllable(token) is not None:
            first_seen[token] = len(first_seen)
    ranked = sorted(first_seen, key=lambda w: (-g2p.coverage(w), first_seen[w]))
    sentences = []
    for i in range(n_sentences):
        words = ranked[i * words_per_sentence : (i + 1) * words_per_sentence]
        if not words:
            break
        sentences.append(words)
    return sentences


def prompt_coverage(prompts: list[list[str]], g2p, corpus: str) -> pd.DataFrame:
    """Per-sentence and cumulative phoneme coverage of generated prompts."""
    inventory = g2p.phoneme_set(corpus)
    covered: set[str] = set()
    rows = []
    for i, words in enumerate(prompts, 1):
        units = g2p.phoneme_set(" ".join(words))
        covered |= units
        rows.append(
            {
                "sentence": i,
                "text": " ".join(words),
                "coverage": len(units),
                "cumulative_coverage": len(covered),
                "share_of_corpus_inventory": len(covered) / max(len(inventory), 1),
            }
        )
    return pd.DataFrame(rows)
