"""Phoneme units from viphoneme (Nguyen Minh Tri et al., NICS 2020), the extractor
cited by the PRICAI 2025 paper, with the paper's refinement: a nucleus followed by an
off-glide coda is one compound-vowel unit (*câu* -> k, ɤ̆ʷ instead of k, ɤ̆, w).

The viphoneme package imports text normalisation and word segmentation tools at
module level (vinorm, underthesea, eng_to_ipa).  Only its syllable rules are needed
here; they use the standard library only and are loaded from the installed package
source without those imports.  Install the rules with::

    pip install --no-deps viphoneme==3.0.0

Settings match viphoneme's ``T2IPA_split``: Northern dialect, six tones, no glottal or
palatal options.
"""

from __future__ import annotations

import importlib.util
import re

from .g2p import tokenize

GLIDE_MARK = {"w": "ʷ", "j": "ʲ"}
_TONE_MODES = ("separate", "attached", "none")


def _load_rules():
    spec = importlib.util.find_spec("viphoneme")
    if spec is None or not spec.origin:
        raise ImportError("viphoneme is not installed: pip install --no-deps viphoneme==3.0.0")
    source = open(spec.origin, encoding="utf-8").read()
    cut = re.search(r"^from vinorm import", source, flags=re.M)
    if not cut or "def trans(" not in source[: cut.start()]:
        raise ImportError(f"unexpected viphoneme source layout in {spec.origin}; install viphoneme==3.0.0")
    namespace: dict = {"__name__": "viphoneme_rules"}
    exec(compile(source[: cut.start()], spec.origin, "exec"), namespace)  # noqa: S102 - trusted package source
    return namespace["trans"]


class ViphonemeG2P:
    """Text -> phoneme units with viphoneme's syllable rules (Northern, six tones)."""

    def __init__(self, compound_vowels: bool = True, tones: str = "separate"):
        if tones not in _TONE_MODES:
            raise ValueError(f"tones must be one of {_TONE_MODES}")
        self._trans = _load_rules()
        self.compound_vowels = compound_vowels
        self.tones = tones
        self._cache: dict[str, list[str] | None] = {}

    def syllable_units(self, word: str) -> list[str] | None:
        """Units of one syllable, or ``None`` if it is not a Vietnamese syllable."""
        if word in self._cache:
            return self._cache[word]
        try:
            parsed = self._trans(word, "n", 0, 1, 0, 0)
        except Exception:
            parsed = None
        if parsed is None or None in parsed or not parsed[1]:
            self._cache[word] = None
            return None
        onset, nucleus, coda, tone = parsed
        if self.compound_vowels and coda in GLIDE_MARK:
            nucleus, coda = nucleus + GLIDE_MARK[coda], ""
        if self.tones == "attached":
            nucleus = f"{nucleus}{tone}"
        units = [u for u in (onset, nucleus, coda) if u]
        if self.tones == "separate":
            units.append(f"T{tone}")
        self._cache[word] = units
        return units

    def analyze(self, text: str) -> tuple[list[str], int, list[str]]:
        """(phoneme units, number of syllables, tokens that are not Vietnamese syllables)."""
        units: list[str] = []
        n_syllables = 0
        unparsed: list[str] = []
        for token in tokenize(text):
            syllable = self.syllable_units(token)
            if syllable is None:
                unparsed.append(token)
            else:
                units.extend(syllable)
                n_syllables += 1
        return units, n_syllables, unparsed

    def units(self, text: str) -> list[str]:
        return self.analyze(text)[0]

    def phoneme_set(self, text: str) -> set[str]:
        return set(self.units(text))

    def coverage(self, text: str) -> int:
        """Number of distinct phoneme units."""
        return len(self.phoneme_set(text))
