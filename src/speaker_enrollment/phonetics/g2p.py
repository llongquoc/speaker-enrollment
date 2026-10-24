"""Rule-based Vietnamese grapheme-to-phoneme conversion (built-in backend).

Each orthographic syllable is parsed into onset, medial glide, nucleus, off-glide,
coda and tone.  Conventions follow the transcriptions in Table 1 of the PRICAI 2025
paper (Northern variety: s/x -> s, d/gi -> z; tr and r kept distinct; final -ch -> k),
and a nucleus followed by an off-glide is a single compound-vowel unit
(*câu* -> /k/ /əʷ/), as in the paper's refined extractor.

This backend has no third-party dependency; the evaluation protocol uses the
viphoneme backend (:mod:`.viphoneme`), the extractor cited by the paper.  Phoneme
tables are plain dictionaries so that the inventory can be adapted in one place.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------- tones
# Numbering used by viphoneme / VLSP: 1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng.
TONE_MARKS = {"̀": 2, "̃": 3, "̉": 4, "́": 5, "̣": 6}
TONE_NAMES = {1: "ngang", 2: "huyền", 3: "ngã", 4: "hỏi", 5: "sắc", 6: "nặng"}

# --------------------------------------------------------------------------- onsets
ONSETS = {
    "ngh": "ŋ",
    "ng": "ŋ",
    "nh": "ɲ",
    "ch": "c",
    "tr": "ʈ",
    "th": "tʰ",
    "kh": "x",
    "ph": "f",
    "gh": "ɣ",
    "b": "ɓ",
    "c": "k",
    "d": "z",
    "đ": "ɗ",
    "g": "ɣ",
    "h": "h",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "v": "v",
    "x": "s",
}
GI_ONSET = "z"  # gi-
QU_ONSET = "k"  # qu- (the u is the medial /w/, except in quô-)
_ONSET_KEYS = sorted(ONSETS, key=len, reverse=True)

# --------------------------------------------------------------------------- rhymes
NUCLEUS = {
    "a": "aː",
    "ă": "a",
    "â": "ə",
    "e": "ɛ",
    "ê": "e",
    "i": "i",
    "y": "i",
    "o": "ɔ",
    "oo": "ɔ",
    "ô": "o",
    "ơ": "əː",
    "u": "u",
    "ư": "ɯ",
    "ia": "iə",
    "iê": "iə",
    "ya": "iə",
    "yê": "iə",
    "ua": "uə",
    "uô": "uə",
    "ưa": "ɯə",
    "ươ": "ɯə",
}
CODA = {"ng": "ŋ", "nh": "ŋ", "ch": "k", "m": "m", "n": "n", "p": "p", "t": "t", "c": "k"}
_CODA_KEYS = ("ng", "nh", "ch", "m", "n", "p", "t", "c")

# Off-glide letters allowed after each nucleus spelling (w = -o/-u, j = -i/-y).
_W_GLIDE = {"a": "ou", "â": "u", "e": "o", "ê": "u", "i": "u", "y": "u", "ư": "u", "iê": "u", "yê": "u", "ươ": "u"}
_J_GLIDE = {"a": "iy", "â": "y", "o": "i", "ô": "i", "ơ": "i", "u": "i", "ư": "i", "uô": "i", "ươ": "i"}

_NEEDS_FINAL = {"ă", "â", "iê", "yê", "uô", "ươ", "oo"}
_OPEN_ONLY = {"ia", "ya", "ua", "ưa"}
_PALATAL_OK = {"a", "ê", "i", "y"}  # may take -nh / -ch
_NO_VELAR = {"ê", "i", "y"}  # take -nh/-ch instead of -ng/-c
_MEDIAL_O_NEXT = set("aăe")  # hoa, hoặc, khoẻ
_MEDIAL_U_NEXT = set("âêơy")  # tuần, huế, thuở, thuỷ, khuya

VOWEL_LETTERS = set("aăâeêioôơuưy")
LETTERS = VOWEL_LETTERS | set("bcdđghklmnpqrstvx")

GLIDE_MARK = {"w": "ʷ", "j": "ʲ"}


@dataclass(frozen=True)
class Syllable:
    text: str
    onset: str  # phoneme or ""
    medial: str  # "w" or ""
    nucleus: str  # vowel / diphthong phoneme
    glide: str  # "w", "j" or ""
    coda: str  # phoneme or ""
    tone: int  # 1..6

    def units(self, compound_vowels: bool = True, tones: str = "separate") -> list[str]:
        """Phoneme units of the syllable.

        compound_vowels: merge nucleus + off-glide into one unit (/əʷ/), as in the
            paper's refined extractor; False emits them separately (/ə/ /w/).
        tones: "separate" (tone is its own unit T1..T6), "attached" (tone fused to
            the vowel unit, e.g. aː5) or "none".
        """
        out = [self.onset] if self.onset else []
        if self.medial:
            out.append(self.medial)
        vowel = self.nucleus
        if self.glide and compound_vowels:
            vowel += GLIDE_MARK[self.glide]
        if tones == "attached":
            vowel += str(self.tone)
        out.append(vowel)
        if self.glide and not compound_vowels:
            out.append(self.glide)
        if self.coda:
            out.append(self.coda)
        if tones == "separate":
            out.append(f"T{self.tone}")
        return out


def split_tone(word: str) -> tuple[str, int]:
    """Remove the tone mark; return (toneless NFC word, tone number)."""
    tone = 1
    kept = []
    for ch in unicodedata.normalize("NFD", word):
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
        elif ch in ("̀", "́"):  # deprecated tone marks
            tone = 2 if ch == "̀" else 5
        else:
            kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), tone


def _parse_rhyme(r: str, after_q: bool):
    medial = ""
    if after_q:
        if len(r) < 2 or r[0] != "u":
            return None
        if not r.startswith("uô"):
            medial, r = "w", r[1:]
    elif len(r) >= 2 and ((r[0] == "o" and r[1] in _MEDIAL_O_NEXT) or (r[0] == "u" and r[1] in _MEDIAL_U_NEXT)):
        medial, r = "w", r[1:]

    coda = ""
    for c in _CODA_KEYS:
        if r.endswith(c) and len(r) > len(c):
            coda, r = c, r[: -len(c)]
            break

    glide = glide_letter = ""
    if not coda and r not in NUCLEUS and len(r) >= 2:
        base, g = r[:-1], r[-1]
        if g in _W_GLIDE.get(base, ""):
            glide, glide_letter, r = "w", g, base
        elif g in _J_GLIDE.get(base, ""):
            glide, glide_letter, r = "j", g, base

    if r not in NUCLEUS:
        return None
    final = coda or glide
    if final and r in _OPEN_ONLY:
        return None
    if not final and r in _NEEDS_FINAL:
        return None
    if coda in ("nh", "ch") and r not in _PALATAL_OK:
        return None
    if coda in ("ng", "c") and r in _NO_VELAR:
        return None

    nucleus = NUCLEUS[r]
    if r == "a":
        if glide_letter in ("u", "y"):
            nucleus = "a"  # au, ay: short a
        elif coda in ("nh", "ch"):
            nucleus = "ɛ"  # anh, ach
    return medial, nucleus, glide, CODA.get(coda, "")


def parse_syllable(word: str) -> Syllable | None:
    """Parse one orthographic syllable; returns None if it is not Vietnamese."""
    w = unicodedata.normalize("NFC", word).lower()
    base, tone = split_tone(w)
    if not base or any(ch not in LETTERS for ch in base):
        return None

    after_q = False
    if base.startswith("gi"):
        onset, rest = GI_ONSET, base[2:]
        if not rest or rest[0] not in VOWEL_LETTERS or rest[0] == "ê":
            rest = "i" + rest  # gì, gìn, giếng
    elif base.startswith("qu"):
        onset, rest, after_q = QU_ONSET, base[1:], True
    else:
        onset, rest = "", base
        for key in _ONSET_KEYS:
            if base.startswith(key):
                onset, rest = ONSETS[key], base[len(key) :]
                break
    if not rest:
        return None
    parsed = _parse_rhyme(rest, after_q)
    if parsed is None:
        return None
    medial, nucleus, glide, coda = parsed
    return Syllable(w, onset, medial, nucleus, glide, coda, tone)


# --------------------------------------------------------------------------- numbers
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_GROUPS = ["", "nghìn", "triệu", "tỷ"]


def _read_triple(n: int, full: bool) -> list[str]:
    h, t, u = n // 100, (n // 10) % 10, n % 10
    out: list[str] = []
    if h or full:
        out += [_DIGITS[h], "trăm"]
    if t == 0:
        if u and (h or full):
            out.append("linh")
        if u:
            out.append(_DIGITS[u])
    elif t == 1:
        out.append("mười")
        if u:
            out.append("lăm" if u == 5 else _DIGITS[u])
    else:
        out += [_DIGITS[t], "mươi"]
        if u:
            out.append({1: "mốt", 4: "tư", 5: "lăm"}.get(u, _DIGITS[u]))
    return out


def read_number(digits: str) -> list[str]:
    """Spoken form of a digit string (integers; long or 0-led strings digit by digit)."""
    if len(digits) > 12 or (len(digits) > 1 and digits[0] == "0"):
        return [_DIGITS[int(d)] for d in digits]
    n = int(digits)
    if n == 0:
        return ["không"]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    out: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i] == 0:
            continue
        out += _read_triple(groups[i], full=i < len(groups) - 1)
        if _GROUPS[i]:
            out.append(_GROUPS[i])
    return out


# --------------------------------------------------------------------------- text level
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lower-cased syllable tokens; digit runs are expanded to their spoken form."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower()):
        out.extend(read_number(tok) if tok.isdigit() else [tok])
    return out


class VietnameseG2P:
    """Text -> phoneme units, phoneme sets and phoneme coverage (rule-based)."""

    def __init__(self, compound_vowels: bool = True, tones: str = "separate"):
        if tones not in ("separate", "attached", "none"):
            raise ValueError("tones must be 'separate', 'attached' or 'none'")
        self.compound_vowels = compound_vowels
        self.tones = tones
        self._cache: dict[str, Syllable | None] = {}

    def syllables(self, text: str) -> tuple[list[Syllable], list[str]]:
        """Parsed syllables and the tokens that could not be parsed (OOV)."""
        sylls, oov = [], []
        for tok in tokenize(text):
            if tok not in self._cache:
                self._cache[tok] = parse_syllable(tok)
            s = self._cache[tok]
            (sylls.append(s) if s is not None else oov.append(tok))
        return sylls, oov

    def analyze(self, text: str) -> tuple[list[str], int, list[str]]:
        """(phoneme units, number of parsed syllables, unparsed tokens)."""
        sylls, oov = self.syllables(text)
        units = [p for s in sylls for p in s.units(self.compound_vowels, self.tones)]
        return units, len(sylls), oov

    def units(self, text: str) -> list[str]:
        return self.analyze(text)[0]

    def phoneme_set(self, text: str) -> set[str]:
        return set(self.units(text))

    def coverage(self, text: str) -> int:
        """Phoneme coverage: number of distinct phoneme units (PRICAI 2025, Sec. 3.1)."""
        return len(self.phoneme_set(text))
