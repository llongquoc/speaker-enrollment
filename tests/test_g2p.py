import unicodedata

import pytest

from speaker_enrollment.phonetics import VietnameseG2P, make_g2p, parse_syllable, read_number, tokenize

# word: (onset, medial, nucleus, off-glide, coda, tone)
CASES = {
    "câu": ("k", "", "ə", "w", "", 1),  # PRICAI-2025 example: /k/ /əʷ/
    "quốc": ("k", "", "uə", "", "k", 5),
    "quý": ("k", "w", "i", "", "", 5),
    "quanh": ("k", "w", "ɛ", "", "ŋ", 1),
    "quyết": ("k", "w", "iə", "", "t", 5),
    "gì": ("z", "", "i", "", "", 2),
    "gìn": ("z", "", "i", "", "n", 2),
    "giếng": ("z", "", "iə", "", "ŋ", 5),
    "gieo": ("z", "", "ɛ", "w", "", 1),
    "giữ": ("z", "", "ɯ", "", "", 3),
    "khuya": ("x", "w", "iə", "", "", 1),
    "khuỷu": ("x", "w", "i", "w", "", 4),
    "người": ("ŋ", "", "ɯə", "j", "", 2),
    "rượu": ("r", "", "ɯə", "w", "", 6),
    "tuổi": ("t", "", "uə", "j", "", 4),
    "của": ("k", "", "uə", "", "", 4),
    "tuần": ("t", "w", "ə", "", "n", 2),
    "thuở": ("tʰ", "w", "əː", "", "", 4),
    "hoặc": ("h", "w", "a", "", "k", 6),
    "xoay": ("s", "w", "a", "j", "", 1),
    "ngoài": ("ŋ", "w", "aː", "j", "", 2),
    "sao": ("s", "", "aː", "w", "", 1),
    "sau": ("s", "", "a", "w", "", 1),
    "anh": ("", "", "ɛ", "", "ŋ", 1),
    "bịch": ("ɓ", "", "i", "", "k", 6),
    "nhặt": ("ɲ", "", "a", "", "t", 6),
    "nghiêng": ("ŋ", "", "iə", "", "ŋ", 1),
    "yêu": ("", "", "iə", "w", "", 1),
    "uống": ("", "", "uə", "", "ŋ", 5),
    "oà": ("", "w", "aː", "", "", 2),
    "xoong": ("s", "", "ɔ", "", "ŋ", 1),
    "trên": ("ʈ", "", "e", "", "n", 1),
    "đường": ("ɗ", "", "ɯə", "", "ŋ", 2),
    "phở": ("f", "", "əː", "", "", 4),
}


@pytest.mark.parametrize("word,expected", CASES.items())
def test_parse(word, expected):
    s = parse_syllable(word)
    assert s is not None
    assert (s.onset, s.medial, s.nucleus, s.glide, s.coda, s.tone) == expected


@pytest.mark.parametrize("a,b", [("hoà", "hòa"), ("khoẻ", "khỏe"), ("thuý", "thúy"), ("Người", "người")])
def test_tone_placement_and_case(a, b):
    assert parse_syllable(a).units() == parse_syllable(b).units()


def test_decomposed_input():
    assert parse_syllable(unicodedata.normalize("NFD", "người")).units() == ["ŋ", "ɯəʲ", "T2"]


@pytest.mark.parametrize("word", ["facebook", "ok", "wifi", "xyz", "q", "hm"])
def test_non_vietnamese(word):
    assert parse_syllable(word) is None


def test_unit_modes():
    s = parse_syllable("câu")
    assert s.units() == ["k", "əʷ", "T1"]
    assert s.units(compound_vowels=False) == ["k", "ə", "w", "T1"]
    assert s.units(tones="attached") == ["k", "əʷ1"]
    assert s.units(tones="none") == ["k", "əʷ"]


def test_numbers():
    assert read_number("2020") == "hai nghìn không trăm hai mươi".split()
    assert read_number("105") == "một trăm linh năm".split()
    assert read_number("21") == "hai mươi mốt".split()
    assert read_number("15") == "mười lăm".split()
    assert read_number("24") == "hai mươi tư".split()
    assert read_number("0") == ["không"]
    assert read_number("091") == "không chín một".split()
    assert tokenize("Năm 25 tuổi.") == "năm hai mươi lăm tuổi".split()


def test_coverage_counts_unique_units():
    g = VietnameseG2P()
    assert g.coverage("ba ba") == 3  # ɓ, aː, T1
    sylls, oov = g.syllables("tôi dùng facebook")
    assert len(sylls) == 2 and oov == ["facebook"]


def test_backend_factory():
    assert isinstance(make_g2p("rules"), VietnameseG2P)
    with pytest.raises(ValueError):
        make_g2p("espeak")


@pytest.fixture
def viphoneme_g2p():
    try:
        return make_g2p("viphoneme")
    except ImportError:
        pytest.skip("viphoneme not installed (pip install --no-deps viphoneme==3.0.0)")


def test_viphoneme_compound_vowels(viphoneme_g2p):
    assert viphoneme_g2p.syllable_units("câu") == ["k", "ɤ̆ʷ", "T1"]  # PRICAI 2025 example
    assert viphoneme_g2p.syllable_units("người") == ["ŋ", "ɯəʲ", "T2"]
    assert viphoneme_g2p.syllable_units("ok") is None
    units, n, unparsed = viphoneme_g2p.analyze("Bà Tư vừa nhặt phở, 25 bịch ok.")
    assert n == 9 and unparsed == ["ok"] and units[:3] == ["b", "a", "T2"]


def test_viphoneme_without_fix_splits_off_glide():
    try:
        raw = make_g2p("viphoneme", compound_vowels=False)
    except ImportError:
        pytest.skip("viphoneme not installed")
    assert raw.syllable_units("câu") == ["k", "ɤ̆", "w", "T1"]
