"""Experiment configuration (YAML).

Relative paths are resolved against the current working directory; run the commands
from the repository root.  Any value can be overridden on the command line with
``--set section.key=value`` (the value is parsed as YAML).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from .identification import SCORING_RULES
from .phonetics import G2P_BACKENDS
from .selection import CANDIDATE_RULES, QUALITY_CHECKS
from .verification import THRESHOLD_RULES


@dataclass
class CorpusConfig:
    path: str = "data/full-dataset.zip"  # ZIP archive or extracted directory
    audio_subdir: str = "data"  # audio lies under <audio_subdir>/<speaker>/<file>


@dataclass
class ProtocolConfig:
    test_utterances: str = "protocols/vietnam_celeb/test_utterances.txt"
    validation_speakers: str = "protocols/vietnam_celeb/validation_speakers.txt"
    test_trials: str | None = "protocols/vietnam_celeb/test_trials.txt.gz"
    transcripts: str | None = "protocols/vietnam_celeb/test_transcripts.json"
    relabel: str | None = None  # optional CSV utterance,speaker


# SHA-256 of the released Vietnam-Celeb checkpoint (see checkpoints/README.md).
RELEASED_CHECKPOINT_SHA256 = "af01bb9dcb87a42e820aa7679d6cc6d3617b3e8f10be40b10a05f1117bdd8149"


@dataclass
class ModelConfig:
    checkpoint: str = "checkpoints/ecapa_tdnn_vietnam_celeb.model"
    sha256: str | None = RELEASED_CHECKPOINT_SHA256  # expected checksum of the checkpoint (None: not checked)
    device: str = "auto"  # auto | cpu | cuda | mps
    max_seconds: float | None = None  # centre-crop longer utterances (None: whole utterances)


@dataclass
class ThresholdConfig:
    rule: str = "eer"  # eer | far | fixed
    target_far: float | None = None  # for rule "far"
    value: float | None = None  # for rule "fixed"


@dataclass
class IdentificationConfig:
    scoring: str = "mean"  # mean | centroid | max
    reject: bool = True  # reject queries scoring below the threshold


@dataclass
class SimilaritySelectionConfig:  # ICPRAM 2025
    candidate_rule: str = "clique"  # clique | mean | all
    max_size: int = 10
    repetitions: int = 30
    seed: int = 0


@dataclass
class PhonemeSelectionConfig:  # PRICAI 2025
    g2p: str = "viphoneme"  # viphoneme | rules
    compound_vowels: bool = True
    tones: str = "separate"  # separate | attached | none
    quality_check: str = "pairwise"  # pairwise | mean


@dataclass
class AnalysisConfig:
    thresholds: list[float] = field(default_factory=lambda: [round(0.10 + 0.02 * i, 2) for i in range(21)])
    candidate_rules: list[str] = field(default_factory=lambda: list(CANDIDATE_RULES))
    scorings: list[str] = field(default_factory=lambda: list(SCORING_RULES))
    quality_checks: list[str] = field(default_factory=lambda: list(QUALITY_CHECKS))
    repetitions: int = 10
    report_size: int = 5  # enrollment size shown in the operating-point analysis


@dataclass
class Config:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    similarity_selection: SimilaritySelectionConfig = field(default_factory=SimilaritySelectionConfig)
    phoneme_selection: PhonemeSelectionConfig = field(default_factory=PhonemeSelectionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output_dir: str = "outputs/vietnam_celeb"

    def __post_init__(self) -> None:
        self.source: str | None = None  # configuration file, set by load_config

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> Config:
        checks = [
            (self.threshold.rule, THRESHOLD_RULES, "threshold.rule"),
            (self.identification.scoring, SCORING_RULES, "identification.scoring"),
            (self.similarity_selection.candidate_rule, CANDIDATE_RULES, "similarity_selection.candidate_rule"),
            (self.phoneme_selection.quality_check, QUALITY_CHECKS, "phoneme_selection.quality_check"),
            (self.phoneme_selection.g2p, G2P_BACKENDS, "phoneme_selection.g2p"),
            (self.phoneme_selection.tones, ("separate", "attached", "none"), "phoneme_selection.tones"),
        ]
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}; allowed: {allowed}")
        for values, allowed, name in (
            (self.analysis.candidate_rules, CANDIDATE_RULES, "analysis.candidate_rules"),
            (self.analysis.scorings, SCORING_RULES, "analysis.scorings"),
            (self.analysis.quality_checks, QUALITY_CHECKS, "analysis.quality_checks"),
        ):
            bad = [v for v in values if v not in allowed]
            if bad:
                raise ValueError(f"{name} contains {bad}; allowed: {allowed}")
        if self.threshold.rule == "far" and not (self.threshold.target_far and 0 < self.threshold.target_far < 1):
            raise ValueError("threshold.rule 'far' needs threshold.target_far in (0, 1)")
        if self.threshold.rule == "fixed" and self.threshold.value is None:
            raise ValueError("threshold.rule 'fixed' needs threshold.value")
        if self.similarity_selection.max_size < 1 or self.similarity_selection.repetitions < 1:
            raise ValueError("similarity_selection.max_size and repetitions must be >= 1")
        return self


def _update(obj, values: dict, prefix: str = "") -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown configuration key {prefix}{key!r}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            _update(current, value, f"{prefix}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults, updated by the YAML file at ``path`` and ``section.key=value`` overrides.
    The file path is kept in ``config.source``."""
    config = Config()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping of configuration sections")
        _update(config, data)
        config.source = str(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value: {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        nested: dict = {}
        cursor = nested
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
        _update(config, nested)
    return config.validate()
