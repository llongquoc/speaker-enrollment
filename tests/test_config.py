import pytest

from speaker_enrollment.config import Config, load_config


def test_repository_config_matches_defaults(repository_root):
    loaded = load_config(repository_root / "configs" / "vietnam_celeb.yaml")
    assert loaded.to_dict() == Config().to_dict()


def test_overrides_and_validation():
    config = load_config(None, ["threshold.rule=far", "threshold.target_far=0.01", "analysis.thresholds=[0.2, 0.3]"])
    assert config.threshold.rule == "far" and config.analysis.thresholds == [0.2, 0.3]
    with pytest.raises(ValueError):
        load_config(None, ["threshold.rule=far"])
    with pytest.raises(ValueError):
        load_config(None, ["identification.scoring=median"])
    with pytest.raises(KeyError):
        load_config(None, ["model.weights=x"])
    with pytest.raises(ValueError):
        load_config(None, ["threshold.value"])
