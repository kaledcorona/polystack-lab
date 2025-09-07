import pytest
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import config.py from the same directory
from config import (
    _deep_update, _substitute_placeholders, _import_object, _make_estimator,
    _label_for, _expand_seeds, load_and_merge, _validate_config, resolve_config, ExperimentConfig
)

# Fixture for temporary YAML files
@pytest.fixture
def temp_yaml(tmp_path):
    def _create_yaml(content: dict) -> Path:
        path = tmp_path / "temp.yaml"
        with open(path, 'w') as f:
            yaml.safe_dump(content, f)
        return path
    return _create_yaml

# Unit Tests
def test_deep_update():
    base = {'a': 1, 'b': {'c': 2}}
    override = {'b': {'c': 3, 'd': 4}, 'e': 5}
    result = _deep_update(base, override)
    assert result == {'a': 1, 'b': {'c': 3, 'd': 4}, 'e': 5}
    # Ensure base is not mutated
    assert base == {'a': 1, 'b': {'c': 2}}

def test_substitute_placeholders():
    s = "path/to/${exp_name}/${now}"
    vars = {"exp_name": "test_exp", "now": "20250906_185300"}
    result = _substitute_placeholders(s, vars)
    assert result == "path/to/test_exp/20250906_185300"

def test_import_object():
    obj = _import_object("sklearn.ensemble.RandomForestClassifier")
    assert obj is RandomForestClassifier

def test_make_estimator():
    spec = {"class": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 50}}
    est = _make_estimator(spec)
    assert isinstance(est, RandomForestClassifier)
    assert est.n_estimators == 50
    with pytest.raises(TypeError):
        _make_estimator("invalid")

def test_label_for():
    assert _label_for({"class": "sklearn.linear_model.LogisticRegression"}) == "LogisticRegression"
    assert _label_for("sklearn.ensemble.RandomForestClassifier(n_estimators=100)") == "RandomForestClassifier"
    assert _label_for(None) == "final"

@pytest.mark.parametrize("cfg, expected", [
    ({"seeds": [1, 2, 3]}, [1, 2, 3]),
    ({"seeds": {"base": 10, "count": 3}}, [10, 11, 12]),
    ({"seeds": 42}, [42]),
    ({"iterations": 3}, [0, 1, 2]),
   ({}, [0]),
])
def test_expand_seeds(cfg, expected):
    assert _expand_seeds(cfg) == expected

def test_validate_config():
    valid_cfg = {"dataset": {}, "views": {"available": [{"name": "raw"}]}, "models": {}}
    _validate_config(valid_cfg)  # Should not raise
    with pytest.raises(ValueError, match="Missing required config section"):
        _validate_config({"views": {}, "models": {}})
    with pytest.raises(ValueError, match="'views.available' must be a list"):
        _validate_config({"dataset": {}, "views": {"available": {}}, "models": {}})
    with pytest.raises(ValueError, match="Each view in 'views.available' must be a dict with 'name'"):
        _validate_config({"dataset": {}, "views": {"available": [{}]}, "models": {}})

def test_load_and_merge(temp_yaml):
    cfg1 = {"a": 1, "b": {"c": 2}}
    cfg2 = {"b": {"c": 3, "d": 4}, "e": 5}
    path1 = temp_yaml(cfg1)
    path2 = temp_yaml(cfg2)
    result = load_and_merge([path1, path2])
    assert result == {"a": 1, "b": {"c": 3, "d": 4}, "e": 5}

# System Tests
def test_resolve_config_basic(temp_yaml):
    cfg = {
        "exp_name": "test",
        "iterations": 2,
        "dataset": {"name": "mnist"},
        "views": {"available": [{"name": "raw"}]},
        "models": {
            "base": {
                "default": {"class": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 100}},
                "by_view": {"view1": {"class": "sklearn.linear_model.LogisticRegression"}}
            },
            "final": {
                "choices": [
                    {"class": "sklearn.ensemble.RandomForestClassifier"},
                    {"class": "sklearn.linear_model.LogisticRegression"}
                ],
                "select": 0
            }
        },
        "metafeatures": {"name": "test_meta"}
    }
    path = temp_yaml(cfg)
    raw_cfg = load_and_merge([path])
    config = resolve_config(raw_cfg)
    
    assert isinstance(config, ExperimentConfig)
    assert config.seeds == [0, 1]
    assert config.meta_name == "test_meta"
    assert isinstance(config.base_default, RandomForestClassifier)
    assert isinstance(config.base_est_by_view["view1"], LogisticRegression)
    assert len(config.final_ests) == 1
    assert config.final_ests[0][0] == "RandomForestClassifier"
    assert isinstance(config.final_ests[0][1], RandomForestClassifier)
    assert str(config.output_dir).startswith("experiments/runs/test/")

def test_resolve_config_all_finals(temp_yaml):
    cfg = {
        "dataset": {"name": "mnist"},
        "views": {"available": [{"name": "raw"}]},
        "models": {
            "final": {
                "choices": [
                    {"class": "sklearn.ensemble.RandomForestClassifier"},
                    {"class": "sklearn.linear_model.LogisticRegression"}
                ],
                "select": "all"
            }
        }
    }
    path = temp_yaml(cfg)
    raw_cfg = load_and_merge([path])
    config = resolve_config(raw_cfg)
    assert len(config.final_ests) == 2
    assert config.final_ests[0][0] == "RandomForestClassifier"
    assert config.final_ests[1][0] == "LogisticRegression"

def test_resolve_config_invalid(temp_yaml):
    cfg = {"models": {}}  # Missing required keys
    path = temp_yaml(cfg)
    raw_cfg = load_and_merge([path])
    with pytest.raises(ValueError, match="Missing required config section"):
        resolve_config(raw_cfg)
