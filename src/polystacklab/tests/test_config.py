import importlib
import sys
import types
from pathlib import Path
import textwrap
import yaml
import pytest

import config as cfg


# --------------------
# Helpers for tests
# --------------------

def make_tmp_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def install_dummy_module(mod_name: str = "dummy_mod"):
    """Install a tiny module in sys.modules with a Dummy class for import tests."""
    m = types.ModuleType(mod_name)

    class Dummy:
        def __init__(self, **kw):
            self.kw = dict(kw)

    m.Dummy = Dummy
    sys.modules[mod_name] = m
    return m


# --------------------
# _load_yaml
# --------------------

def test_load_yaml_mapping(tmp_path: Path):
    p = make_tmp_yaml(tmp_path, "a.yaml", "a: 1\nb: {c: 2}")
    out = cfg._load_yaml(p)
    assert out == {"a": 1, "b": {"c": 2}}  # mapping required【5:5†turn3file5†L45-L51】


def test_load_yaml_empty_returns_empty_mapping(tmp_path: Path):
    p = make_tmp_yaml(tmp_path, "empty.yaml", "")
    assert cfg._load_yaml(p) == {}


def test_load_yaml_non_mapping_raises_type_error(tmp_path: Path):
    p = make_tmp_yaml(tmp_path, "list.yaml", "- 1\n- 2")
    with pytest.raises(TypeError):
        cfg._load_yaml(p)  # must be mapping【5:5†turn3file5†L48-L51】


def test_load_yaml_invalid_yaml_raises(tmp_path: Path):
    p = make_tmp_yaml(tmp_path, "bad.yaml", "a: [1, 2")  # unclosed list
    with pytest.raises(yaml.YAMLError):
        cfg._load_yaml(p)  # YAML parse error【3:4†turn3file4†L5-L6】


def test_load_yaml_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        cfg._load_yaml("/no/such/file.yaml")  # not found【3:4†turn3file4†L2-L2】


def test_load_yaml_directory_raises(tmp_path: Path):
    with pytest.raises(IsADirectoryError):
        cfg._load_yaml(tmp_path)  # path is a directory【3:4†turn3file4†L4-L4】


# --------------------
# _deep_update
# --------------------

def test_deep_update_merges_recursively():
    base = {"a": {"x": 1}, "b": 1}
    override = {"a": {"y": 2}, "b": 3}
    out = cfg._deep_update(base, override)
    assert out == {"a": {"x": 1, "y": 2}, "b": 3}  # recursive merge【5:5†turn3file5†L73-L80】
    # inputs not mutated
    assert base == {"a": {"x": 1}, "b": 1} and override == {"a": {"y": 2}, "b": 3}


# --------------------
# load_and_merge
# --------------------

def test_load_and_merge_merges_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p1 = make_tmp_yaml(tmp_path, "1.yaml", "a: {x: 1}\nb: 1\n")
    p2 = make_tmp_yaml(tmp_path, "2.yaml", "a: {y: 2}\nb: 2\n")
    out = cfg.load_and_merge([p1, p2])
    assert out == {"a": {"x": 1, "y": 2}, "b": 2}  # later overrides earlier【3:9†turn3file9†L56-L59】


def test_load_and_merge_empty_list_returns_empty_dict():
    assert cfg.load_and_merge([]) == {}  # explicit behavior【5:5†turn3file5†L96-L98】


# --------------------
# _substitute_placeholders
# --------------------

def test_substitute_placeholders_happy_path():
    s = cfg._substitute_placeholders("Hello ${name}", {"name": "Alice"})
    assert s == "Hello Alice"  # example in docstring【3:9†turn3file9†L77-L80】


def test_substitute_placeholders_non_str_replacement_raises():
    with pytest.raises(TypeError):
        cfg._substitute_placeholders("x=${n}", {"n": 3})  # replace expects str【3:9†turn3file9†L73-L76】


def test_substitute_placeholders_wrong_vars_type_raises_attribute_error():
    with pytest.raises(AttributeError):
        cfg._substitute_placeholders("x", None)  # vars must support .items()【3:9†turn3file9†L61-L66】


# --------------------
# _import_object
# --------------------

def test_import_object_valid():
    assert cfg._import_object("math.sqrt").__name__ == "sqrt"  # valid dotted path【3:7†turn3file7†L13-L21】


def test_import_object_path_without_dot_raises():
    with pytest.raises(ValueError):
        cfg._import_object("math")  # must contain a dot【3:7†turn3file7†L19-L21】


def test_import_object_module_not_found_raises():
    with pytest.raises(ModuleNotFoundError):
        cfg._import_object("nope_xxx.mod")


def test_import_object_attribute_error_raises():
    with pytest.raises(AttributeError):
        cfg._import_object("math.nopeattr")


# --------------------
# _make_estimator
# --------------------

def test_make_estimator_happy_path(monkeypatch: pytest.MonkeyPatch):
    install_dummy_module()
    spec = {"class": "dummy_mod.Dummy", "params": {"a": 1, "b": 2}}
    inst = cfg._make_estimator(spec)
    assert inst.__class__.__name__ == "Dummy" and inst.kw == {"a": 1, "b": 2}  # constructs object【3:7†turn3file7†L58-L63】


def test_make_estimator_bad_spec_type_raises():
    with pytest.raises(TypeError):
        cfg._make_estimator("not-a-dict")  # must be mapping with 'class'【3:7†turn3file7†L58-L60】


def test_make_estimator_missing_class_module_raises():
    with pytest.raises(ModuleNotFoundError):
        cfg._make_estimator({"class": "no.mod.Class"})


def test_make_estimator_missing_attr_raises(monkeypatch: pytest.MonkeyPatch):
    install_dummy_module("another_mod")
    with pytest.raises(AttributeError):
        cfg._make_estimator({"class": "another_mod.Nope"})


def test_make_estimator_bad_params_type_raises(monkeypatch: pytest.MonkeyPatch):
    # Use a strict built-in that rejects kwargs to reliably trigger a TypeError across Python versions.
    with pytest.raises(TypeError):
        cfg._make_estimator({"class": "builtins.object", "params": {"bad": 1}})

# --------------------
# _label_for
# --------------------

def test_label_for_from_dict_and_string():
    assert cfg._label_for({"class": "sklearn.tree.DecisionTreeClassifier"}) == "DecisionTreeClassifier"  #【3:1†turn3file1†L34-L38】
    assert cfg._label_for("sklearn.linear_model.LogisticRegression(max_iter=200)") == "LogisticRegression"  #【3:1†turn3file1†L40-L43】


def test_label_for_other_returns_final():
    assert cfg._label_for(123) == "final"  # fallback【3:1†turn3file1†L41-L43】


def test_label_for_wrong_class_type_raises():
    with pytest.raises(TypeError):
        cfg._label_for({"class": 123})  # class must be str【3:1†turn3file1†L34-L38】


# --------------------
# _expand_seeds
# --------------------

def test_expand_seeds_all_cases():
    assert cfg._expand_seeds({"random_seed": {"base": 100, "iterations": 3}}) == [100, 101, 102]  #【3:1†turn3file1†L74-L82】
    assert cfg._expand_seeds({"random_seed": {"base": 7}}) == [7]  #【3:6†turn3file6†L29-L34】
    assert cfg._expand_seeds({"random_seed": {"iterations": 3}}) == [0, 1, 2]  #【3:6†turn3file6†L31-L34】
    assert cfg._expand_seeds({"random_seed": {}}) == [0]


def test_expand_seeds_type_errors_and_missing_key():
    with pytest.raises(KeyError):
        cfg._expand_seeds({})  # must contain 'random_seed'【3:6†turn3file6†L36-L38】
    with pytest.raises(TypeError):
        cfg._expand_seeds({"random_seed": {"base": "x"}})  # base must be int or None【3:6†turn3file6†L43-L45】
    with pytest.raises(TypeError):
        cfg._expand_seeds({"random_seed": {"iterations": 2.5}})  # iterations must be int or None【3:6†turn3file6†L45-L47】


# --------------------
# Validation helpers
# --------------------

def test_require_keys_and_errors():
    cfg._require_keys({"a": 1}, ["a"])  # ok
    with pytest.raises(ValueError):
        cfg._require_keys({"a": 1}, ["b"])  # missing key【1:7†turn2file7†L62-L65】


def test_ensure_type_and_one_of():
    cfg._ensure_type(1, int, "x")  # ok
    with pytest.raises(ValueError):
        cfg._ensure_type("1", int, "x")  # error message mentions context【1:7†turn2file7†L81-L89】

    cfg._ensure_one_of("skf", ["skf", "kf"], "cv.type")  # ok【3:14†turn3file14†L1-L6】
    with pytest.raises(ValueError):
        cfg._ensure_one_of("bad", ["skf", "kf"], "cv.type")  # bad choice【3:14†turn3file14†L1-L9】


# --------------------
# _validate_estimator_spec
# --------------------

def test_validate_estimator_spec_cases():
    # valid
    cfg._validate_estimator_spec(
        {"class": "pathlib.Path", "params": {"path": "."}}, "ctx")  # ok【3:14†turn3file14†L40-L55】

    # errors
    with pytest.raises(ValueError):
        cfg._validate_estimator_spec({}, "ctx")  # missing 'class'【3:14†turn3file14†L40-L44】
    with pytest.raises(ValueError):
        cfg._validate_estimator_spec({"class": 123}, "ctx")  # class must be str【3:14†turn3file14†L41-L44】
    with pytest.raises(ValueError):
        cfg._validate_estimator_spec({"class": "X", "params": []}, "ctx")  # params must be mapping or None【3:14†turn3file14†L45-L49】
    with pytest.raises(ValueError):
        cfg._validate_estimator_spec({"class": "X", "extra": 1}, "ctx")  # unknown keys rejected【1:9†turn2file9†L4-L9】


# --------------------
# _validate_dataset
# --------------------

def test_validate_dataset_valid_and_errors():
    cfg._validate_dataset({"name": "mnist", "params": {"csv_path": "data.csv"}})  # ok【1:9†turn2file9†L36-L45】
    with pytest.raises(ValueError):
        cfg._validate_dataset({"name": "mnist", "params": {}})  # missing csv_path【1:9†turn2file9†L41-L45】


# --------------------
# _validate_data
# --------------------

def test_validate_data_valid_and_errors():
    valid = {"test_size": 0.2, "chunk_size": None, "subsample": None,
             "noise": {"active": False, "name": "n", "params": None, "target": None}}
    cfg._validate_data(valid)  # ok【1:4†turn2file4†L21-L31】【1:1†turn2file1†L13-L18】

    with pytest.raises(ValueError):
        cfg._validate_data({"test_size": 0})  # out of (0,1)【1:4†turn2file4†L28-L31】
    with pytest.raises(ValueError):
        cfg._validate_data({"test_size": "0.2"})  # type error【1:9†turn2file9†L88-L91】
    with pytest.raises(ValueError):
        cfg._validate_data({"test_size": 0.2, "chunk_size": -1})  # positive int or None【1:4†turn2file4†L33-L37】
    with pytest.raises(ValueError):
        cfg._validate_data({"test_size": 0.2, "subsample": 0})  # positive int or None【1:4†turn2file4†L39-L43】
    with pytest.raises(ValueError):
        cfg._validate_data({"test_size": 0.2, "noise": {"active": True, "name": "n", "params": None, "target": "bad"}})  # bad target【1:4†turn2file4†L56-L60】


# --------------------
# _validate_views
# --------------------

def test_validate_views_valid_and_errors():
    names = cfg._validate_views({"available": [{"name": "raw"}], "always_include": ["raw"], "permutations": False})
    assert names == ["raw"]  # returns defined view names【1:12†turn3file12†L20-L28】【1:12†turn3file12†L55-L62】

    with pytest.raises(ValueError):
        cfg._validate_views({})  # missing 'available'【1:12†turn3file12†L55-L57】

    with pytest.raises(ValueError):
        cfg._validate_views({"available": ["not-dict"]})  # entry must be dict with name【1:12†turn3file12†L61-L66】

    with pytest.raises(ValueError):
        cfg._validate_views({"available": [{"name": ""}]})  # non-empty string【1:12†turn3file12†L65-L66】

    with pytest.raises(ValueError):
        cfg._validate_views({"available": [{"name": "v", "params": []}]})  # params must be mapping or null【1:12†turn3file12†L68-L70】

    with pytest.raises(ValueError):
        cfg._validate_views({"available": [{"name": "A"}], "always_include": ["B"]})  # unknown include【1:12†turn3file12†L79-L81】

    with pytest.raises(ValueError):
        cfg._validate_views({"available": [{"name": "A"}], "permutations": "x"})  # bool required【1:12†turn3file12†L83-L86】


# --------------------
# _validate_models
# --------------------

def test_validate_models_valid_and_errors():
    view_names = ["raw", "grid"]
    # valid
    cfg._validate_models(
        {
            "base": {
                "default": {"class": "pathlib.Path"},
                "by_view": {"raw": {"class": "pathlib.Path"}},
            },
            "final": {
                "choices": [{"class": "pathlib.Path"}],
                "select": 0,
            },
        },
        view_names,
    )  # ok【1:11†turn2file11†L1-L13】【1:11†turn2file11†L14-L37】

    # bad by_view key
    with pytest.raises(ValueError):
        cfg._validate_models(
            {
                "base": {"default": {"class": "pathlib.Path"}, "by_view": {"nope": {"class": "pathlib.Path"}}},
                "final": {"choices": [{"class": "pathlib.Path"}], "select": 0},
            },
            view_names,
        )  # unknown view name【1:11†turn2file11†L9-L13】

    # empty choices
    with pytest.raises(ValueError):
        cfg._validate_models(
            {
                "base": {"default": {"class": "pathlib.Path"}},
                "final": {"choices": [], "select": 0},
            },
            view_names,
        )  # choices must be non-empty【1:11†turn2file11†L21-L23】

    # bad select type
    with pytest.raises(ValueError):
        cfg._validate_models(
            {
                "base": {"default": {"class": "pathlib.Path"}},
                "final": {"choices": [{"class": "pathlib.Path"}], "select": 0.5},
            },
            view_names,
        )  # select must be int or 'all'【1:11†turn2file11†L27-L36】


# --------------------
# _validate_cv / _validate_metafeatures / _validate_hardware / _validate_paths
# --------------------

def test_validate_cv_valid_and_errors():
    cfg._validate_cv({"type": "skf", "n_splits": 2, "shuffle": True})  # ok【1:11†turn2file11†L62-L75】
    with pytest.raises(ValueError):
        cfg._validate_cv({"type": "kf", "n_splits": 1, "shuffle": False})  # n_splits >=2【1:11†turn2file11†L68-L71】


def test_validate_metafeatures_valid_and_errors():
    cfg._validate_metafeatures({"name": "avg_proba_ohe"})  # ok【1:8†turn2file8†L55-L61】
    with pytest.raises(ValueError):
        cfg._validate_metafeatures({"name": "invalid"})  # must be allowed【1:8†turn2file8†L55-L60】


def test_validate_hardware_and_paths_and_exp():
    cfg._validate_hardware(4)  # ok【1:8†turn2file8†L83-L85】
    with pytest.raises(ValueError):
        cfg._validate_hardware("auto")  # must be int【1:8†turn2file8†L76-L82】

    cfg._validate_paths("out", "logs")  # ok【1:14†turn3file14†L52-L54】
    with pytest.raises(ValueError):
        cfg._validate_paths(1, "logs")  # output_dir must be str【1:14†turn3file14†L46-L53】

    cfg._validate_exp("mnist_exp1")  # ok【1:14†turn3file14†L69-L77】
    with pytest.raises(ValueError):
        cfg._validate_exp(123)  # must be str【1:14†turn3file14†L75-L77】
    with pytest.raises(ValueError):
        cfg._validate_exp("")  # non-empty【1:14†turn3file14†L75-L77】


# --------------------
# _validate_config (integration of validators)
# --------------------

def test_validate_config_minimal_valid(monkeypatch: pytest.MonkeyPatch):
    install_dummy_module()
    minimal = {
        "dataset": {"name": "mnist", "params": {"csv_path": "data.csv"}},
        "data": {"test_size": 0.2, "chunk_size": None, "subsample": None, "noise": {"active": False, "name": "n", "params": None, "target": None}},
        "views": {"available": [{"name": "raw"}]},
        "models": {
            "base": {"default": {"class": "pathlib.Path"}},
            "final": {"choices": [{"class": "pathlib.Path"}], "select": 0},
        },
        "cv": {"type": "skf", "n_splits": 2, "shuffle": True},
        "metafeatures": {"name": "avg_proba_ohe"},
        "n_jobs": 1,
        "output_dir": "out",
        "logs_dir": "logs",
        "exp_name": "exp",
        "random_seed": {"base": 0, "iterations": 1},
    }
    assert cfg._validate_config(minimal) is None  # orchestrates sub-validators【1:7†turn2file7†L26-L39】


# --------------------
# resolve_config (smoke tests)
# --------------------
def test_resolve_config_smoke(monkeypatch: pytest.MonkeyPatch):
    # See apparent issues: vars_map malformed【0:0†turn3file0†L67-L67】, subst_path body malformed【0:0†turn3file0†L69-L71】,
    # and 'final_est' used without being set【0:3†turn3file3†L70-L75】.
    install_dummy_module()
    raw = {
        "dataset": {"name": "mnist", "params": {"csv_path": "data.csv"}},
        "data": {"test_size": 0.2, "chunk_size": None, "subsample": None, "noise": {"active": False, "name": "n", "params": None, "target": None}},
        "views": {"available": [{"name": "raw"}]},
        "models": {
            "base": {"default": {"class": "dummy_mod.Dummy"}},
            "final": {"choices": [{"class": "dummy_mod.Dummy"}], "select": 0},
        },
        "cv": {"type": "skf", "n_splits": 2, "shuffle": True},
        "metafeatures": {"name": "avg_proba_ohe"},
        "n_jobs": 1,
        "output_dir": "out",
        "logs_dir": "logs",
        "exp_name": "exp",
        "random_seed": {"base": 7, "iterations": 2},
    }
    ec = cfg.resolve_config(raw)
    assert isinstance(ec.cfg, dict)
    assert ec.seeds == [7, 8]  # expanded seeds【3:6†turn3file6†L48-L54】
    assert isinstance(ec.base_default, object)
    assert ec.base_est_by_view == {}
    assert ec.meta_name == "avg_proba_ohe"  # defaulting handled earlier【0:3†turn3file3†L66-L69】
    assert ec.output_dir == Path("out") and ec.logs_dir == Path("logs")
