"""
Module Description:
-------------------
Briefly describe what this script/module does.

Author: Your Name
Date: YYYY-MM-DD
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib
import datetime as dt
import yaml


#########
# Helpers
#########

def _load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file into a dictionary.

    Args:
        path (str | Path): Path to the YAML file.
        It must exists, be readable and contain a YAML mapping.

    Returns:
        dict[str, Any]: Parsed YAML content as a dictionary.
        Returns an empty dictionary if the file is empty.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        IsADirectoryError: If `path` is a directory.
        yaml.YAMLError: If the file contains invalid YAML.
        TypeError: If the YAML content is valid but not a mapping.
    """
    txt = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(txt)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping, got {type(data).__name__}")
    return data

def _deep_update(
        base: dict[str, Any],
        override: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two dictionaries recursively.

    keys present in 'override' replace those in 'base' unless both values are
    mappings, in which case they are merged recursively. Inputs are not mutated.

    Args:
        base: Thes base dictionary to start from.
        override: The dictionary whose keys/vales overlay 'base'.

    Returns:
        A new dictionary containing the deep merge of 'base' and 'overide'.

    Raises:
        RecursionError: If nesting exceed Python's recursion limit.

    Examples:
        >>> _deep_update({"a": {"x": 1}}, {"a": {"y": 2}, "b": 3})
        {'a': {'x': 1, 'y': 2}, 'b': 3}
        >>> _deep_update({"a": 1}, {"a": {"x": 2}})
        {'a': {'x': 2}}
        >>> _deep_update({"a": {"x": 1}}, {"a": 7})
        {'a': 7}
    """
    result: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result

def load_and_merge(paths: list[str | Path]) -> dict:
    """
    Load multiple YAML files and deep-merge them.

    Args:
        paths: List of file paths. Order matters: later paths overide earlier.

    Returns:
        A new dictionary with the merged configuration.
        If 'path' is empty, {}.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        IsADirectoryError: If `path` is a directory.
        yaml.YAMLError: If the file contains invalid YAML.
        TypeError: If the YAML content is valid but not a mapping.
        RecursionError: if nested mappings are to deep. 
    """
    cfg: dict[str, Any] = {}
    for p in paths:
        cfg = _deep_update(cfg, _load_yaml(p))
    return cfg

def _substitute_placeholders(s: str, vars: dict[str, str]) -> str:
    """
    Substitute placeholders of the for `${var}` in a string by its value.

    Args:
        s (str): Input string possibly containing placeholders.
        vars (dict[str, Any]): Dict of variables name to replacement values.

    Returns:
        str: String with all placeholders substituted.
            It remains unchange if no placeholders found.

    Raises:
        AttributeError: If wrong types passed in place of str or dict.
        TypeError: if a replacement is not a string.

    Example:
        >>> _substitute_placeholders("Hello ${name}", {"name": "Alice"})
        'Hello Alice'
    """
    for key, value in vars.items():
        s = s.replace(f"${{{key}}}", value)
    return s

def _import_object(dotted: str):
    """
    Import an object given its dotted path.
    
    The path must include both the module and the attribute name, 
    separated by a dot (e.g. `package.module.ClassName`). The module
    is imported and the attribute is returned.

    Args:
        dotted (str): Fully dotted path of the object to import.
            Must contain at least one dot.

    Returns:
        Any: The imported object (class, function, or variable).

    Raises:
        ValueError: If the dotted path does not contain a dot.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the attribute does not exist in the module.

    Examples:
        >>> _import_object("math.sqrt")
        <built-in function sqrt>
        >>> _import_object("pathlib.Path")
        <class 'pathlib.Path'>
    """
    if "." not in dotted:
        raise ValueError(f"Invalid dotted path '{dotted}'; must contain a '.'")

    module_name, attr_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

def _make_estimator(spec: dict[str, Any]) -> Any:
    """
    Instantiate an object from a specification dictionary.

    The specification must be a dictionary containing:
        * 'class': A fully dotted path to the class to import.
        * 'params' (optional): A dictionary of keyword arguments to pass
            to the class constructor.

    Args:
        spec (Any): The specification of the estimator. Must be a dictionary
        with a 'class' key and optionally a 'params' mapping.

    Returns:
        Any: An instance of the specified class, constructed with the given
        parameters.

    Raises:
        TypeError: If ``spec`` is not a dictionary with a ``"class"`` key.
        ModuleNotFoundError: If the module in the dotted path cannot be imported.
        AttributeError: If the class is not found in the module.
        TypeError: If the provided parameters do not match the class
            constructor.

    Examples:
        >>> spec = {
        ...     "class": "sklearn.ensemble.RandomForestClassifier",
        ...     "params": {"n_estimators": 200}
        ... }
        >>> model = _make_estimator(spec)
        >>> model.fit(X, Y)
    """
    if not (isinstance(spec, dict) and "class" in spec):
        raise TypeError(f"Unsupported estimator spec: {spec!r}")

    cls = _import_object(spec["class"])
    params = dict(spec.get("params", {}))
    return cls(**params)

def _label_for(spec: Any) -> str:
    """
    Retrieve a short label from a specification.

    The label is typically the class name at the end of a dotted path.
    If it is a dictionary, the final segment after the last dot is returned.
    Same if it is a string. For all other cases, return 'final' (string).

    Args:
        spec (Any): Either a dict with a 'class' entry, a string or any other
        object.

    Returns:
        str: The segment after the last dot, or 'final' if the
        input does not match the expected formats.

    Raises:
        TypeError: If 'spec['class']' is present but not a string.

    Examples:
        >>> _label_for({"class": "sklearn.tree.DecisionTreeClassifier"})
        'DecisionTreeClassifier'

        >>> _label_for("sklearn.linear_model.LogisticRegression(max_iter=200)")
        'LogisticRegression'

        >>> _label_for(123)
        'final'
    """
    if isinstance(spec, dict) and "class" in spec:
        cls_path = spec["class"]
        if not isinstance(cls_path, str):
            raise TypeError(f"Expected string for 'class', got {type(cls_path).__name__}")
        return cls_path.split(".")[-1]

    if isinstance(spec, str):
        return spec.split("(")[0].split(".")[-1]
    
    return "final"

def _expand_seeds(cfg: dict) -> list[int]:
    # priority: explicit list > seeds.{count,base} > iterations
    if "seeds" in cfg:
        s = cfg["seeds"]
        if isinstance(s, list): return [int(x) for x in s]
        if isinstance(s, dict): return [int(s.get("base", 0)) + i for i in range(int(s.get("count", 1)))]
        return [int(s)]
    if "iterations" in cfg:
        n = int(cfg["iterations"])
        return list(range(n))  # [0..n-1]
    return [0]

def _validate_config(cfg: dict[str, Any]) -> None:
    required = ["dataset", "views", "models"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required config section: {k}")
    views = cfg.get("views", {}).get("available", [])
    if not isinstance(views, list):
        raise ValueError("'views.available' must be a list")
    for v in views:
        if not isinstance(v, dict) or "name" not in v:
            raise ValueError("Each view in 'views.available' must be a dict with 'name'")

@dataclass
class ExperimentConfig:
    cfg: dict[str, Any]
    seeds: list[int]
    base_default : Any | None
    base_est_by_view: dict[str, Any]
    final_est: list[tuple[str, Any]]
    meta_name: str
    output_dir: Path
    logs_dir: Path

def resolve_config(raw_cfg: dict[str, Any]) -> ExperimentConfig:
    _validate_config(raw_cfg)
    cfg = dict(raw_cfg)  # shallow copy
    
    seeds = _expand_seeds(cfg)

    # 1) output paths (expand ${now} , ${exp_name})
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("exp_name", "exp")
    def _subst(s: str) -> str:
        return s.replace("${exp_name}", exp_name).replace("${now}", now)
    output_dir = Path(_subst(cfg.get("output_dir", "experiments/runs/${exp_name}/${now}")))
    logs_dir   = Path(_subst(cfg.get("logs_dir",   "experiments/runs/${exp_name}/logs")))


    # 2) Model instantiation
    models = cfg.get("models", {})
    base = models.get("base", {})
    final = models.get("final", {})

    # base estimator: default + per-view overrides
    base_default = _make_estimator(base["default"]) if base.get("default") else None
    base_by_view = {k: _make_estimator(v) for k, v in base.get("by_view", {}).items()}

     # finals: index or "all"
    final = models.get("final", {})
    choices = final.get("choices", [])
    select = final.get("select", 0)
    if select == "all":
        final_specs = list(choices)
    else:
        idx = int(select)
        final_specs = [choices[idx]] if choices else ([final] if final else [])
    final_ests = [(_label_for(spec), _make_estimator(spec)) for spec in final_specs]


    # metafeatures (accept both keys)
    meta_name = (cfg.get("metafeatures", {}) or cfg.get("meta", {})).get("name", "concat_proba")

    return ExperimentConfig(
        cfg=cfg,
        seeds=seeds,
        base_default=base_default,
        base_by_view=base_by_view,
        final_ests=final_ests,
        meta_name=meta_name,
        output_dir=output_dir,
        logs_dir=logs_dir,
    )
