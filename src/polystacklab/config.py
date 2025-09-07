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
    cfg: dict[str, Any] = {}
    for p in paths:
        cfg = _deep_update(cfg, _load_yaml(p))
    return cfg

def _substitute_placeholders(s: str, vars: dict[str, str]) -> str:
    # ${var} expander
    for k, v in vars.items():
        s = s.replace(f"${{{k}}}", v)
    return s

def _import_object(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

def _make_estimator(spec: Any):
    if isinstance(spec, dict) and "class" in spec:
        cls = _import_object(spec["class"])
        params = dict(spec.get("params", {}))
        return cls(**params)
    raise TypeError(f"Unsupported estimator spec: {spec!r}")

def _label_for(spec: Any) -> str:
    if isinstance(spec, dict) and "class" in spec:
        return spec["class"].split(".")[-1]
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
