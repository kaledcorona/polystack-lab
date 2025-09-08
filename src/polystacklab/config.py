"""
Configuration utilities for PolystackLab

Provides functions to load, merge, and resolve YAML configurations for
multiview learning experiments. Used by the experiment runner to parse
settings from files like `mnist.yaml`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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

def _expand_seeds(cfg: dict[str, Any]) -> list[int]:
    """
    Expand a random seed configuration into a list of integer seeds.

    The configuration is expected to be under the 'random_seed' key with
    optional 'base' and 'iterations' entries:

    Example YAML:
        random_seed:
            base: 100       # optional, defaults to 0
            iterations: 3   # optional, defauls to 1

    Behavior:
        * If both 'base' and 'iterations' are provided, returns a sequential
        list of seeds starting at 'base' with 'iterations' values.
        * If only 'base' is provided, returns '[base]'.
        * If only 'iterations' is provided, returns '[0,1,2,...]'.
        * If neither is provided, returns '[0]'.

    Args:
        cfg (dict[str, Any]): Configuration dictionary containing a 'random_seed' block.

    Returns:
        List[int]: Expanded list of seeds.

    Raises:
        TypeError: If ``base`` or ``iterations`` are not integers or ``None``.
        KeyError: If ``"random_seed"`` is missing from ``cfg``.

    Examples:
        >>> _expand_seeds({"random_seed": {"base": 100, "iterations": 3}})
        [100, 101, 102]
        >>> _expand_seeds({"random_seed": {"base": 7}})
        [7]
        >>> _expand_seeds({"random_seed": {"iterations": 3}})
        [0, 1, 2]
        >>> _expand_seeds({"random_seed": {}})
        [0]
    """
    if "random_seed" not in cfg:
        raise KeyError("Missing 'random_seed' in configuration")

    seed_cfg = cfg["random_seed"]
    base = seed_cfg.get("base")
    iterations = seed_cfg.get("iterations")

    if base is not None and not isinstance(base, int):
        raise TypeError(f"Expected int or None for base, got {type(base).__name__}")
    if iterations is not None and not isinstance(iterations, int):
        raise TypeError(f"Expected int or None for iterations, got {type(iterations).__name__}")

    if base is not None and iterations is not None:
        return list(range(base, base + iterations))
    if base is not None:
        return [base]
    if iterations is not None:
        return list(range(iterations))
    return [0]

#########
# Validation
#########

def _validate_config(cfg: dict[str, Any]) -> None:
    """
    Validate YAML configuration.

    Validates that the YAML config has required sections,
    types, allowed values, and cross-field constraints.

    Args:
        cfg: Parsed configuration dict (e.g., from YAML).

    Raises:
        ValueError: If any section/key is missing or malformed.
    """
    _require_keys(cfg, ["dataset", "data", "views", "models", "cv",
                        "metafeatures", "n_jobs", "output_dir", "logs_dir",
                        "exp_name", "random_seed"])

    available_view_names = _validate_views(cfg["views"])
    _validate_dataset(cfg["dataset"])
    _validate_data(cfg["data"])
    _validate_models(cfg["models"], available_view_names)
    _validate_cv(cfg["cv"])
    _validate_metafeatures(cfg["metafeatures"])
    _validate_hardware(cfg["n_jobs"])
    _validate_paths(cfg["output_dir"], cfg["logs_dir"])
    _validate_exp(cfg["exp_name"])
    _validate_random_seed(cfg["random_seed"])
    return None

# ----- validation helpers

def _require_keys(mapping: dict[str, Any], keys: Sequence[str]) -> None:
    """
    Ensure that all required keys are present in a dictionary.

    Args:
        mapping (dict[str, Any]): The configuration or dictionary to check.
        keys (Sequence[str]): The list of keys that must be present.

    Raises:
        ValueError: If any key is missing from ``mapping``.

    Examples:
        >>> _require_keys({"a": 1, "b": 2}, ["a"])
        >>> _require_keys({"a": 1, "b": 2}, ["a", "c"])
        Traceback (most recent call last):
            ...
        ValueError: Missing required config section/key: 'c'
    """
    for key in keys:
        if key not in mapping:
            raise ValueError(f"Missing required config section/key: '{key}'")
    return None

def _ensure_type(value: Any, expected_type: type[Any], context: str) -> None:
    """
    Ensure that a value has the expected type.

    Args:
        value (Any): The value to check.
        expected_type (type[Any]): The expected type.
        context (str): A label identifying the field or context for clearer
            error messages.

    Raises:
        ValueError: If ``value`` is not an instance of ``expected_type``.

    Examples:
        >>> _ensure_type(42, int, "n_splits")
        >>> _ensure_type("abc", int, "n_splits")
        Traceback (most recent call last):
            ...
        ValueError: n_splits must be int; got str
    """
    if not isinstance(value, expected_type):
        raise ValueError(f"{context} must be {expected_type.__name__}; got {type(value).__name__}")
    return None

def _ensure_one_of(value: Any, allowed: Sequence[Any], context: str) -> None:
    """
    Ensure that a value belongs to an allowed set.

    Args:
        value (Any): The value to check.
        allowed (Sequence[Any]): The sequence of allowed values.
        context (str): A label identifying the field or context for clearer
            error messages.

    Raises:
        ValueError: If ``value`` is not one of the allowed values.

    Examples:
        >>> _ensure_one_of("skf", ["skf", "kf"], "cv.type")
        >>> _ensure_one_of("bad", ["skf", "kf"], "cv.type")
        Traceback (most recent call last):
            ...
        ValueError: cv.type must be one of ['skf', 'kf']; got 'bad'
    """
    if value not in allowed:
        raise ValueError(f"{context} must be one of {list(allowed)}; got {value!r}")
    return None

def _validate_estimator_spec(spec: dict[str, Any], context: str) -> None:
    """
    Validate an estimator specification mapping.

    Expected shape:
      - ``class`` (str, required): fully qualified dotted path to the class.
      - ``params`` (dict[str, Any] | None, optional): kwargs for the constructor.

    Args:
        spec: The estimator specification to validate.
        context: Label used in error messages to indicate where the spec lives
            (e.g., "models.base.default" or "models.final.choices[0]").

    Raises:
        ValueError: If required keys are missing, types are incorrect, or
            unexpected keys are present.

    Examples:
        >>> _validate_estimator_spec(
        ...     {"class": "sklearn.linear_model.LogisticRegression",
        ...      "params": {"max_iter": 1000}},
        ...     "models.final.choices[0]"
        ... )
        >>> _validate_estimator_spec({"class": 123}, "models.base.default")
        Traceback (most recent call last):
            ...
        ValueError: models.base.default.class must be str; got int
    """
    # Required keys
    _require_keys(spec, ["class"])
    _ensure_type(spec["class"], str, f"{context}.class")

    # Optional params
    params = spec.get("params", None)

    if params is not None and not isinstance(params, dict):
        raise ValueError(f"{context}.params must be a mapping or None; got {type(params).__name__}")
    
    # Guard against typos
    allowed = {"class", "params"}
    unknown = set(spec.keys()) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
    return None

def _validate_dataset(dataset: dict[str, Any]) -> None:
    """
    Validate that a dataset config contains required keys and correct types.

    Expected schema:
      - ``name`` (str, required): Dataset identifier.
      - ``params`` (Mapping[str, Any], required):
          - ``csv_path`` (str, required): Path to dataset file.

    Args:
        dataset: The dataset configuration mapping to validate.

    Raises:
        ValueError: If required keys are missing or types are incorrect.

    Examples:
        >>> valid = {"name": "mnist", "params": {"csv_path": "data/mnist.csv"}}
        >>> _validate_dataset(valid)

        >>> invalid = {"name": "mnist", "params": {}}
        >>> _validate_dataset(invalid)
        Traceback (most recent call last):
            ...
        ValueError: Missing required config section/key: 'csv_path'
    """
    # Required top-level keys
    _require_keys(dataset, ["name", "params"])
    _ensure_type(dataset["name"], str, "dataset.name")
    _ensure_type(dataset["params"], dict, "dataset.params")

    # Required params
    params = dataset["params"]
    _require_keys(params, ["csv_path"])
    _ensure_type(params["csv_path"], str, "dataset.params.csv_path")
    return None

def _validate_data(data: dict[str, Any]) -> None:
    """
    Validate the data preprocessing configuration block.

    Expected schema:
      - ``test_size`` (float in (0, 1), required): Fraction of dataset to use
        as test split.
      - ``chunk_size`` (int > 0 | None, optional): If set, load dataset in
        chunks of this size.
      - ``subsample`` (int > 0 | None, optional): If set, subsample the dataset
        to this many rows.
      - ``noise`` (dict | None, optional):
          - ``active`` (bool, required): Whether noise injection is enabled.
          - ``name`` (str, required): Noise type identifier.
          - ``params`` (dict | None, required): Parameters for noise.
          - ``target`` (str | None, required): One of {"train", "test", "both"}.

    Args:
        data: The configuration mapping for preprocessing.

    Raises:
        ValueError: If any required keys are missing, values are of the wrong
            type, or constraints are violated.

    Examples:
        >>> valid = {
        ...     "test_size": 0.2,
        ...     "chunk_size": None,
        ...     "subsample": None,
        ...     "noise": {
        ...         "active": False,
        ...         "name": "image/gaussian",
        ...         "params": {"sigma": 0.05},
        ...         "target": None,
        ...     },
        ... }
        >>> _validate_data(valid)
    """
    # --- test_size ---
    _require_keys(data, ["test_size"])
    test_size = data["test_size"]
    if not isinstance(test_size, (int, float)):
        raise ValueError(
            f"data.test_size must be a number in (0,1); got {type(test_size).__name__}"
        )
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"data.test_size must be in (0,1); got {test_size!r}")

    # --- chunk_size ---
    chunk_size = data.get("chunk_size")
    if chunk_size is not None:
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("data.chunk_size must be a positive int or null")

    # --- subsample ---
    subsample = data.get("subsample")
    if subsample is not None:
        if not isinstance(subsample, int) or subsample <= 0:
            raise ValueError("data.subsample must be a positive int or null")

    # --- noise ---
    noise = data.get("noise")
    if noise is not None:
        _ensure_type(noise, dict, "data.noise")
        _require_keys(noise, ["active", "name", "params", "target"])

        _ensure_type(noise["active"], bool, "data.noise.active")
        _ensure_type(noise["name"], str, "data.noise.name")

        if noise["params"] is not None:
            _ensure_type(noise["params"], dict, "data.noise.params")

        allowed_targets = ("train", "test", "both", None)
        if noise["target"] not in allowed_targets:
            raise ValueError(
                f"data.noise.target must be one of {allowed_targets}; got {noise['target']!r}"
            )
    return None

def _validate_views(views: dict[str, Any]) -> list[str]:
    """
    Validate the views configuration block.

    Expected schema:
      - ``available`` (list[dict], required): Each entry must contain:
          - ``name`` (str, required): Name of the view (non-empty).
          - ``params`` (dict | None, optional): Parameters for the view.
      - ``always_include`` (list[str] | None, optional): View names that must
        always be included. All names must exist in ``available``.
      - ``permutations`` (bool | None, optional): Whether to generate all
        permutations of views containing those in ``always_include``.

    Args:
        views: The configuration mapping for views.

    Returns:
        list[str]: A list of view names defined in ``views.available``.

    Raises:
        ValueError: If required keys are missing, types are incorrect, or
            references in ``always_include`` are invalid.

    Examples:
        >>> valid = {
        ...     "available": [
        ...         {"name": "raw"},
        ...         {"name": "grid", "params": {"rows": 3, "cols": 3}},
        ...     ],
        ...     "always_include": ["raw"],
        ...     "permutations": True,
        ... }
        >>> _validate_views(valid)
        ['raw', 'grid']
    """
    # --- available views ---
    _require_keys(views, ["available"])
    available = views["available"]
    _ensure_type(available, list, "views.available")

    view_names: list[str] = []
    for i, view in enumerate(available):
        if not isinstance(view, dict):
            raise ValueError(f"views.available[{i}] must be a dict with 'name'")

        if "name" not in view or not isinstance(view["name"], str) or not view["name"]:
            raise ValueError(f"views.available[{i}].name must be a non-empty string")

        if "params" in view and view["params"] is not None and not isinstance(view["params"], Mapping):
            raise ValueError(f"views.available[{i}].params must be a mapping or null")

        view_names.append(view["name"])

    # --- always_include ---
    always_include = views.get("always_include", [])
    if always_include is not None:
        _ensure_type(always_include, list, "views.always_include")
        for i, name in enumerate(always_include):
            _ensure_type(name, str, f"views.always_include[{i}]")
            if name not in view_names:
                raise ValueError(f"views.always_include contains unknown view '{name}'")

    # --- permutations ---
    permutations = views.get("permutations", False)
    if permutations is not None and not isinstance(permutations, bool):
        raise ValueError("views.permutations must be a boolean")

    return view_names

def _validate_models(models: dict[str, Any], view_names: Sequence[str]) -> None:
    """
    Validate the structure of the 'models' configuration.

    Ensures required keys are present and correctly typed for base and final models.

    Args:
        models (dict[str, Any]): The models configuration.
        view_names (Sequence[str]): List of valid view names for `by_view`.

    Raises:
        ValueError: If required keys are missing or values are of invalid type.
    """
    # Validate base section
    _require_keys(models, ["base", "final"])
    base = models["base"]
    _ensure_type(base, dict, "models.base")

    _require_keys(base, ["default"])
    _ensure_type(base["default"], dict, "models.base.default")
    _validate_estimator_spec(base["default"], "models.base.default")

    by_view = base.get("by_view", {})
    if by_view is not None:
        _ensure_type(by_view, dict, "models.base.by_view")
        for view, spec in by_view.items():
            if view not in view_names:
                raise ValueError(f"models.base.by_view contains unknown view '{view}'")
            _ensure_type(spec, dict, f"models.base.by_view['{view}']")
            _validate_estimator_spec(spec, f"models.base.by_view['{view}']")

    # Validate final section
    final = models["final"]
    _ensure_type(final, dict, "models.final")
    _require_keys(final, ["choices", "select"])

    choices = final["choices"]
    _ensure_type(choices, list, "models.final.choices")
    if not choices:
        raise ValueError("models.final.choices must be a non-empty list")
    for i, spec in enumerate(choices):
        _ensure_type(spec, dict, f"models.final.choices[{i}]")
        _validate_estimator_spec(spec, f"models.final.choices[{i}]")

    select = final["select"]
    if isinstance(select, int):
        if not (0 <= select < len(choices)):
            raise ValueError(
                f"models.final.select index out of range 0..{len(choices) - 1}: {select}"
            )
    elif isinstance(select, str):
        _ensure_one_of(select, ["all"], "models.final.select")
    else:
        raise ValueError("models.final.select must be an int index or 'all'")
    return None

def _validate_cv(cv: dict[str, Any]) -> None:
    """
    Validate the cross-validation configuration block.

    Expected schema:
      - ``type`` (str, required): One of {"skf", "kf", "group", "timeseries"}.
      - ``n_splits`` (int, required): Number of folds, must be >= 2.
      - ``shuffle`` (bool, required): Whether to shuffle before splitting.

    Args:
        cv: Cross-validation configuration dictionary.

    Raises:
        ValueError: If required keys are missing, types are incorrect, or values
            violate constraints.

    Examples:
        >>> _validate_cv({"type": "skf", "n_splits": 5, "shuffle": True})
        >>> _validate_cv({"type": "kf", "n_splits": 1, "shuffle": False})
        Traceback (most recent call last):
            ...
        ValueError: cv.n_splits must be >= 2
    """
    # Required keys
    _require_keys(cv, ["type", "n_splits", "shuffle"])

    # Validate type
    _ensure_one_of(cv["type"], ["skf", "kf", "group", "timeseries"], "cv.type")

    # Validate n_splits
    _ensure_type(cv["n_splits"], int, "cv.n_splits")
    if cv["n_splits"] < 2:
        raise ValueError("cv.n_splits must be >= 2")

    # Validate shuffle
    _ensure_type(cv["shuffle"], bool, "cv.shuffle")
    return None

def _validate_metafeatures(mf: dict[str, Any]) -> None:
    """
    Validate the metafeatures configuration block.

    Expected schema:
      - ``name`` (str, required): Must be one of {"avg_proba_ohe", "proba_margin_entropy"}.

    Args:
        mf: Metafeatures configuration dictionary.

    Raises:
        ValueError: If required keys are missing, the type is invalid, or the
            value is not in the allowed set.

    Examples:
        >>> _validate_metafeatures({"name": "avg_proba_ohe"})
        >>> _validate_metafeatures({"name": "invalid"})
        Traceback (most recent call last):
            ...
        ValueError: metafeatures.name must be one of ['avg_proba_ohe', 'proba_margin_entropy']; got 'invalid'
    """
    # Required key
    _require_keys(mf, ["name"])
    _ensure_type(mf["name"], str, "metafeatures.name")

    # Validate against allowed set
    allowed = {"avg_proba_ohe", "proba_margin_entropy"}
    if mf["name"] not in allowed:
        raise ValueError(
            f"metafeatures.name must be one of {sorted(allowed)}; got {mf['name']!r}"
        )
    return None

def _validate_hardware(n_jobs: Any) -> None:
    """
    Validate the hardware configuration.

    Ensures that ``n_jobs`` is an integer, representing the number of
    parallel workers to use.

    Args:
        n_jobs: The configured number of jobs.

    Raises:
        ValueError: If ``n_jobs`` is not an integer.

    Examples:
        >>> _validate_hardware(4)
        >>> _validate_hardware("auto")
        Traceback (most recent call last):
            ...
        ValueError: n_jobs must be int; got str
    """
    _ensure_type(n_jobs, int, "n_jobs")
    return None

def _validate_paths(output_dir: Any, logs_dir: Any) -> None:
    """
    Validate output and log directory paths in the configuration.

    Both paths must be strings. No filesystem existence checks are performed.

    Args:
        output_dir: Path to the directory where experiment outputs will be stored.
        logs_dir: Path to the directory where logs will be stored.

    Raises:
        ValueError: If either argument is not a string.

    Examples:
        >>> _validate_paths("experiments/run1", "experiments/run1/logs")
        >>> _validate_paths(123, "logs")
        Traceback (most recent call last):
            ...
        ValueError: output_dir must be str; got int
    """
    _ensure_type(output_dir, str, "output_dir")
    _ensure_type(logs_dir, str, "logs_dir")
    return None

def _validate_exp(exp_name: Any) -> None:
    """
    Validate the experiment name configuration.

    Ensures that the experiment name is a non-empty string.

    Args:
        exp_name: The configured experiment name.

    Raises:
        ValueError: If ``exp_name`` is not a string or is empty.

    Examples:
        >>> _validate_exp("mnist_exp1")
        >>> _validate_exp("")
        Traceback (most recent call last):
            ...
        ValueError: exp_name must be a non-empty string
    """
    _ensure_type(exp_name, str, "exp_name")
    if not exp_name:
        raise ValueError("exp_name must be a non-empty string")
    return None

def _validate_random_seed(seed_cfg: dict[str, Any]) -> None:
    """
    Validate the random seed configuration block.

    Expected schema:
      - ``base`` (int | None, required): Starting seed or null.
      - ``iterations`` (int | None, required): Number of sequential seeds to
        generate; must be >= 0 if provided.

    Args:
        seed_cfg: Random seed configuration dictionary.

    Raises:
        ValueError: If required keys are missing, types are incorrect, or
            constraints are violated.

    Examples:
        >>> _validate_random_seed({"base": 100, "iterations": 3})
        >>> _validate_random_seed({"base": None, "iterations": 0})
        >>> _validate_random_seed({"base": "abc", "iterations": 3})
        Traceback (most recent call last):
            ...
        ValueError: random_seed.base must be int or null
    """
    # Required keys
    _require_keys(seed_cfg, ["base", "iterations"])

    base = seed_cfg["base"]
    iterations = seed_cfg["iterations"]

    # Validate base
    if base is not None and not isinstance(base, int):
        raise ValueError("random_seed.base must be int or null")

    # Validate iterations
    if iterations is not None and not isinstance(iterations, int):
        raise ValueError("random_seed.iterations must be int or null")

    if isinstance(iterations, int) and iterations < 0:
        raise ValueError("random_seed.iterations must be >= 0")
    return None

@dataclass
class ExperimentConfig:
    """
    Container for fully resolved experiment settings.

    Attributes:
        cfg: Full (validated) configuration dictionary.
        seeds: Expanded list of random seeds for this run/set of runs.
        base_default: Default base estimator instance, or None if omitted.
        base_est_by_view: Per-view base estimator instances.
        final_est: List of (human_label, estimator_instance) for final stage.
        meta_name: Selected metafeature strategy name.
        output_dir: Destination directory for artifacts.
        logs_dir: Destination directory for logs.
    """
    cfg: dict[str, Any]
    seeds: list[int]
    base_default : Any | None
    base_est_by_view: dict[str, Any]
    final_est: list[tuple[str, Any]]
    meta_name: str
    output_dir: Path
    logs_dir: Path

def resolve_config(raw_cfg: dict[str, Any]) -> ExperimentConfig:
  """
  Resolve a raw configuration into a structured `ExperimentConfig`.

    Steps:
      1) Validate the raw configuration.
      2) Expand `random_seed` into a concrete list of integers.
      3) Substitute placeholders in output paths (`${exp_name}`, `${now}`).
      4) Instantiate base and final estimator(s).

    Args:
        raw_cfg: Raw configuration dictionary (merged from YAML files).

    Returns:
        ExperimentConfig: The resolved experiment configuration.

    Raises:
        ValueError: For invalid configuration (via validators) or invalid selection index.
        TypeError: For invalid estimator specifications (via `_make_estimator`).
    """
    # 1) Validate & normalize
    _validate_config(raw_cfg)
    cfg: dict[str, Any] = dict(raw_cfg)  # shallow copy

    # 2) Seeds
    seeds = _expand_seeds(cfg)

    # 3) Pahs with placeholders
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("exp_name", "exp")
    vars_map: dict[str, str]  = {"exp_name": exp_name, "now", now}

    def subst_path(s: str, default: str) -> Path:
        raw s if isinstance(s, str) else default
        return Path(_substitute_placeholders(raw, vars_map))

    output_dir = subst_path(cfg.get("output_dir", ""), "experiments/runs/${exp_name}/${now}")
    logs_dir = subst_path(cfg.get("logs_dir", ""), "experiments/runs/${exp_name}/logs")

    # 2) Models
    models = cfg.get("models", {})
    base_cfg: dict[str, Any] = models.get("base", {}) or {}
    final_cfg: dict[str, Any] = models.get("final", {}) or {}

    # 4a) Base: default + per-view overrides
    base_default = _make_estimator(base_cfg.get("default")) if base_cfg.get("default") else None
    base_est_by_view: dict[str, Any] = {
        view: _make_estimator(spec) for view, spec in base_cfg.get("by_view", {}).items()
    }

    # 4b) Final: choices + select
    choices: list[dict[str, Any]] = list(final_cfg.get("choices") or [])
    select = final_cfg.get("select", 0)

    if select == "all":
        final_specs = choices
    else:
        # robust index handling
        try:
            idx = int(select)
        except (TypeError, ValueError) as e:
            raise ValueError(f"models.final.select must be an int index or 'all'; got {select!r}") from e
        if not choices:
            # allow single final spec fallback if provided (legacy shape)
            single = final_cfg if "class" in final_cfg else None
            if single is None:
                raise ValueError("models.final.choices is empty and no single final spec provided")
            final_specs = [single]
        else:
            if not (0 <= idx < len(choices)):
                raise ValueError(
                    f"models.final.select index {idx} out of range 0..{len(choices)-1}"
                )
            final_specs = [choices[idx]]

    # 5) Metafeatures (accept legacy key 'meta' as fallback)
    meta_block = cfg.get("metafeatures") or cfg.get("meta") or {}
    meta_name: str = meta_block.get("name", "concat_proba")

    return ExperimentConfig(
        cfg=cfg,
        seeds=seeds,
        base_default=base_default,
        base_est_by_view=base_est_by_view,
        final_est=final_est,
        meta_name=meta_name,
        output_dir=output_dir,
        logs_dir=logs_dir,
    )
