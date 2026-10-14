"""Read and validate the data and model settings from YAML."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _positive_integer(value: Any, name: str) -> None:
    """Reject fractional, boolean, zero and negative sizes."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _finite_number(value: Any, name: str, positive: bool = False) -> None:
    """Check numerical hyperparameters without accepting booleans or NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")


def validate_training_config(config: dict[str, Any]) -> None:
    """Fail before expensive training when model settings are malformed."""
    try:
        for path in ("models", "reports"):
            if not isinstance(config["paths"][path], str) or not config["paths"][path]:
                raise ValueError(f"paths.{path} must be a nonempty path string.")
        training = config["training"]
        for key in ("cv_splits", "n_jobs"):
            _positive_integer(training[key], f"training.{key}")
        if training["cv_splits"] < 2:
            raise ValueError("training.cv_splits must be at least 2.")
        for key in ("cv_shuffle", "clip_predictions"):
            if type(training[key]) is not bool:
                raise ValueError(f"training.{key} must be true or false.")
        if training["selection_metric"] not in {"rmse", "mae", "nasa_score"}:
            raise ValueError("selection_metric must be rmse, mae or nasa_score.")
        models = training["models"]
        if not isinstance(models, list) or not models or any(
            name not in ("ridge", "random_forest", "lightgbm") for name in models
        ):
            raise ValueError("training.models must list ridge, random_forest and/or lightgbm.")
        if len(set(models)) != len(models):
            raise ValueError("training.models must contain distinct model names.")
        if "ridge" in models:
            _finite_number(training["ridge"]["alpha"], "ridge.alpha")
            if training["ridge"]["solver"] not in {"auto", "svd", "cholesky"}:
                raise ValueError("ridge.solver must be auto, svd or cholesky.")
        for name in ("random_forest", "lightgbm"):
            if name not in models:
                continue
            params = training[name]
            _positive_integer(params["n_estimators"], f"{name}.n_estimators")
            depth = params["max_depth"]
            if name == "lightgbm" and depth == -1:
                pass  # LightGBM's documented unlimited-depth setting.
            else:
                _positive_integer(depth, f"{name}.max_depth")
        if "random_forest" in models:
            forest = training["random_forest"]
            _positive_integer(forest["min_samples_leaf"], "random_forest.min_samples_leaf")
            _finite_number(forest["max_features"], "random_forest.max_features", positive=True)
            if forest["max_features"] > 1:
                raise ValueError("random_forest.max_features must be a fraction in (0, 1].")
        if "lightgbm" in models:
            light = training["lightgbm"]
            for key in ("num_leaves", "min_child_samples"):
                _positive_integer(light[key], f"lightgbm.{key}")
            if light["num_leaves"] < 2:
                raise ValueError("lightgbm.num_leaves must be at least 2.")
            _finite_number(light["reg_lambda"], "lightgbm.reg_lambda")
            for key in ("learning_rate", "subsample", "colsample_bytree"):
                _finite_number(light[key], f"lightgbm.{key}", positive=True)
            if light["subsample"] > 1 or light["colsample_bytree"] > 1:
                raise ValueError("LightGBM sampling fractions must be at most 1.")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed training setting: {error}") from error


def load_config(path: Path) -> dict[str, Any]:
    """Load explicit settings; fail early when a required setting is invalid."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The config must be a YAML mapping.")
    try:
        seed = config["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        for name in ("raw", "processed", "incoming"):
            if not isinstance(config["paths"][name], str) or not config["paths"][name]:
                raise ValueError(f"paths.{name} must be a nonempty path string.")
        cleaning = config["cleaning"]
        if cleaning["smoothing"] not in {"ewma", "rolling_median"}:
            raise ValueError("cleaning.smoothing must be ewma or rolling_median.")
        alpha = cleaning["ewma_alpha"]
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
            raise ValueError("cleaning.ewma_alpha must be in (0, 1].")
        threshold = cleaning["min_sensor_std"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("cleaning.min_sensor_std must be a finite nonnegative number.")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("cleaning.min_sensor_std must be a finite nonnegative number.")
        _positive_integer(cleaning["median_window"], "cleaning.median_window")
        features = config["features"]
        windows = features["windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError("features.windows must be a nonempty list.")
        for window in windows:
            _positive_integer(window, "features.windows entry")
        if len(set(windows)) != len(windows):
            raise ValueError("features.windows must contain distinct sizes.")
        _positive_integer(features["baseline_cycles"], "features.baseline_cycles")
        _positive_integer(config["target"]["rul_cap"], "target.rul_cap")
        if "training" in config:
            validate_training_config(config)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed config setting: {error}") from error
    return config
