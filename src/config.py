"""Read and validate the Phase 1 settings from YAML."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _positive_integer(value: Any, name: str) -> None:
    """Reject fractional, boolean, zero and negative sizes."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


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
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed config setting: {error}") from error
    return config
