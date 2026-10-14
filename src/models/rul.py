"""Readable model construction, engine folds and reusable RUL inference."""

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.clean import SensorPreprocessor
from src.data.features import build_features, capped_rul
from src.data.load import validate_trajectories

MODEL_NAMES = ("ridge", "random_forest", "lightgbm")
METADATA_COLUMNS = ["unit", "cycle"]


def make_model(name: str, config: dict[str, Any]) -> RegressorMixin:
    """Construct a fresh CPU model; tuning values come from YAML."""
    training = config["training"]
    if name == "ridge":
        # Derived features have different scales; this scaler is fitted by
        # model.fit on the fold's TRAIN engineered features only.
        return make_pipeline(StandardScaler(), Ridge(**training[name]))
    if name == "random_forest":
        return RandomForestRegressor(
            **training[name], random_state=config["seed"], n_jobs=training["n_jobs"]
        )
    if name == "lightgbm":
        return LGBMRegressor(
            **training[name], random_state=config["seed"], n_jobs=training["n_jobs"],
            objective="regression", deterministic=True, force_col_wise=True, verbosity=-1,
        )
    raise ValueError(f"Unknown RUL model: {name}")


def engine_folds(raw: pd.DataFrame, config: dict[str, Any]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield row positions while keeping every engine in exactly one fold."""
    settings = config["training"]
    if settings["cv_splits"] > raw["unit"].nunique():
        raise ValueError("cv_splits cannot exceed the number of training engines.")
    splitter = GroupKFold(
        n_splits=settings["cv_splits"], shuffle=settings["cv_shuffle"],
        random_state=config["seed"] if settings["cv_shuffle"] else None,
    )
    yield from splitter.split(raw, groups=raw["unit"])


def transform_features(
    raw: pd.DataFrame, processor: SensorPreprocessor, config: dict[str, Any]
) -> pd.DataFrame:
    """Build features with an already fitted preprocessor; never learn here."""
    features = build_features(
        processor.transform(raw), processor.sensor_columns,
        config["features"]["windows"], config["features"]["baseline_cycles"],
    )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("RUL features contain nonfinite values.")
    return features


@dataclass
class PreparedFold:
    """Bundle the fold's independently fitted preprocessing and aligned data."""

    processor: SensorPreprocessor
    train: pd.DataFrame
    validation: pd.DataFrame
    train_target: np.ndarray
    validation_target: np.ndarray


def prepare_fold(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> PreparedFold:
    """Fit all preprocessing on training engines before transforming held-out engines."""
    train = validate_trajectories(train)
    validation = validate_trajectories(validation)
    if set(train["unit"]) & set(validation["unit"]):
        raise ValueError("Engine leakage: train and validation engine IDs overlap.")
    processor = SensorPreprocessor(config["cleaning"]).fit(train)
    return PreparedFold(
        processor=processor,
        train=transform_features(train, processor, config),
        validation=transform_features(validation, processor, config),
        train_target=capped_rul(train, config["target"]["rul_cap"]).to_numpy(),
        validation_target=capped_rul(validation, config["target"]["rul_cap"]).to_numpy(),
    )


def bound_predictions(predictions: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Optionally clip outputs to the capped target domain; never clip test truth."""
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Model returned nonfinite predictions.")
    if config["training"]["clip_predictions"]:
        return np.clip(predictions, 0., config["target"]["rul_cap"])
    return predictions


@dataclass
class RULPredictor:
    """Save model, feature order, preprocessing and config together for inference."""

    model_name: str
    model: Any
    processor: SensorPreprocessor
    feature_columns: list[str]
    config: dict[str, Any]

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Predict each observed engine cycle from its full available history."""
        features = transform_features(raw, self.processor, self.config)
        predicted = self.model.predict(features[self.feature_columns])
        result = features[METADATA_COLUMNS].copy()
        result["predicted_rul"] = bound_predictions(predicted, self.config)
        return result
