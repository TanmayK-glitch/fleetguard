"""RUL error metrics and engine-aligned last-cycle test evaluation."""

import numpy as np
import pandas as pd


def _vectors(true: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require matching, finite, nonempty vectors to avoid silent broadcasting."""
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if true.ndim != 1 or predicted.shape != true.shape or not true.size:
        raise ValueError("Expected matching nonempty one-dimensional prediction/target vectors.")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all() or (true < 0).any():
        raise ValueError("RUL values must be finite and true RUL must be nonnegative.")
    return true, predicted


def nasa_score(true: np.ndarray, predicted: np.ndarray) -> float:
    """Sum the specified asymmetric penalty; overestimation costs more.

    d = predicted - true. The denominators 13 (early) and 10 (late) are the
    NASA metric's definition, not model hyperparameters. Lower scores are better.
    """
    true, predicted = _vectors(true, predicted)
    difference = predicted - true
    exponent = np.where(difference < 0, -difference / 13., difference / 10.)
    try:
        with np.errstate(over="raise"):
            return float(np.expm1(exponent).sum())
    except FloatingPointError as error:
        raise ValueError("NASA score overflow: prediction errors are too large.") from error


def regression_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate RMSE/MAE in cycles and the total NASA score."""
    true, predicted = _vectors(true, predicted)
    error = predicted - true
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": nasa_score(true, predicted),
    }


def last_cycle_evaluation(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Take one last observed row per test engine and join uncapped official RUL."""
    if predictions.empty or predictions.duplicated(["unit", "cycle"]).any():
        raise ValueError("Predictions must be nonempty with unique (unit, cycle) rows.")
    if labels["unit"].duplicated().any():
        raise ValueError("Expected exactly one true RUL label per engine.")
    if set(predictions["unit"]) != set(labels["unit"]):
        raise ValueError("Prediction engines and test-label engines do not match.")
    last = predictions.sort_values(["unit", "cycle"]).groupby("unit", sort=True).tail(1)
    return last.merge(labels, on="unit", validate="one_to_one").sort_values("unit").reset_index(drop=True)
