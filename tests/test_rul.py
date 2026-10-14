"""Engine separation, train-only fold state and numerical slope equivalence."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import validate_training_config
from src.data.features import build_features
from src.models.rul import engine_folds, prepare_fold, bound_predictions, make_model
from src.models.train import cross_validate, select_model


def test_group_folds_have_no_engine_overlap_and_cover_every_row(raw_train, config) -> None:
    config["training"]["cv_splits"] = 2
    validation_rows = []
    for train_idx, validation_idx in engine_folds(raw_train, config):
        train_units = set(raw_train.iloc[train_idx].unit)
        validation_units = set(raw_train.iloc[validation_idx].unit)
        assert train_units.isdisjoint(validation_units)
        validation_rows.extend(validation_idx.tolist())
    assert sorted(validation_rows) == list(range(len(raw_train)))
    first = list(engine_folds(raw_train, config))
    repeated = list(engine_folds(raw_train, config))
    for left, right in zip(first, repeated):
        np.testing.assert_array_equal(left[1], right[1])


def test_fold_preprocessor_cannot_learn_from_validation(raw_train, config) -> None:
    train = raw_train[raw_train.unit == 1]
    validation = raw_train[raw_train.unit == 2].copy()
    original = prepare_fold(train, validation, config)
    validation["sensor_1"] += 10000.
    validation["sensor_2"] = np.arange(len(validation)) * 100.
    changed = prepare_fold(train, validation, config)
    assert original.processor.state() == changed.processor.state()
    pd.testing.assert_frame_equal(original.train, changed.train)
    assert "sensor_2" not in changed.validation
    assert original.processor.medians["sensor_1"] == pytest.approx(7.)


def test_prepare_fold_rejects_overlapping_engines(raw_train, config) -> None:
    with pytest.raises(ValueError, match="Engine leakage"):
        prepare_fold(raw_train, raw_train, config)


def test_fold_count_cannot_exceed_engine_count(raw_train, config) -> None:
    with pytest.raises(ValueError, match="number of training engines"):
        list(engine_folds(raw_train, config))


def test_oof_predictions_cover_every_row_and_metrics_are_reproducible(raw_train, config) -> None:
    config["training"].update(cv_splits=2, models=["ridge"])
    pooled, folds, oof, audits = cross_validate(raw_train, config)
    repeated, _, _, _ = cross_validate(raw_train, config)
    assert pooled == repeated
    assert len(oof) == len(raw_train) and not oof.duplicated(["unit", "cycle"]).any()
    assert len(folds) == 2
    for audit in audits:
        assert set(audit["train_units"]).isdisjoint(audit["validation_units"])


def test_model_selection_uses_validation_metric() -> None:
    validation = {"ridge": {"rmse": 30.}, "random_forest": {"rmse": 20.}, "lightgbm": {"rmse": 15.}}
    assert select_model(validation, "rmse") == "lightgbm"


def test_prediction_clipping_is_configurable_and_does_not_change_truth(config) -> None:
    predictions = np.array([-10., 30., 200.])
    np.testing.assert_array_equal(bound_predictions(predictions, config), [0., 30., 125.])
    config["training"]["clip_predictions"] = False
    np.testing.assert_array_equal(bound_predictions(predictions, config), predictions)


@pytest.mark.parametrize("model", ["random_forest", "lightgbm"])
def test_tree_models_use_fixed_seed_and_cpu_limit(model, config) -> None:
    params = make_model(model, config).get_params()
    assert params["random_state"] == config["seed"]
    assert params["n_jobs"] == config["training"]["n_jobs"]


def test_vectorized_slopes_match_least_squares_reference(config) -> None:
    readings = np.random.default_rng(config["seed"]).normal(size=40)
    frame = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 41), "sensor_1": readings})
    result = build_features(frame, ["sensor_1"], [5, 10, 30], 10)
    for window in [5, 10, 30]:
        expected = []
        for end in range(1, len(readings) + 1):
            values = readings[max(0, end-window):end]
            x = np.arange(len(values), dtype=float)
            x -= x.mean()
            expected.append(0. if len(values) == 1 else np.dot(x, values-values.mean()) / np.dot(x, x))
        np.testing.assert_allclose(result[f"sensor_1_slope_{window}"], expected, atol=1e-12)


@pytest.mark.parametrize("setting,value", [
    ("cv_splits", 1), ("n_jobs", 0), ("models", ["unknown"]),
    ("models", ["ridge", "ridge"]), ("clip_predictions", "true"),
    ("selection_metric", "accuracy"),
])
def test_bad_training_settings_fail(config, setting, value) -> None:
    altered = copy.deepcopy(config)
    altered["training"][setting] = value
    with pytest.raises(ValueError):
        validate_training_config(altered)
