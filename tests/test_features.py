"""Prove prefix invariance and check rolling statistics by hand."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import SensorPreprocessor
from src.data.features import build_features, capped_rul


def test_rolling_statistics_and_baseline_are_correct() -> None:
    cleaned = pd.DataFrame({"unit": [1]*4, "cycle": [1, 2, 3, 4], "sensor_1": [2., 4., 6., 100.]})
    result = build_features(cleaned, ["sensor_1"], [3], baseline_cycles=3)
    np.testing.assert_allclose(result["sensor_1_mean_3"], [2., 3., 4., 110./3])
    np.testing.assert_allclose(result["sensor_1_std_3"].iloc[:3], [0., 1., np.sqrt(8./3)])
    np.testing.assert_allclose(result["sensor_1_slope_3"], [0., 2., 2., 48.])
    np.testing.assert_allclose(result["sensor_1_baseline_delta"], [0., 1., 2., 96.])


@pytest.mark.parametrize("method", ["ewma", "rolling_median"])
def test_full_history_equals_every_available_prefix(raw_train, config, method) -> None:
    processor = SensorPreprocessor({**config["cleaning"], "smoothing": method}).fit(raw_train)
    history = raw_train[raw_train["unit"] == 1].copy()
    history.loc[[0, 2, 3], "sensor_1"] = np.nan

    def features(frame):
        return build_features(processor.transform(frame), processor.sensor_columns, [2, 3, 5], 3)

    full = features(history)
    for prefix_length in range(1, len(history) + 1):
        pd.testing.assert_frame_equal(features(history.iloc[:prefix_length]), full.iloc[:prefix_length])


@pytest.mark.parametrize("method", ["ewma", "rolling_median"])
def test_changed_future_cannot_change_past_features(raw_train, config, method) -> None:
    processor = SensorPreprocessor({**config["cleaning"], "smoothing": method}).fit(raw_train)
    changed = raw_train.copy()
    changed.loc[changed["cycle"] > 2, "sensor_1"] = 100000.
    original = build_features(processor.transform(raw_train), processor.sensor_columns, [2, 5], 5)
    modified = build_features(processor.transform(changed), processor.sensor_columns, [2, 5], 5)
    pd.testing.assert_frame_equal(original[original.cycle <= 2], modified[modified.cycle <= 2])


def test_rolling_features_reset_for_new_engines(raw_train) -> None:
    result = build_features(raw_train[["unit", "cycle", "sensor_1"]], ["sensor_1"], [5], 3)
    first = result[result.unit == 2].iloc[0]
    assert first.sensor_1_mean_5 == 20.
    assert first.sensor_1_std_5 == 0.
    assert first.sensor_1_slope_5 == 0.
    assert first.sensor_1_baseline_delta == 0.


def test_rul_is_capped_and_zero_at_failure(raw_train) -> None:
    target = capped_rul(raw_train, cap=3)
    assert target.tolist() == [3, 3, 3, 2, 1, 0, 3, 2, 1, 0]


def test_unsorted_input_is_sorted(raw_train) -> None:
    columns = ["unit", "cycle", "sensor_1"]
    expected = build_features(raw_train[columns], ["sensor_1"], [3], 3)
    actual = build_features(raw_train[columns].iloc[::-1], ["sensor_1"], [3], 3)
    pd.testing.assert_frame_equal(actual, expected)
