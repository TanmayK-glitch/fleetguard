"""Verify training-only state, per-engine isolation and causal gap handling."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import SensorPreprocessor
from src.data.load import SENSOR_COLUMNS


def test_sensor_selection_and_scaling_use_train_only(raw_train, config) -> None:
    processor = SensorPreprocessor(config["cleaning"]).fit(raw_train)
    assert processor.sensor_columns == ["sensor_1"]
    train = processor.transform(raw_train)
    assert train["sensor_1"].mean() == pytest.approx(0., abs=1e-12)
    assert train["sensor_1"].std(ddof=0) == pytest.approx(1.)
    saved = processor.state()
    test = raw_train.copy()
    test["sensor_1"] += 10000.
    test["sensor_2"] = np.arange(len(test)) * 1000.
    result = processor.transform(test)
    assert processor.state() == saved
    assert "sensor_2" not in result
    assert result["sensor_1"].mean() > 100.


@pytest.mark.parametrize("method", ["ewma", "rolling_median"])
def test_missing_readings_and_smoothing_do_not_cross_engines(raw_train, config, method) -> None:
    settings = {**config["cleaning"], "smoothing": method, "ewma_alpha": 1., "median_window": 1}
    processor = SensorPreprocessor(settings).fit(raw_train)
    test = raw_train.copy()
    test.loc[1, "sensor_1"] = np.nan
    test.loc[6, "sensor_1"] = np.nan  # first row of the second engine
    cleaned = processor.transform(test)
    restored = cleaned["sensor_1"] * processor.scales["sensor_1"] + processor.means["sensor_1"]
    assert restored.iloc[1] == pytest.approx(2.)
    assert restored.iloc[6] == pytest.approx(processor.medians["sensor_1"])
    assert np.isfinite(cleaned.to_numpy()).all()


def test_ewma_resets_at_each_engine(raw_train, config) -> None:
    processor = SensorPreprocessor(config["cleaning"]).fit(raw_train)
    cleaned = processor.transform(raw_train)
    restored = cleaned["sensor_1"] * processor.scales["sensor_1"] + processor.means["sensor_1"]
    assert restored.iloc[0] == pytest.approx(2.)
    assert restored.iloc[1] == pytest.approx(2.6)
    assert restored.iloc[6] == pytest.approx(20.)


def test_saved_state_reproduces_transform(tmp_path, raw_train, config) -> None:
    processor = SensorPreprocessor(config["cleaning"]).fit(raw_train)
    path = tmp_path / "preprocessor.json"
    processor.save(path)
    restored = SensorPreprocessor.load(path)
    assert restored.state() == processor.state()
    pd.testing.assert_frame_equal(restored.transform(raw_train), processor.transform(raw_train))


def test_all_missing_and_all_constant_sensors_fail(raw_train, config) -> None:
    raw_train[SENSOR_COLUMNS] = np.nan
    with pytest.raises(ValueError, match="All training sensors are missing"):
        SensorPreprocessor(config["cleaning"]).fit(raw_train)
    raw_train[SENSOR_COLUMNS] = 5.
    with pytest.raises(ValueError, match="constant/near-constant"):
        SensorPreprocessor(config["cleaning"]).fit(raw_train)


def test_unfitted_transform_fails(raw_train, config) -> None:
    with pytest.raises(RuntimeError, match="Call fit"):
        SensorPreprocessor(config["cleaning"]).transform(raw_train)
