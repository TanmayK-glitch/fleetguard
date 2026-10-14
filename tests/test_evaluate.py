"""Test the NASA formula, error validation and last-cycle engine alignment."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import nasa_score, regression_metrics, last_cycle_evaluation


def test_nasa_score_matches_hand_calculated_branches() -> None:
    true = np.array([20., 20., 20.])
    predicted = np.array([7., 30., 20.])  # errors -13, +10 and 0
    assert nasa_score(true, predicted) == pytest.approx(2. * np.expm1(1.))


def test_nasa_penalizes_late_predictions_more() -> None:
    early = nasa_score(np.array([30.]), np.array([20.]))
    late = nasa_score(np.array([30.]), np.array([40.]))
    assert late > early > 0.


def test_perfect_predictions_have_zero_error() -> None:
    assert regression_metrics(np.array([0., 20.]), np.array([0., 20.])) == {
        "rmse": 0., "mae": 0., "nasa_score": 0.,
    }


def test_rmse_and_mae_match_hand_calculation() -> None:
    metrics = regression_metrics(np.array([10., 20.]), np.array([13., 16.]))
    assert metrics["rmse"] == pytest.approx(np.sqrt(12.5))
    assert metrics["mae"] == 3.5


@pytest.mark.parametrize("true,predicted", [
    ([], []), ([1., 2.], [1.]), ([[1.]], [[1.]]),
    ([1.], [np.nan]), ([np.inf], [1.]), ([-1.], [1.]),
])
def test_bad_metric_inputs_are_rejected(true, predicted) -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.array(true), np.array(predicted))


def test_nasa_overflow_is_explicit() -> None:
    with pytest.raises(ValueError, match="overflow"):
        nasa_score(np.array([0.]), np.array([10000.]))


def test_last_cycles_are_selected_and_labels_are_joined_by_unit() -> None:
    predictions = pd.DataFrame({
        "unit": [2, 1, 2, 1], "cycle": [2, 1, 1, 3], "predicted_rul": [80., 90., 95., 110.],
    })
    labels = pd.DataFrame({"unit": [2, 1], "true_rul": [70, 180]})
    result = last_cycle_evaluation(predictions, labels)
    assert result.unit.tolist() == [1, 2]
    assert result.cycle.tolist() == [3, 2]
    assert result.true_rul.tolist() == [180, 70]  # truth above cap remains unchanged
    assert regression_metrics(result.true_rul.to_numpy(), result.predicted_rul.to_numpy())["mae"] == 40.


def test_missing_test_engines_are_rejected() -> None:
    predictions = pd.DataFrame({"unit": [1], "cycle": [1], "predicted_rul": [10.]})
    labels = pd.DataFrame({"unit": [1, 2], "true_rul": [10, 20]})
    with pytest.raises(ValueError, match="do not match"):
        last_cycle_evaluation(predictions, labels)
