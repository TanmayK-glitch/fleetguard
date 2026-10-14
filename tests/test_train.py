"""Run a tiny full comparison and check serialized inference and reports."""

import json

import joblib
import pandas as pd

from src.models.train import train_models
from src.models.evaluate import regression_metrics


def test_training_artifacts_and_loaded_model_predictions(tmp_path, raw_train, config) -> None:
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for split in ("train", "test"):
        raw_train.to_csv(raw / f"{split}_FD001.txt", sep=" ", index=False, header=False)
    (raw / "RUL_FD001.txt").write_text("12\n180\n")
    config["training"]["cv_splits"] = 2
    config["training"]["random_forest"].update(n_estimators=3, max_depth=2, min_samples_leaf=1)
    config["training"]["lightgbm"].update(n_estimators=3, num_leaves=2, min_child_samples=1)
    config["features"]["windows"] = [2]
    report = train_models(config, "FD001", tmp_path, progress=None)
    assert set(report["experiments"]) == {"ridge", "random_forest", "lightgbm"}
    chosen = min(report["experiments"], key=lambda name: report["experiments"][name]["validation"]["rmse"])
    assert report["selected_model"] == chosen
    output = tmp_path / "artifacts/reports/FD001"
    assert json.loads((output / "metrics.json").read_text())["selected_model"] == chosen
    assert (output / "metrics.md").is_file()
    saved_predictions = pd.read_csv(output / "test_predictions.csv")
    last = pd.read_csv(output / "test_last_cycle_predictions.csv")
    assert len(last) == 6  # two engines for each of three models
    for name in report["experiments"]:
        bundle = joblib.load(tmp_path / f"artifacts/models/FD001/{name}.joblib")
        assert not {"unit", "cycle", "rul", "true_rul"}.intersection(bundle.feature_columns)
        expected = saved_predictions[saved_predictions.model == name].drop(columns="model").reset_index(drop=True)
        pd.testing.assert_frame_equal(bundle.predict(raw_train), expected, check_exact=False, atol=1e-10)
        comparison = last[last.model == name]
        assert comparison.true_rul.tolist() == [12, 180]
        metrics = regression_metrics(comparison.true_rul.to_numpy(), comparison.predicted_rul.to_numpy())
        for key, value in metrics.items():
            assert abs(value-report["experiments"][name]["test"][key]) < 1e-10
    audits = json.loads((output / "fold_audit.json").read_text())
    assert all(set(a["train_units"]).isdisjoint(a["validation_units"]) for a in audits)
