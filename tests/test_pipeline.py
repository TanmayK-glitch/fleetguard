"""Exercise artifacts and label separation using a tiny test fixture."""

import json

import pandas as pd
import pytest

from src.config import load_config
from src.data.pipeline import prepare_data


def test_pipeline_artifacts_and_reproducibility(tmp_path, raw_train, config) -> None:
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    for split in ("train", "test"):
        raw_train.to_csv(raw_dir / f"{split}_FD001.txt", sep=" ", index=False, header=False)
    labels = raw_dir / "RUL_FD001.txt"
    labels.write_text("12\n35\n")
    report = prepare_data(config, "FD001", tmp_path)
    output = tmp_path / "data/processed/FD001"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    repeated = prepare_data(config, "FD001", tmp_path)
    assert report == repeated
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    train = pd.read_csv(output / "train_features.csv")
    test = pd.read_csv(output / "test_features.csv")
    assert "rul" in train and "rul" not in test
    assert "true_rul" not in test
    assert report["feature_count"] == 11
    assert json.loads((output / "data_report.json").read_text())["nonfinite_feature_values"] == 0
    # Even changing the ground truth cannot alter test features or fitted state.
    labels.write_text("100\n200\n")
    prepare_data(config, "FD001", tmp_path)
    assert before["test_features.csv"] == (output / "test_features.csv").read_bytes()
    assert before["preprocessor.json"] == (output / "preprocessor.json").read_bytes()


def test_other_subsets_are_explicitly_deferred(tmp_path, config) -> None:
    with pytest.raises(ValueError, match="Phase 6"):
        prepare_data(config, "FD002", tmp_path)


@pytest.mark.parametrize("setting,value", [
    ("windows", [5, 0]), ("windows", [5, 5]), ("baseline_cycles", -1),
])
def test_config_rejects_invalid_feature_settings(tmp_path, config, setting, value) -> None:
    import yaml
    config["features"][setting] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
