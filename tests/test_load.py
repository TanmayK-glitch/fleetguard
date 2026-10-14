"""Guard the raw-data contract and the last-cycle test-label alignment."""

import numpy as np
import pytest

from src.data.load import load_subset, validate_trajectories


def test_missing_data_has_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="scripts/download_data.md"):
        load_subset(tmp_path, "FD001")


def test_loader_aligns_labels_and_sorts_engines(tmp_path, raw_train) -> None:
    raw_train.iloc[::-1].to_csv(tmp_path / "train_FD001.txt", sep=" ", header=False, index=False)
    raw_train.to_csv(tmp_path / "test_FD001.txt", sep=" ", header=False, index=False)
    (tmp_path / "RUL_FD001.txt").write_text("12\n35\n")
    dataset = load_subset(tmp_path, "FD001")
    assert dataset.train[["unit", "cycle"]].iloc[0].tolist() == [1, 1]
    assert dataset.test_rul.to_dict("list") == {"unit": [1, 2], "true_rul": [12, 35]}


def test_loader_rejects_wrong_column_count(tmp_path) -> None:
    (tmp_path / "train_FD001.txt").write_text("1 1 0\n")
    with pytest.raises(ValueError, match="26 columns"):
        load_subset(tmp_path, "FD001")


def test_loader_rejects_wrong_label_count(tmp_path, raw_train) -> None:
    for split in ("train", "test"):
        raw_train.to_csv(tmp_path / f"{split}_FD001.txt", sep=" ", header=False, index=False)
    (tmp_path / "RUL_FD001.txt").write_text("12\n")
    with pytest.raises(ValueError, match="exactly one RUL label"):
        load_subset(tmp_path, "FD001")


@pytest.mark.parametrize("column,value", [("unit", 1.5), ("cycle", 0), ("unit", np.nan)])
def test_bad_identifiers_are_rejected(raw_train, column, value) -> None:
    raw_train[column] = raw_train[column].astype(float)
    raw_train.loc[0, column] = value
    with pytest.raises(ValueError, match="unit and cycle"):
        validate_trajectories(raw_train)


def test_duplicate_cycles_are_rejected(raw_train) -> None:
    raw_train.loc[1, "cycle"] = 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_trajectories(raw_train)


def test_missing_cycles_are_rejected(raw_train) -> None:
    with pytest.raises(ValueError, match="consecutive"):
        validate_trajectories(raw_train.drop(index=1))


def test_missing_sensors_are_allowed_but_infinities_are_not(raw_train) -> None:
    raw_train.loc[0, "sensor_1"] = np.nan
    validate_trajectories(raw_train)
    raw_train.loc[0, "sensor_1"] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        validate_trajectories(raw_train)
