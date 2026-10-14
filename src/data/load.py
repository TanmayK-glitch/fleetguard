"""Read the official C-MAPSS text format and check engine trajectories."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [f"sensor_{number}" for number in range(1, 22)]
SETTING_COLUMNS = [f"setting_{number}" for number in range(1, 4)]
COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass
class Dataset:
    """Keep raw train, test and the last-cycle test labels separate."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame


def _read_numeric(path: Path, column_count: int) -> pd.DataFrame:
    """Read whitespace-delimited numbers with a helpful missing-data error."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing C-MAPSS file: {path}. Put the official files in the raw "
            "directory; see scripts/download_data.md. No data is generated."
        )
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] != column_count:
            raise ValueError(f"Expected {column_count} columns, got {frame.shape[1]}.")
        return frame.apply(pd.to_numeric, errors="raise")
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Invalid C-MAPSS file {path}: {error}") from error


def validate_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    """Check schema, IDs, ordering, duplicates and consecutive observed cycles.

    Missing sensor values are permitted for cleaning. Settings and identifiers
    must be finite; infinities are never valid sensor measurements.
    """
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("Expected a nonempty table with the 26 C-MAPSS columns.")
    result = frame.copy()
    identifiers = result[["unit", "cycle"]].to_numpy(dtype=float)
    if not np.isfinite(identifiers).all() or (identifiers < 1).any():
        raise ValueError("unit and cycle must be finite positive integers.")
    if not np.equal(identifiers, np.floor(identifiers)).all():
        raise ValueError("unit and cycle must be integers.")
    if not np.isfinite(result[SETTING_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Operating settings must be finite.")
    if np.isinf(result[SENSOR_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Sensors contain infinity; use NaN for missing readings.")
    result[["unit", "cycle"]] = result[["unit", "cycle"]].astype(int)
    if result.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate (unit, cycle) observations.")
    result = result.sort_values(["unit", "cycle"]).reset_index(drop=True)
    for unit, group in result.groupby("unit", sort=True):
        expected = np.arange(1, len(group) + 1)
        if not np.array_equal(group["cycle"].to_numpy(), expected):
            raise ValueError(f"Engine {unit}: cycles must start at 1 and be consecutive.")
    return result


def load_subset(raw_dir: Path, subset: str) -> Dataset:
    """Read train/test trajectories and align last-cycle RUL labels by engine."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset}; choose from {SUBSETS}.")
    frames = []
    for split in ("train", "test"):
        frame = _read_numeric(raw_dir / f"{split}_{subset}.txt", len(COLUMNS))
        frame.columns = COLUMNS
        frames.append(validate_trajectories(frame))
    train, test = frames
    labels = _read_numeric(raw_dir / f"RUL_{subset}.txt", 1).iloc[:, 0]
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Test RUL labels must be finite and nonnegative.")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Test RUL labels must be integer cycle counts.")
    units = sorted(test["unit"].unique())
    if units != list(range(1, len(units) + 1)) or len(labels) != len(units):
        raise ValueError("Test engine IDs must be 1..N with exactly one RUL label each.")
    test_rul = pd.DataFrame({"unit": units, "true_rul": labels.to_numpy(dtype=int)})
    return Dataset(train, test, test_rul)
