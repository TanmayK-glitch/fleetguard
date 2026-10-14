"""Training-only sensor selection and scaling with causal engine cleaning."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import SENSOR_COLUMNS, validate_trajectories


class SensorPreprocessor:
    """Fit sensor statistics on train, then reuse them without refitting.

    Each transform takes complete engine histories starting at cycle 1. Missing
    readings use the latest earlier reading; leading gaps use a train median.
    EWMA/median smoothing resets at each engine. Scaling uses population std.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self.sensor_columns: list[str] = []
        self.dropped_sensors: list[str] = []
        self.medians = pd.Series(dtype=float)
        self.means = pd.Series(dtype=float)
        self.scales = pd.Series(dtype=float)

    def _clean(self, frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
        """Fill and smooth each engine separately, looking only backwards."""
        columns = medians.index.tolist()
        result = frame[["unit", "cycle", *columns]].copy()
        result[columns] = result.groupby("unit")[columns].ffill().fillna(medians)
        grouped = result.groupby("unit")[columns]
        if self.settings["smoothing"] == "ewma":
            result[columns] = grouped.transform(
                lambda values: values.ewm(
                    alpha=self.settings["ewma_alpha"], adjust=False
                ).mean()
            )
        else:
            result[columns] = grouped.transform(
                lambda values: values.rolling(
                    self.settings["median_window"], min_periods=1
                ).median()
            )
        return result

    def fit(self, train: pd.DataFrame) -> "SensorPreprocessor":
        """Learn fallback medians, retained sensors and z-score statistics."""
        train = validate_trajectories(train)
        medians = train[SENSOR_COLUMNS].median().dropna()
        if medians.empty:
            raise ValueError("All training sensors are missing; nothing can be fitted.")
        cleaned = self._clean(train, medians)
        std = cleaned[medians.index].std(ddof=0)
        self.sensor_columns = std[std > self.settings["min_sensor_std"]].index.tolist()
        self.dropped_sensors = [c for c in SENSOR_COLUMNS if c not in self.sensor_columns]
        if not self.sensor_columns:
            raise ValueError("All training sensors are constant/near-constant; check the data/config.")
        self.medians = medians[self.sensor_columns]
        self.means = cleaned[self.sensor_columns].mean()
        self.scales = std[self.sensor_columns]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clean a history and scale it with the previously fitted train state."""
        if self.means.empty:
            raise RuntimeError("Call fit(train) or load(saved_state) before transform.")
        frame = validate_trajectories(frame)
        result = self._clean(frame, self.medians)
        result[self.sensor_columns] = (
            result[self.sensor_columns] - self.means
        ) / self.scales
        return result

    def state(self) -> dict[str, Any]:
        """Expose a small, human-readable JSON representation of fitted state."""
        if self.means.empty:
            raise RuntimeError("No fitted preprocessing state to save.")
        return {
            "schema_version": 1,
            "settings": self.settings,
            "sensor_columns": self.sensor_columns,
            "dropped_sensors": self.dropped_sensors,
            "medians": self.medians.to_dict(),
            "means": self.means.to_dict(),
            "scales": self.scales.to_dict(),
        }

    def save(self, path: Path) -> None:
        """Save only JSON; no executable pickle file is needed for this phase."""
        path.write_text(json.dumps(self.state(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SensorPreprocessor":
        """Restore a trusted local state file without seeing test labels."""
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["schema_version"] != 1:
            raise ValueError("Unsupported preprocessing state version.")
        processor = cls(state["settings"])
        processor.sensor_columns = state["sensor_columns"]
        processor.dropped_sensors = state["dropped_sensors"]
        for attribute in ("medians", "means", "scales"):
            values = pd.Series(state[attribute], dtype=float)[processor.sensor_columns]
            setattr(processor, attribute, values)
        return processor
