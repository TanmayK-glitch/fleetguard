"""Orchestrate the Phase 1 data stage and write reproducible artifacts."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.clean import SensorPreprocessor
from src.data.features import build_features, capped_rul
from src.data.load import load_subset


def prepare_data(config: dict[str, Any], subset: str, root: Path) -> dict[str, Any]:
    """Fit on FD001 train and write train/test features, labels and state."""
    if subset != "FD001":
        raise ValueError("Phase 1 supports FD001 only; other subsets are deferred to Phase 6.")
    np.random.seed(config["seed"])
    dataset = load_subset(root / config["paths"]["raw"], subset)
    processor = SensorPreprocessor(config["cleaning"]).fit(dataset.train)
    tables = {}
    nonfinite_counts = {}
    for split, raw in (("train", dataset.train), ("test", dataset.test)):
        cleaned = processor.transform(raw)
        features = build_features(
            cleaned, processor.sensor_columns,
            config["features"]["windows"], config["features"]["baseline_cycles"],
        )
        if split == "train":
            features["rul"] = capped_rul(raw, config["target"]["rul_cap"]).to_numpy()
        nonfinite_counts[split] = int((~np.isfinite(features.to_numpy(dtype=float))).sum())
        if nonfinite_counts[split]:
            raise ValueError(f"Nonfinite values in {split} features; no artifacts written.")
        tables[split] = features

    output_dir = root / config["paths"]["processed"] / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, features in tables.items():
        features.to_csv(output_dir / f"{split}_features.csv", index=False)
    dataset.test_rul.to_csv(output_dir / "test_rul.csv", index=False)
    processor.save(output_dir / "preprocessor.json")
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    feature_columns = [c for c in tables["test"].columns if c not in {"unit", "cycle"}]
    report = {
        "subset": subset,
        "train_engines": int(dataset.train["unit"].nunique()),
        "test_engines": int(dataset.test["unit"].nunique()),
        "train_rows": len(tables["train"]),
        "test_rows": len(tables["test"]),
        "retained_sensors": processor.sensor_columns,
        "dropped_sensors": processor.dropped_sensors,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "nonfinite_feature_values": sum(nonfinite_counts.values()),
        "raw_missing_sensor_values": {
            split: int(frame.filter(like="sensor_").isna().sum().sum())
            for split, frame in (("train", dataset.train), ("test", dataset.test))
        },
        "train_rul_min": int(tables["train"]["rul"].min()),
        "train_rul_max": int(tables["train"]["rul"].max()),
        "output_dir": str(output_dir),
        "validation_note": "Phase 2 must refit preprocessing inside each training fold.",
    }
    (output_dir / "data_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
