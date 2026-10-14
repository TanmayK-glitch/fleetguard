"""One-command entry point; only the Phase 1 data stage is implemented yet."""

import argparse
from pathlib import Path

from src.config import load_config
from src.data.pipeline import prepare_data


def main() -> None:
    """Parse options and print real data-stage statistics, never model metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--stage", choices=["data", "train", "agent", "all"], default="data")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    if args.stage != "data":
        parser.error(f"Stage '{args.stage}' is not implemented in Phase 1; use --stage data.")
    root = Path(__file__).resolve().parent
    try:
        config = load_config(root / args.config)
        report = prepare_data(config, args.subset, root)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(status=1, message=f"Data stage failed: {error}\n")
    print(f"Prepared {report['subset']}")
    print(f"Train: {report['train_engines']} engines, {report['train_rows']:,} rows")
    print(f"Test:  {report['test_engines']} engines, {report['test_rows']:,} rows")
    print(f"Sensors: {len(report['retained_sensors'])} retained, {len(report['dropped_sensors'])} dropped")
    print(f"Dropped: {', '.join(report['dropped_sensors'])}")
    print(f"Features: {report['feature_count']}; nonfinite values: {report['nonfinite_feature_values']}")
    missing = report["raw_missing_sensor_values"]
    print(f"Raw missing sensor readings: train={missing['train']}, test={missing['test']}")
    print(f"Train RUL range: {report['train_rul_min']}..{report['train_rul_max']}")
    print(f"Artifacts: {report['output_dir']}")
    print("Model metrics: not available until Phase 2.")


if __name__ == "__main__":
    main()
