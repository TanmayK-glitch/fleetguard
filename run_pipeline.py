"""One-command entry point for the implemented data and RUL training stages."""

import argparse
from pathlib import Path

from src.config import load_config
from src.data.pipeline import prepare_data


def main() -> None:
    """Parse options and print real data/model results for the selected stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--stage", choices=["data", "train", "agent", "all"], default="data")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    if args.stage not in {"data", "train"}:
        parser.error(f"Stage '{args.stage}' is not implemented in Phase 2; use --stage data or train.")
    root = Path(__file__).resolve().parent
    try:
        config = load_config(root / args.config)
        if args.stage == "data":
            report = prepare_data(config, args.subset, root)
        else:
            from src.models.train import train_models
            report = train_models(config, args.subset, root)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(status=1, message=f"{args.stage.capitalize()} stage failed: {error}\n")
    if args.stage == "train":
        print("Model          CV RMSE   Test RMSE   Test MAE   Test NASA")
        for name, experiment in report["experiments"].items():
            cv, test = experiment["validation"], experiment["test"]
            print(f"{name:<14} {cv['rmse']:8.3f} {test['rmse']:11.3f} {test['mae']:10.3f} {test['nasa_score']:11.3f}")
        print(f"Selected by CV {report['selection_metric']}: {report['selected_model']}")
        print(f"Reports: {report['report_dir']}")
        print(f"Models: {report['model_dir']}")
        return
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
    print("To train/evaluate RUL models, run --stage train.")


if __name__ == "__main__":
    main()
