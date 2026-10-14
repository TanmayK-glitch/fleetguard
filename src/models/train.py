"""Compare engine-safe baselines, refit models and log local Phase 2 artifacts."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import validate_training_config
from src.data.clean import SensorPreprocessor
from src.data.features import capped_rul
from src.data.load import load_subset, validate_trajectories
from src.models.evaluate import last_cycle_evaluation, regression_metrics
from src.models.rul import (
    METADATA_COLUMNS, RULPredictor, bound_predictions, engine_folds,
    make_model, prepare_fold, transform_features,
)


def cross_validate(
    raw: pd.DataFrame, config: dict[str, Any], progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate every model on the same folds; pool all out-of-fold predictions."""
    raw = validate_trajectories(raw)
    rows, predictions, audits = [], [], []
    for fold_number, (train_idx, validation_idx) in enumerate(engine_folds(raw, config), start=1):
        if progress:
            progress(f"Preparing fold {fold_number}/{config['training']['cv_splits']}...",)
        fold = prepare_fold(raw.iloc[train_idx], raw.iloc[validation_idx], config)
        audits.append({
            "fold": fold_number,
            "train_units": sorted(int(u) for u in fold.train["unit"].unique()),
            "validation_units": sorted(int(u) for u in fold.validation["unit"].unique()),
            "preprocessor": fold.processor.state(),
        })
        columns = [c for c in fold.train.columns if c not in METADATA_COLUMNS]
        for name in config["training"]["models"]:
            started = perf_counter()
            model = make_model(name, config)
            with threadpool_limits(limits=config["training"]["n_jobs"]):
                model.fit(fold.train[columns], fold.train_target)
                predicted = bound_predictions(model.predict(fold.validation[columns]), config)
            metrics = regression_metrics(fold.validation_target, predicted)
            rows.append({
                "model": name, "fold": fold_number, **metrics,
                "validation_rows": len(fold.validation), "fit_predict_seconds": perf_counter() - started,
            })
            table = fold.validation[METADATA_COLUMNS].copy()
            table["fold"] = fold_number
            table["model"] = name
            table["true_rul"] = fold.validation_target
            table["predicted_rul"] = predicted
            predictions.append(table)
            if progress:
                progress(f"  {name}: validation RMSE={metrics['rmse']:.3f}")
    oof = pd.concat(predictions, ignore_index=True)
    pooled = {}
    for name in config["training"]["models"]:
        values = oof[oof["model"] == name]
        if len(values) != len(raw) or values.duplicated(METADATA_COLUMNS).any():
            raise ValueError("Out-of-fold predictions must cover each training row exactly once.")
        pooled[name] = regression_metrics(values["true_rul"].to_numpy(), values["predicted_rul"].to_numpy())
    return pooled, pd.DataFrame(rows), oof, audits


def select_model(metrics: dict[str, dict[str, float]], metric: str) -> str:
    """Choose lowest validation error; alphabetical order breaks exact ties."""
    return min(metrics, key=lambda name: (metrics[name][metric], name))


def _write_json(path: Path, value: Any) -> None:
    """Write inspectable JSON, rejecting NaN/infinity in experiment results."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    """Generate a readable comparison from actual computed metrics."""
    lines = [
        f"# {report['subset']} Phase 2 RUL comparison", "",
        "CV uses pooled held-out predictions over all training cycles with capped targets.",
        "Test uses one last observed cycle per engine with uncapped official targets.",
        "NASA scores are totals and depend on sample count; lower is better for all metrics.", "",
        "| Model | CV RMSE | CV MAE | CV NASA | Test RMSE | Test MAE | Test NASA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, experiment in report["experiments"].items():
        cv, test = experiment["validation"], experiment["test"]
        lines.append(
            f"| {name} | {cv['rmse']:.3f} | {cv['mae']:.3f} | {cv['nasa_score']:.3f} | "
            f"{test['rmse']:.3f} | {test['mae']:.3f} | {test['nasa_score']:.3f} |"
        )
    lines.extend([
        "", f"Selected model: **{report['selected_model']}** by pooled CV {report['selection_metric']}.",
        "Test scores were calculated after this selection and were not used for tuning.",
        f"Training rows: {report['train_rows']}; test engines: {report['test_engines']}.",
        f"GroupKFold splits: {report['cv_splits']}; seed: {report['seed']}.",
        "", "Local artifacts include model bundles, configuration, fold membership/state,",
        "predictions and dependency/data fingerprints. MLflow registry integration is Phase 4.", "",
    ])
    return "\n".join(lines)


def train_models(
    config: dict[str, Any], subset: str, root: Path,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run CV from RAW engines, select by validation, then refit/evaluate all models."""
    if subset != "FD001":
        raise ValueError("Phase 2 supports FD001 only; other subsets are deferred to Phase 6.")
    validate_training_config(config)
    started = perf_counter()
    raw_dir = root / config["paths"]["raw"]
    dataset = load_subset(raw_dir, subset)
    validation, folds, oof, audits = cross_validate(dataset.train, config, progress)
    selected = select_model(validation, config["training"]["selection_metric"])
    if progress:
        progress(f"Selected {selected} by validation; refitting on all training engines...")
    # Only now fit the final preprocessing on all train engines. Never reuse
    # the Phase 1 feature CSV for CV or let test targets choose the winner.
    processor = SensorPreprocessor(config["cleaning"]).fit(dataset.train)
    train = transform_features(dataset.train, processor, config)
    test = transform_features(dataset.test, processor, config)
    columns = [c for c in train.columns if c not in METADATA_COLUMNS]
    target = capped_rul(dataset.train, config["target"]["rul_cap"]).to_numpy()
    model_dir = root / config["paths"]["models"] / subset
    report_dir = root / config["paths"]["reports"] / subset
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    experiments, all_test, last_test = {}, [], []
    for name in config["training"]["models"]:
        model_started = perf_counter()
        model = make_model(name, config)
        with threadpool_limits(limits=config["training"]["n_jobs"]):
            model.fit(train[columns], target)
            predicted = bound_predictions(model.predict(test[columns]), config)
        bundle = RULPredictor(name, model, processor, columns, config)
        joblib.dump(bundle, model_dir / f"{name}.joblib")
        predictions = test[METADATA_COLUMNS].copy()
        predictions["predicted_rul"] = predicted
        predictions["model"] = name
        last = last_cycle_evaluation(predictions, dataset.test_rul)
        all_test.append(predictions)
        last_test.append(last)
        experiments[name] = {
            "params": config["training"][name],
            "validation": validation[name],
            "test": regression_metrics(last["true_rul"].to_numpy(), last["predicted_rul"].to_numpy()),
            "final_fit_predict_seconds": perf_counter() - model_started,
        }
    report = {
        "schema_version": 1, "subset": subset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected, "selection_metric": config["training"]["selection_metric"],
        "cv_splits": config["training"]["cv_splits"], "seed": config["seed"],
        "validation_scope": "all_cycles_capped_targets_pooled_out_of_fold",
        "test_scope": "last_cycle_per_engine_uncapped_official_targets",
        "train_rows": len(train), "train_engines": int(dataset.train["unit"].nunique()),
        "test_engines": len(dataset.test_rul), "feature_columns": columns,
        "config": config, "experiments": experiments,
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib")
        }},
        "data_sha256": {name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() for name in (
            f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt",
        )},
        "elapsed_seconds": perf_counter() - started,
        "model_dir": str(model_dir), "report_dir": str(report_dir),
    }
    processor.save(model_dir / "preprocessor.json")
    _write_json(model_dir / "metadata.json", {
        "selected_model": selected, "feature_columns": columns, "config": config,
        "environment": report["environment"], "created_at": report["created_at"],
    })
    folds.to_csv(report_dir / "fold_metrics.csv", index=False)
    oof.to_csv(report_dir / "oof_predictions.csv", index=False)
    pd.concat(all_test, ignore_index=True).to_csv(report_dir / "test_predictions.csv", index=False)
    pd.concat(last_test, ignore_index=True).to_csv(report_dir / "test_last_cycle_predictions.csv", index=False)
    _write_json(report_dir / "fold_audit.json", audits)
    _write_json(report_dir / "metrics.json", report)
    (report_dir / "metrics.md").write_text(_markdown_report(report), encoding="utf-8")
    return report
