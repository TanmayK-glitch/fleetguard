"""Generate presentation-ready slide assets from existing Phase 2 artifacts.

Usage:
    python scripts/generate_slide_assets.py

Reads:
    artifacts/reports/FD001/metrics.json
    artifacts/reports/FD001/test_last_cycle_predictions.csv
    artifacts/reports/FD001/oof_predictions.csv
    artifacts/reports/FD001/fold_metrics.csv
    artifacts/models/FD001/preprocessor.json
    artifacts/models/FD001/random_forest.joblib

Writes to artifacts/slide_assets/:
    slide_facts.md
    pred_vs_true_test.png
    engine_rul_curve.png
    error_hist.png
    feature_importance_top10.png
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

# Ensure `src` is importable so joblib can unpickle RULPredictor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "artifacts" / "reports" / "FD001"
MODEL_DIR = ROOT / "artifacts" / "models" / "FD001"
OUTPUT_DIR = ROOT / "artifacts" / "slide_assets"

FIG_W, FIG_H = 16.0, 9.0       # 1600x900 at 100 dpi
FIG_DPI = 100
FONT_SIZE = 14
TITLE_SIZE = 18


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    """Apply consistent styling for all plots."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": FIG_DPI,
    })


def plot_pred_vs_true(test_last: pd.DataFrame, output: Path) -> None:
    """Scatter of predicted vs true RUL for all 100 test engines (RF only)."""
    rf = test_last[test_last["model"] == "random_forest"].copy()
    true = rf["true_rul"].to_numpy()
    pred = rf["predicted_rul"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    lo, hi = 0, max(true.max(), pred.max()) * 1.05

    # y=x reference line
    ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=1.5,
            linestyle="--", label="y = x (perfect)", zorder=1)

    # scatter
    ax.scatter(true, pred, s=90, alpha=0.8, edgecolors="white",
               linewidths=0.6, color="#2563EB", zorder=2)

    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Random Forest: Predicted vs True RUL  —  100 FD001 Test Engines")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [OK] {output.name}")


def plot_engine_rul_curve(oof: pd.DataFrame, metrics_json: dict, output: Path) -> None:
    """True (capped) RUL vs out-of-fold predicted RUL for one train engine."""
    rf_oof = oof[oof["model"] == "random_forest"].copy()
    rul_cap = metrics_json["config"]["target"]["rul_cap"]

    # Pick a mid-length engine for a visually informative curve.
    engine_lengths = rf_oof.groupby("unit").size()
    median_len = engine_lengths.median()
    best_unit = (engine_lengths - median_len).abs().idxmin()

    eng = rf_oof[rf_oof["unit"] == best_unit].sort_values("cycle")
    cycles = eng["cycle"].to_numpy()
    true_rul = eng["true_rul"].to_numpy()
    pred_rul = eng["predicted_rul"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(cycles, true_rul, color="#059669", linewidth=2.5, label=f"True RUL (capped at {rul_cap})")
    ax.plot(cycles, pred_rul, color="#DC2626", linewidth=2.0, linestyle="--",
            label="RF out-of-fold predicted RUL")
    ax.fill_between(cycles, true_rul, pred_rul, alpha=0.12, color="#6366F1")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine {best_unit}: True vs Predicted RUL Over Full Lifetime  ({len(cycles)} cycles)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [OK] {output.name}  (engine {best_unit}, {len(cycles)} cycles)")


def plot_error_hist(test_last: pd.DataFrame, output: Path) -> None:
    """Histogram of test error (predicted − true) for RF on all 100 test engines."""
    rf = test_last[test_last["model"] == "random_forest"].copy()
    error = rf["predicted_rul"].to_numpy() - rf["true_rul"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bins = np.arange(error.min() - 2, error.max() + 3, 3)
    ax.hist(error, bins=bins, color="#7C3AED", edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.axvline(0, color="#DC2626", linewidth=2, linestyle="--", label="Zero error")
    mean_err = error.mean()
    ax.axvline(mean_err, color="#F59E0B", linewidth=2, linestyle="-.", label=f"Mean error = {mean_err:.1f}")

    ax.set_xlabel("Prediction Error  (predicted − true RUL, cycles)")
    ax.set_ylabel("Number of Engines")
    ax.set_title("Test Error Distribution  —  Random Forest on 100 FD001 Test Engines")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [OK] {output.name}")


def plot_feature_importance(model_path: Path, feature_columns: list[str], output: Path) -> None:
    """Top-10 features by RF impurity importance."""
    bundle = joblib.load(model_path)
    # bundle is a RULPredictor dataclass; model is the sklearn RF
    rf_model = bundle.model
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]  # top 10 descending

    names = [feature_columns[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, values, color="#2563EB", edgecolor="white", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Top 10 Feature Importances  —  Random Forest (FD001)")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate bars with values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    fig.savefig(output, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [OK] {output.name}")


def generate_slide_facts(metrics: dict, preprocessor: dict, fold_metrics: pd.DataFrame, output: Path) -> None:
    """Write slide_facts.md from real artifact data."""
    cfg = metrics["config"]
    exps = metrics["experiments"]

    kept = preprocessor["sensor_columns"]
    dropped = preprocessor["dropped_sensors"]
    windows = cfg["features"]["windows"]
    rul_cap = cfg["target"]["rul_cap"]
    scaling = f"z-score (population mean & std, EWMA α={cfg['cleaning']['ewma_alpha']} smoothing)"

    # RF hyperparams
    rf_params = exps["random_forest"]["params"]
    rf_hyp = ", ".join(f"{k}={v}" for k, v in rf_params.items())

    # Per-fold RF timing
    rf_folds = fold_metrics[fold_metrics["model"] == "random_forest"]
    rf_cv_time = rf_folds["fit_predict_seconds"].sum()
    rf_final_time = exps["random_forest"]["final_fit_predict_seconds"]
    total_time = metrics["elapsed_seconds"]

    lines = [
        "# FD001 Slide Facts",
        "",
        "## Sensors",
        "",
        f"**Kept ({len(kept)}):** {', '.join(kept)}",
        "",
        f"**Dropped ({len(dropped)}):** {', '.join(dropped)}",
        "",
        f"**Drop criterion:** population std ≤ {cfg['cleaning']['min_sensor_std']} on EWMA-smoothed train data",
        "",
        "## Feature Engineering",
        "",
        f"- Rolling window sizes: **{windows}**",
        f"- Per window: rolling mean, std, slope (causal only)",
        f"- Baseline delta: deviation from first {cfg['features']['baseline_cycles']} cycles",
        f"- Total feature columns: **{len(metrics['feature_columns'])}**",
        f"- RUL cap: **{rul_cap}** cycles",
        f"- Scaling: {scaling}",
        "",
        "## Random Forest Hyperparameters",
        "",
        f"  {rf_hyp}",
        "",
        "## Cross-Validation Metrics (pooled out-of-fold, capped true RUL)",
        "",
        "| Model | CV RMSE | CV MAE | CV NASA Score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("ridge", "random_forest", "lightgbm"):
        cv = exps[name]["validation"]
        lines.append(
            f"| {name} | {cv['rmse']:.3f} | {cv['mae']:.3f} | {cv['nasa_score']:.1f} |"
        )
    lines += [
        "",
        "## Test Metrics (last cycle per engine, **raw / uncapped** true RUL)",
        "",
        "| Model | Test RMSE | Test MAE | Test NASA Score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("ridge", "random_forest", "lightgbm"):
        t = exps[name]["test"]
        lines.append(
            f"| {name} | {t['rmse']:.3f} | {t['mae']:.3f} | {t['nasa_score']:.1f} |"
        )
    lines += [
        "",
        "## Key Details",
        "",
        f"- **Test true RUL used for scoring:** raw / uncapped (from `RUL_FD001.txt`)",
        f"- **Final model retrained on all train engines?** Yes — after CV selection, "
        f"all 3 models are refit on 100% of the {metrics['train_engines']} train engines",
        f"- **Selected model:** `{metrics['selected_model']}` by pooled CV {metrics['selection_metric']}",
        f"- **CV scheme:** GroupKFold, {metrics['cv_splits']} splits, seed {metrics['seed']}",
        f"- **Training rows:** {metrics['train_rows']} (train), {metrics['test_engines']} test engines",
        f"- **Test count:** {metrics['test_engines']} engines (one last-cycle prediction each)",
        f"- **RF CV time:** {rf_cv_time:.1f}s  |  RF final fit: {rf_final_time:.1f}s  |  Total pipeline: {total_time:.1f}s",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {output.name}")


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load saved artifacts
    metrics = _load_json(REPORT_DIR / "metrics.json")
    preprocessor = _load_json(MODEL_DIR / "preprocessor.json")
    test_last = pd.read_csv(REPORT_DIR / "test_last_cycle_predictions.csv")
    oof = pd.read_csv(REPORT_DIR / "oof_predictions.csv")
    fold_metrics = pd.read_csv(REPORT_DIR / "fold_metrics.csv")

    print("Generating slide assets from saved Phase 2 artifacts...\n")

    plot_pred_vs_true(test_last, OUTPUT_DIR / "pred_vs_true_test.png")
    plot_engine_rul_curve(oof, metrics, OUTPUT_DIR / "engine_rul_curve.png")
    plot_error_hist(test_last, OUTPUT_DIR / "error_hist.png")
    plot_feature_importance(
        MODEL_DIR / "random_forest.joblib",
        metrics["feature_columns"],
        OUTPUT_DIR / "feature_importance_top10.png",
    )
    generate_slide_facts(metrics, preprocessor, fold_metrics, OUTPUT_DIR / "slide_facts.md")

    print(f"\nDone. All assets written to {OUTPUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
