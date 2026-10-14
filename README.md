# Self-Healing Machines

A student-friendly, CPU-only predictive maintenance project using
[NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
The planned flow is sensors → cleaning/features → RUL and anomaly models →
plain Python triage rules → maintenance work orders.

**Implemented through Phase 2:** FD001 loading, causal preprocessing/features,
Ridge/RandomForest/LightGBM RUL models, engine-grouped validation, last-cycle test
evaluation, saved models, local experiment reports and pytest tests. Anomaly
detection, maintenance rules, MLflow and the dashboard are later phases.

## Setup

Use Python 3.10 or newer. In PowerShell, from the repo root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, use the full path to an installed Python executable in
the first command. No environment activation or PowerShell policy changes are
required. A project-local `.venv` is already prepared in this workspace.

On Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Phase 1 uses NumPy, pandas, PyYAML and pytest. Phase 2 adds the approved
scikit-learn/LightGBM stack. Direct and transitive dependencies are pinned;
tracking/dashboard dependencies are deferred to their phases.

## Run Phase 1

Put the three official FD001 files in `data/raw/`; see
[download instructions](scripts/download_data.md). The files are already present
in this workspace.

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --subset FD001 --stage data
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
With an active environment you can also use `make data` and `make test`.

The CLI resolves configured relative paths against the repo root. Use
`--config configs/base.yaml` to select a config. `agent`, `all`, and non-FD001
preprocessing/training stop with an explicit message until their phases exist.

## What the data stage does

1. Validate the 26-column schema, positive integer IDs/cycles, consecutive
   engine histories, unique observations, and one last-cycle RUL label per test engine.
2. Learn missing-value fallback medians from training sensor readings only.
3. Fill gaps within each engine from its last observed reading. Leading missing
   values use the learned train median. Smooth with trailing EWMA (default) or
   trailing rolling median. A new engine starts a new smoothing history.
4. Drop sensors with cleaned train standard deviation at or below the configured
   threshold (plus all-missing sensors). Reuse that exact sensor list on test.
5. Normalize each retained sensor with the cleaned train mean and population
   standard deviation. Test data never updates these statistics.
6. For each sensor, keep its current value and add trailing mean, population std,
   and slope for windows 5/10/30, plus deviation from its early-life baseline.
   Partial windows use available readings; one-point std and slope are zero.
7. Add `rul = min(last_training_cycle - current_cycle, rul_cap)` to **train only**.

The spec asks for interpolation/backfill as well as causal features. Two-sided
interpolation and backward filling use future readings, so this implementation
prioritizes causality and uses forward filling. During the first baseline cycles,
the baseline mean uses the prefix observed so far, then freezes. Configured
baseline size and smoothing/window settings live in `configs/base.yaml`.

Transforms require full histories from cycle 1; to process an arriving cycle,
include that engine's history. Streaming state updates are not implemented yet.

## Output files

Outputs live in `data/processed/FD001/`:

| File | Contents |
| --- | --- |
| `train_features.csv` | Engine/cycle IDs, features, and capped train `rul` |
| `test_features.csv` | Engine/cycle IDs and features at every observed test cycle; no targets |
| `test_rul.csv` | `unit, true_rul`, ground truth at each test engine's last observed cycle |
| `preprocessor.json` | Selected/dropped sensors, fill medians, scaling stats and cleaning settings |
| `config.json` | The resolved run settings |
| `data_report.json` | Actual counts, feature names, missing-value checks and target range |

`unit` and `cycle` are metadata, not model input features. Final test evaluation
will select the last row per engine and join `test_rul.csv` by `unit`; test labels
remain uncapped. Artifacts and raw data are excluded from Git, but empty data
directories are kept.

**Phase 2 implements fold-local preprocessing:** it splits raw engines using
GroupKFold, fits a new `SensorPreprocessor` on each fold's training engines, then
transforms its validation engines. The exported Phase 1 feature CSV is fitted on
all training engines and is not used for cross-validation. Frozen train-fitted
statistics can use all training readings; causal features mean each transformed
row's engine history uses only current and past cycles under that frozen state.

## Run Phase 2

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_pipeline.py --subset FD001 --stage train
.\.venv\Scripts\python.exe -m pytest -q
```

`--stage train` loads raw data itself; a preceding data-stage command is not
required. The default uses five deterministic shuffled engine folds and two CPU
workers. Each fold builds features once and reuses them for all three models.
Rolling slopes use a vectorized covariance/variance formula, equivalent to the
Phase 1 least-squares calculation, to keep repeated preprocessing lightweight.
All model tuning settings live in `configs/base.yaml`.

- **Ridge:** a simple linear relationship, with regularization to reduce large
  coefficients. An additional scaler for engineered features is learned within
  each training fold by its scikit-learn pipeline.
- **RandomForest:** averages a collection of decision trees with bounded depth.
- **LightGBM:** fits boosted trees, where each new tree addresses earlier errors.

Model inputs exclude `unit`, `cycle`, and targets. Validation evaluates all
held-out cycles against capped targets. The model with the lowest pooled
out-of-fold RMSE is selected before any test scores are calculated. Each model is
then refitted on all training engines and compared on the last observed cycle
of each test engine, joining the official uncapped truth by engine ID.

Outputs are clipped to `0..rul_cap` by default for all three models, configurable
with `training.clip_predictions`. The official test labels are never clipped.
This means engines with true test RUL above the training cap can still have
substantial error even when the model correctly identifies a healthy engine.

Metrics are RMSE and MAE in cycles, plus the summed NASA asymmetric score.
Overestimating remaining life is penalized more strongly than underestimating
it. Lower is better. NASA totals from validation and test cannot be compared
directly: their sample counts and target definitions differ. No test-driven
parameter search or early stopping is performed.

## Phase 2 artifacts

Files in `artifacts/reports/FD001/`:

| File | Contents |
| --- | --- |
| `metrics.json` | All model parameters/metrics, selected model, config, versions, input hashes and timing |
| `metrics.md` | Readable table of all three validation and test metrics |
| `fold_metrics.csv` | Metrics and timing for each model/fold |
| `fold_audit.json` | Training/validation engine IDs and fitted sensor preprocessing state per fold |
| `oof_predictions.csv` | Exactly one held-out prediction per training row/model, with capped truth |
| `test_predictions.csv` | Predictions for every observed test cycle/model, without test targets |
| `test_last_cycle_predictions.csv` | One test row per engine/model with the uncapped official truth |

Files in `artifacts/models/FD001/`:

| File | Contents |
| --- | --- |
| `ridge.joblib` | Fitted Ridge model plus preprocessing, feature order and configuration |
| `random_forest.joblib` | The same bundle for RandomForest |
| `lightgbm.joblib` | The same bundle for LightGBM |
| `preprocessor.json` | Final sensor preprocessing fitted on all training engines |
| `metadata.json` | Selected model name, feature order, config and environment |

These local outputs are ignored by Git and replaced when the same subset is run
again. Local files record all three experiments; MLflow runs/registry and model
version management are Phase 4. Model bundles contain Python objects: use the
locally generated files and the matching pinned environment for reloading.

For example, after training, load the selected bundle in Python:

```python
import json
from pathlib import Path
import joblib
from src.data.load import load_subset

folder = Path("artifacts/models/FD001")
metadata = json.loads((folder / "metadata.json").read_text())
predictor = joblib.load(folder / f"{metadata['selected_model']}.joblib")
raw_test = load_subset(Path("data/raw"), "FD001").test
predictions = predictor.predict(raw_test)  # full histories from cycle 1
```

With an active environment, `make train` is the equivalent training shortcut.

## Verified Phase 2 result

The supplied FD001 files and default YAML produced these actual results:

| Model | Pooled CV RMSE | Test RMSE | Test MAE | Test NASA score |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 15.614 | 16.184 | 13.041 | 390.200 |
| RandomForest | 11.737 | 13.398 | 9.939 | 261.450 |
| LightGBM | 11.896 | 14.520 | 11.185 | 299.863 |

RandomForest was selected by CV RMSE. Test evaluation covered all 100 engines
with uncapped official RUL ranging from 7 to 145 cycles. The run took 777.4 seconds
(about 13 minutes) in this workspace with two CPU workers; runtime depends on
hardware. All 60 tests passed. The extra real-data audit verified train-only fold
statistics, held-out engine membership, last-cycle/label alignment, recomputed
metrics, saved inference and causal prediction prefixes. The result slightly
below the spec's rough RMSE range passed those checks; no test-driven tuning was
performed. See `artifacts/reports/FD001/metrics.md` for all three metrics on both
validation and test, and `metrics.json` for unrounded values and run provenance.

See [DESIGN.md](DESIGN.md) for a plain-English explanation of every source file.
