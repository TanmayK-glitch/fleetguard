# Self-Healing Machines

A student-friendly, CPU-only predictive maintenance project using
[NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
The planned flow is sensors → cleaning/features → RUL and anomaly models →
plain Python triage rules → maintenance work orders.

**Implemented through Phase 1:** FD001 loading, data checks, causal cleaning,
training-only sensor selection/scaling, rolling features, capped training RUL,
saved preprocessing state, and pytest tests. Models, agents, tracking and the
dashboard are later phases. No ML accuracy metrics exist yet.

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

Phase 1 uses NumPy, pandas, PyYAML and pytest. Direct and transitive dependencies
are pinned; model/tracking/dashboard dependencies are deferred to their phases.

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
`--config configs/base.yaml` to select a config. `train`, `agent`, `all`, and
non-FD001 preprocessing stop with an explicit message until their phases exist.

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

**Phase 2 validation requirement:** split raw engines using GroupKFold, fit a new
`SensorPreprocessor` on each fold's training engines, then transform its validation
engines. The exported train feature CSV is fitted on all training engines and must
not be used directly for cross-validation. Frozen train-fitted statistics can
use all training readings; causal features mean each transformed row's engine
history uses only current and past cycles under that frozen state.

See [DESIGN.md](DESIGN.md) for a plain-English explanation of every source file.
