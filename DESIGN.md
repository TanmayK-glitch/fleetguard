# Design through Phase 1

The repo originally contained instructions and raw data. The user requested
Phase 1, so we added its necessary Phase 0 foundations in the same implementation.
We stop here; later ML, agent, MLOps and dashboard phases remain unimplemented.

## Decisions you can explain in a review

- **Training-only fit:** training data chooses sensors, fill medians, means and
  scales. Test data only passes through these frozen choices. Before future
  GroupKFold evaluation, fit preprocessing separately inside every training fold.
- **Causal gaps:** a missing reading uses the last earlier reading from the same
  engine. If none exists, use a train median. We explicitly prioritize causality
  over the spec's two-sided interpolation/backfill request, because future fills
  would change earlier features when new data arrives.
- **Causal smoothing:** trailing EWMA or rolling median never looks ahead. EWMA
  uses the recursive definition (`adjust=False`). Each engine has separate history.
- **Sensor selection:** drop all-missing sensors and sensors whose cleaned train
  population std is at or below the YAML threshold, measured in original sensor
  units. A raw-unit threshold is simple but unit-dependent; it is configurable.
  For the default FD001 run we use 0.01: cleaned train `sensor_6` std is about
  0.000705, while the next smallest retained sensor std is about 0.031826.
  This choice uses training statistics only, with no test labels or model metrics.
- **Scaling:** subtract the cleaned train mean and divide by the cleaned train
  population std. FD001 has one operating condition, so one set of statistics
  suffices. NumPy/pandas implement this short formula without a new ML dependency.
- **Features:** current scaled sensors, trailing means/std/slopes, and baseline
  deltas. The baseline is the available prefix of the first configured cycles;
  it freezes when that window finishes. All windows include the current cycle.
  Population std and a zero single-point slope are definitions, not tuning knobs.
- **Targets stay separate:** train RUL comes from run-to-failure cycle counts and
  is capped. Official test RUL is kept uncapped in a separate file and never
  enters feature building. Engine/cycle IDs remain metadata.
- **Repeatability:** preprocessing is deterministic; the configured random seed
  is set for future stochastic components. The saved config/state and tests make
  repeated runs inspectable. No synthetic data or model results are fabricated.
- **Batch interface:** transformations take histories beginning at cycle 1.
  Online single-row updates will need persisted per-engine state in a later phase.

## Every new project file

| File | What it does and why |
| --- | --- |
| `.gitignore` | Keeps raw/generated data, environments and test caches out of commits. |
| `requirements.txt` | Pins the small Phase 1 runtime/test stack and its dependencies. |
| `configs/base.yaml` | Holds paths, seed, smoothing settings, sensor cutoff, windows, baseline size and RUL cap. |
| `pytest.ini` | Finds tests and imports the repo; keeps temporary fixtures inside the workspace. |
| `Makefile` | Offers short data/test commands where Make is installed; Python CLI also works directly on Windows. |
| `README.md` | Explains setup, exact commands, current scope, causality and output artifacts. |
| `DESIGN.md` | This explanation of choices/files, for a student review and phase handoff. |
| `scripts/download_data.md` | Gives official manual download instructions and the required filenames. |
| `run_pipeline.py` | Parses subset/stage/config arguments, runs the data stage and prints actual statistics. |
| `src/__init__.py` | Marks project code as an importable Python package. |
| `src/config.py` | Loads YAML and rejects missing or invalid tuning values before processing. |
| `src/data/__init__.py` | Marks the data components as a package. |
| `src/data/load.py` | Assigns all 26 column names, validates trajectories and aligns test RUL by engine. |
| `src/data/clean.py` | Fits training-only medians/sensor selection/scaling; cleans histories causally; saves/reloads JSON state. |
| `src/data/features.py` | Builds per-engine trailing features and computes capped train RUL in a separate function. |
| `src/data/pipeline.py` | Combines loading, train-fit preprocessing, features and artifact/report writing for FD001. |
| `src/models/__init__.py` | Reserves the model package for Phase 2/3 without pretending models exist. |
| `src/agent/__init__.py` | Reserves the rule-based triage package for Phase 3. |
| `src/mlops/__init__.py` | Reserves tracking/retraining/drift components for Phase 4. |
| `app/.gitkeep` | Keeps the future dashboard directory in Git while it is empty. |
| `data/raw/.gitkeep` | Keeps the directory for official input files in Git. |
| `data/processed/.gitkeep` | Keeps the generated-artifact directory in Git. |
| `data/incoming/.gitkeep` | Keeps the directory for future arriving-data batches in Git. |
| `tests/conftest.py` | Supplies YAML settings and explicitly artificial tiny engine histories for unit tests only. |
| `tests/test_load.py` | Checks schema/trajectory errors, missing-file guidance and test-label alignment. |
| `tests/test_clean.py` | Checks train-only fit, sensor removal, scaling, gaps, engine resets and saved-state reuse. |
| `tests/test_features.py` | Checks statistics by hand, causal prefix invariance, future perturbations, engine isolation and target capping. |
| `tests/test_pipeline.py` | Checks artifact repeatability, test-label isolation, subset scope and bad settings. |

The existing `AGENTS.md` and official raw text files are unchanged. Installed
`.venv` files, test caches and generated artifacts are local outputs, not source.
Generated CSV/JSON files are described in the README output table.

## Deferred checks

GroupKFold engine separation and fold-local preprocessing, NASA score and model
metrics belong to Phase 2. Triage rules and work-order dedup tests belong to Phase 3.
Adding those tests now would imply those components already exist.

## Phase 1 verification on the supplied FD001 files

Run from the repo root in PowerShell:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --subset FD001 --stage data
.\.venv\Scripts\python.exe -m pytest -q
```

Actual data-stage output with the saved default config:

```text
Prepared FD001
Train: 100 engines, 20,631 rows
Test:  100 engines, 13,096 rows
Sensors: 14 retained, 7 dropped
Dropped: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
Features: 154; nonfinite values: 0
Raw missing sensor readings: train=0, test=0
Train RUL range: 0..125
Artifacts: C:\Users\Asus\Documents\self-healing-machines\data\processed\FD001
Model metrics: not available until Phase 2.
```

Actual final pytest output:

```text
..............................                                           [100%]
30 passed in 0.93s
```

The environment used Python 3.12.14 with the pinned requirements. Dependency
checking also reported `No broken requirements found.` These are data/test
results, not RMSE/MAE/NASA model scores. Phase 2 has not been started.
