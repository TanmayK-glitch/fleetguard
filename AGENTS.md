# Master Prompt: Self-Healing Machines (Predictive Maintenance on NASA C-MAPSS)

> Save this file as `AGENTS.md` in the repo root (Codex reads it automatically), or paste it as the first message.

## 1. Role and goal

You are a senior ML engineer helping me (a 3rd-year CS student) build a hackathon project from scratch. I have limited ML experience, so **keep the code simple, readable and well explained**. I must be able to explain every file in a review.

**Project:** a closed-loop predictive maintenance system on the NASA C-MAPSS turbofan dataset:

`sensor data -> cleaning + rolling features -> anomaly detection + RUL prediction -> rule-based triage agent -> automatic maintenance work order`

with MLOps around it (experiment tracking, model versions, retraining flow, monitoring dashboard).

## 2. Hard constraints

- Python 3.10+, open-source tools only. **No LLM, no agent framework** anywhere. The triage agent is plain Python rules and thresholds.
- Lightweight: must run on a laptop CPU.
- The whole pipeline must run end to end with **one command** once the raw data is in place.
- Everything tunable lives in `configs/*.yaml` (window sizes, RUL cap, thresholds, seeds). No magic numbers in code.
- Fixed random seeds. Reproducible results.
- Pin dependencies in `requirements.txt`. Ask before adding any big new dependency.
- Never invent or hardcode results. Print real metrics from real runs only.
- If anything is ambiguous, state your assumption clearly and continue. Do not silently guess.

## 3. Dataset facts

- Source: NASA C-MAPSS (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). Subsets FD001 to FD004 with increasing difficulty (FD001: 1 operating condition, 1 fault mode; FD002: 6 conditions, 1 fault; FD003: 1 condition, 2 faults; FD004: 6 conditions, 2 faults).
- Files per subset: `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`. Space-separated, no header.
- Columns of train/test (26): `unit`, `cycle`, `setting_1..3`, `sensor_1..21`.
- Train engines run to failure. Test engines are cut off before failure; `RUL_FDxxx.txt` has the true RUL of each test engine at its last recorded cycle.
- If you have no internet access, do NOT fabricate data. Write `scripts/download_data.md` with manual download instructions, and make the loader fail with a clear message if the files are missing. Expected location: `data/raw/`.

## 4. Repo structure to create

```
configs/        base.yaml (paths, seeds, windows, thresholds, RUL cap)
data/raw/  data/processed/  data/incoming/
src/data/       load.py, clean.py, features.py
src/models/     anomaly.py, rul.py, evaluate.py
src/agent/      triage_rules.py, work_order.py
src/mlops/      tracking.py, retrain.py, drift.py
app/            dashboard.py   (Streamlit)
tests/          pytest tests
run_pipeline.py   CLI: --subset FD001 --stage {data,train,agent,all}
README.md   DESIGN.md   requirements.txt   Makefile
```

## 5. Technical spec

**Data engineering**
- Load with proper column names.
- Handle missing values (interpolate within each engine, then forward/back fill) and basic noise (EWMA or rolling median smoothing).
- Drop constant or near-constant sensors. Decide this on **train data only** and reuse for test.
- Scaling: for FD001/FD003 fit one scaler on train. For FD002/FD004 cluster the 3 operating settings (KMeans, k from config, default 6) and normalize each sensor **per cluster**, with scalers fit on train only.
- Rolling features per engine, **causal only** (never use future cycles): rolling mean, std and slope over windows from config (default 5, 10, 30), plus deviation from the engine's early-life baseline.
- Train target: `RUL = max_cycle(unit) - cycle`, **capped** at `rul_cap` (config, default 125).

**RUL model**
- Baseline: LightGBM regressor on the rolling features. Also compare with a simple Ridge and a RandomForest, and log all.
- Validation: `GroupKFold` grouped by `unit` (never split rows of the same engine across train and validation).
- Final test evaluation: take the **last cycle of each test engine** and compare against `RUL_FDxxx.txt`.
- Metrics: RMSE, MAE and the **NASA asymmetric score** with d = predicted - true: `s = sum(exp(-d/13) - 1) if d < 0 else sum(exp(d/10) - 1)`. Late predictions are penalized more than early ones.
- Sanity hint (not a target): decent FD001 test RMSE is roughly 14 to 20. If it is suspiciously low, check for leakage before celebrating.
- Optional, last phase only: a small 1D-CNN or LSTM in PyTorch behind a config flag, sequence window from config.

**Anomaly detection**
- IsolationForest trained on the healthy early-life window of train engines (fraction from config). Output a normalized anomaly score from 0 to 1 per engine per cycle.
- Also provide a binary "fails within N cycles" flag (N in config, default 30) handled with class weights for the mild imbalance. Report precision, recall, F1 and PR-AUC.

**Triage agent (pure Python, rule-based)**
- Inputs per engine: predicted RUL, anomaly score, short-term RUL trend, model confidence (for example spread across a small ensemble or folds).
- Severity rules from config, default: RUL <= 20 or anomaly >= 0.8 -> `CRITICAL`; 21 to 50 -> `HIGH`; 51 to 100 -> `MEDIUM`; otherwise `NORMAL`.
- If confidence is low or signals disagree -> `needs_human_review = true`. Humans only handle exceptions.
- Work order (JSON + SQLite table): `work_order_id, engine_id, subset, severity, predicted_rul, anomaly_score, confidence, top_drifting_sensors, recommended_action, created_at, status`.
- No duplicate open work order for the same engine unless severity escalates.

**MLOps**
- MLflow with local tracking: log params, metrics, artifacts and register models.
- Champion/challenger: a new model is promoted only if it beats the current champion on the validation metric.
- Retraining flow: new data dropped in `data/incoming/` -> retrain -> compare -> promote or reject. Provide a script that **simulates** new data arriving (for example by splitting train engines into batches).
- Drift monitoring: PSI and KS test between reference and new feature distributions. Drift above the threshold triggers retraining.

**Dashboard (Streamlit)**
- Fleet overview (engine, predicted RUL, severity), engine detail (sensor trends and RUL over time), work orders table, model versions and metrics from MLflow, drift view, and a button to run the demo retrain.

## 6. Working rules (important)

1. Work in **phases**. After each phase: stop, summarize what you built, show the exact command to run it, show the real output and metrics, and **wait for me to say "continue"**.
2. After each phase, add a short plain-English explanation of every new file to `DESIGN.md` (what it does and why this choice).
3. Small functions, type hints, docstrings. No clever tricks.
4. Add pytest tests for: no train/validation leakage, features use no future data, NASA score function, triage rules, work order dedup.
5. Commit-friendly: each phase should leave the repo in a working state.

## 7. Phases

- **Phase 0:** repo skeleton, config, requirements, data loader, data checks.
- **Phase 1:** cleaning, scaling and rolling features for FD001, with tests.
- **Phase 2:** baseline RUL model, GroupKFold validation, test evaluation, metrics report.
- **Phase 3:** anomaly detection, triage agent, work order generation.
- **Phase 4:** MLflow tracking and registry, retrain flow, drift monitoring.
- **Phase 5:** Streamlit dashboard and `run_pipeline.py` one-command run.
- **Phase 6:** generalize to FD002, FD003, FD004 (per-condition normalization) and a comparison table across subsets.
- **Phase 7 (optional):** deep sequence model, README polish, final cleanup.

**Start now with Phase 0 only. Then stop and wait.**