# Viva preparation: 25 likely judge questions

These answers describe the implemented **Phases 1 and 2**. Features marked
**planned** are future work, not completed capabilities. Measured numbers below
come from the saved FD001 run with the default config; a new run may differ if
you change the settings.

## 1. What problem does your project solve?

We estimate how many operating cycles a turbofan engine has left from its sensor
history. The planned closed loop uses that estimate and anomaly signals to
prioritize maintenance and create work orders automatically.

## 2. What does “self-healing” mean here?

It means an automated maintenance workflow, not an engine physically repairing
itself. Today we have preprocessing and RUL prediction; automatic triage and
work orders are planned for Phase 3.

## 3. What dataset are you using?

NASA C-MAPSS contains simulated run-to-failure turbofan sensor histories. Each
row has an engine ID, cycle, three operating settings and 21 sensors; the
current FD001 files have 100 training and 100 test engines.

## 4. Why start with FD001?

It has one operating condition and one fault mode, making it easier to understand
and debug. FD002–FD004 introduce more operating conditions or fault modes and
are planned for Phase 6.

## 5. What is Remaining Useful Life, and how do you create the training label?

RUL is the number of cycles remaining before failure. Training engines run to
failure, so the uncapped label is `last cycle of that engine − current cycle`.

## 6. Why cap training RUL at 125?

The cap expresses the modeling assumption that healthy early-life readings do
not reveal exact far-away failure time reliably. We use `min(RUL, 125)` by
default; the cap is configurable rather than a known law of engine behavior.

## 7. How is test evaluation different from training labels?

Test engines stop before failure, so we use NASA's supplied RUL at each engine's
last observed cycle. We join by engine ID and keep that truth uncapped, even
though predictions are clipped to the training cap by default.

## 8. How do you handle missing sensor readings?

Within an engine, a missing reading uses the latest previous reading. Leading
gaps use a median learned from training data only; the supplied FD001 run has
no missing sensor values, but the behavior is tested with small test fixtures.

## 9. Why did you avoid two-sided interpolation and backward filling?

They can use a later reading to change an earlier cycle. We explicitly
prioritized causal features, so an arriving future cycle cannot alter features
that were already available in the past.

## 10. What is the difference between smoothing and scaling?

Smoothing reduces short-term noise using past readings, with EWMA by default.
Scaling changes numerical units by subtracting a training mean and dividing by
a training standard deviation; it does not remove noise.

## 11. Why remove constant and near-constant sensors?

Sensors that barely change provide little discrimination between engine states.
We select sensors using cleaned training standard deviation and reuse the same
list on test data; the default FD001 run retains 14 sensors.

## 12. Why fit preprocessing on training data only?

If validation or test data influences medians, scaling or sensor selection, the
model receives information it should not have during evaluation. Frozen
training-only choices make the held-out comparison more trustworthy.

## 13. What rolling features did you build?

For each retained sensor we keep its current scaled reading, trailing mean,
standard deviation and slope over configured windows, plus a baseline deviation.
The default 5/10/30-cycle windows produce 154 model input features in FD001.

## 14. What is the early-life baseline, and can it leak future readings?

It is an engine's mean over its first configured baseline cycles. During
warm-up we use only the prefix observed so far, then freeze the baseline once
the window is complete; the default window is 10 cycles.

## 15. Why use GroupKFold instead of randomly splitting rows?

Nearby cycles from the same engine are strongly related. GroupKFold keeps an
engine entirely in training or validation, testing generalization to held-out
engines rather than recognition of an engine already seen during training.

## 16. Is grouping engines enough to prevent leakage?

No: preprocessing must also be refitted within each training fold. We start
from raw histories and fit fresh medians, sensor selection and scaling before
transforming that fold's validation engines; Ridge's extra scaler is fold-local too.

## 17. Why compare Ridge, RandomForest and LightGBM?

Ridge provides a simple regularized linear baseline. RandomForest averages
decision trees, while LightGBM boosts trees to address previous errors; comparing
all three shows whether extra model complexity helps on the same validation folds.

## 18. How did you select the best model?

We pool predictions from the five held-out folds and choose the lowest
validation RMSE. RandomForest won this run; test scores are calculated afterward
and do not choose the winner.

## 19. What do RMSE, MAE and the NASA score tell you?

MAE is average absolute error in cycles, while RMSE gives larger errors more
weight. The NASA score uses `d = predicted − true` and sums `exp(-d/13)-1` for
negative errors or `exp(d/10)-1` otherwise, penalizing overestimation more strongly.

## 20. What are your actual results?

The verified RandomForest run has pooled CV RMSE **11.737**, test RMSE **13.398**,
test MAE **9.939**, and test NASA score **261.450**. Test RMSE is **16.184** for
Ridge and **14.520** for LightGBM; the full report records all models and metrics.

## 21. Why is validation error different from test error?

Validation measures all held-out training cycles against capped targets. Test
evaluation measures only each engine's last observed cycle against uncapped
official truth, so both the sample population and target definition differ.

## 22. How did you investigate the unusually good test RMSE?

We verified disjoint engine folds and recomputed every saved fold's training-only
preprocessing statistics. We also checked target/feature separation, official
label alignment, last-cycle selection, recomputed metrics and causal prefixes;
those checks found no leakage in the tested paths.

## 23. Is this reproducible and lightweight enough for a laptop?

Settings and seeds live in YAML, dependencies are pinned, and reports save
versions and input hashes. The verified default five-fold run took about 13
minutes with two CPU workers; saved models can be reused without retraining.

## 24. Where are anomaly detection, the triage agent and MLOps?

Phase 3 will add anomaly detection, a near-failure flag, plain Python severity
rules, human review for exceptions and deduplicated work orders. Phase 4 will
add MLflow, drift checks and validation-based retraining promotion; today we save
local experiment reports and model bundles.

## 25. What are the main limitations and next steps?

We have evaluated only simulated FD001 data, and RUL estimates are not fault
diagnoses or validated maintenance instructions. Next steps are uncertainty and
anomaly signals, the maintenance loop, monitoring/dashboard work and validation
across more operating conditions before considering real operational use.

For the code choices see [DESIGN.md](DESIGN.md), for commands see
[README.md](README.md), and for measured results see the generated
[Phase 2 metrics](artifacts/reports/FD001/metrics.md). The dataset source is
[NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
