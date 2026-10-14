"""Build trailing-window features without peeking at future engine cycles."""

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """Least-squares slope per cycle for consecutive observations.

    Centering the positions makes this formula easy to read and numerically
    stable. A single observation has zero slope by convention.
    """
    if len(values) == 1:
        return 0.0
    positions = np.arange(len(values), dtype=float)
    centered = positions - positions.mean()
    return float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))


def build_features(
    cleaned: pd.DataFrame,
    sensor_columns: list[str],
    windows: list[int],
    baseline_cycles: int,
) -> pd.DataFrame:
    """Return IDs, current sensors, rolling mean/std/slope and baseline deltas.

    The loader guarantees consecutive cycles starting at 1. Windows include
    the current cycle. Partial windows are valid; population std is zero for
    one observation. Before the baseline window is complete, use only its
    available prefix; after it is complete, freeze its mean.
    """
    if not windows or any(type(w) is not int or w <= 0 for w in windows):
        raise ValueError("windows must be nonempty positive integer sizes.")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must contain distinct sizes.")
    if type(baseline_cycles) is not int or baseline_cycles <= 0:
        raise ValueError("baseline_cycles must be a positive integer.")
    frame = cleaned.sort_values(["unit", "cycle"]).reset_index(drop=True)
    results = []
    for _, engine in frame.groupby("unit", sort=True):
        if not np.array_equal(engine["cycle"].to_numpy(), np.arange(1, len(engine) + 1)):
            raise ValueError("Feature histories must start at cycle 1 and be consecutive.")
        values = engine[sensor_columns]
        blocks = [engine[["unit", "cycle"]], values]
        for window in windows:
            rolling = values.rolling(window, min_periods=1)
            blocks.extend([
                rolling.mean().add_suffix(f"_mean_{window}"),
                rolling.std(ddof=0).add_suffix(f"_std_{window}"),
                rolling.apply(_slope, raw=True).add_suffix(f"_slope_{window}"),
            ])
        # Only already observed baseline readings contribute to earlier rows.
        baseline = values.iloc[:baseline_cycles].expanding(min_periods=1).mean()
        baseline = baseline.reindex(values.index).ffill()
        blocks.append((values - baseline).add_suffix("_baseline_delta"))
        results.append(pd.concat(blocks, axis=1))
    if not results:
        raise ValueError("No engine histories to featurize.")
    return pd.concat(results, ignore_index=True)


def capped_rul(train: pd.DataFrame, cap: int) -> pd.Series:
    """Derive a TRAIN target from run-to-failure histories; never call on test."""
    if type(cap) is not int or cap <= 0:
        raise ValueError("RUL cap must be a positive integer.")
    last_cycle = train.groupby("unit")["cycle"].transform("max")
    return (last_cycle - train["cycle"]).clip(upper=cap).astype(int).rename("rul")
