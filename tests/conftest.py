"""Small artificial examples for unit tests; never used as project data."""

import pandas as pd
import pytest

from src.config import load_config
from src.data.load import COLUMNS
from pathlib import Path


@pytest.fixture
def config() -> dict:
    """Use the real config so the tests follow the configured defaults."""
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def raw_train() -> pd.DataFrame:
    """Two engines: sensor 1 varies, sensor 2 barely varies, others constant."""
    rows = []
    for unit, readings in ((1, [2., 4., 6., 8., 10., 12.]), (2, [20., 22., 24., 26.])):
        for cycle, value in enumerate(readings, start=1):
            sensors = [value, 7. + cycle * 0.000001, *([5.] * 19)]
            rows.append([unit, cycle, 0., 0., 0., *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)
