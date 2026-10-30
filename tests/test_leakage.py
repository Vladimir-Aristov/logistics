"""Leakage tests: features for day D must not change if we hide what is unknown at forecast time.

We rebuild the whole table twice — on full data and with observations hidden —
and require identical features at D.

* D-1 mode (forecast at the end of D-1): everything from day D on is hidden,
  WIM and MTA alike. Only same-day weather (stand-in for the D-1 forecast) may differ.
* Nowcast mode (forecast at 07:00 on D): same, except the morning of day D stays visible.
"""
import numpy as np
import pandas as pd
import pytest

from wim_forecast.build_dataset import (
    MORNING_COLS,
    OBS_COLS,
    PROC,
    RAW,
    SAME_DAY_EXOG,
    feature_cols,
    load_exog,
    make_table,
)

pytestmark = pytest.mark.skipif(not (PROC / "daily_obs.parquet").exists() or not (RAW / "mta_hourly.parquet").exists(),
                                reason="run src/external.py and src/build_dataset.py first")
MTA_DAY_COLS = ["mta_trucks", "mta_trucks_carey", "mta_trucks_vz"]


@pytest.fixture(scope="module")
def data():
    obs = pd.read_parquet(PROC / "daily_obs.parquet")
    w, dsl, mta = load_exog()
    return obs, w, dsl, mta


def cutoffs(obs, n=10, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(obs.index[60:], n, replace=False))


def hide(obs, mta, day, keep_morning: bool):
    o, m = obs.copy(), mta.copy()
    keep_o = o.loc[day, MORNING_COLS].copy()
    keep_m = m.loc[day, "mta_morning_trucks"] if day in m.index else np.nan
    o.loc[day:, OBS_COLS] = np.nan
    m.loc[day:, :] = np.nan
    if keep_morning:
        o.loc[day, MORNING_COLS] = keep_o
        if day in m.index:
            m.loc[day, "mta_morning_trucks"] = keep_m
    return o, m


@pytest.mark.parametrize("nowcast", [False, True], ids=["d-1", "nowcast"])
def test_features_use_only_known_data(data, nowcast):
    obs, w, dsl, mta = data
    full = make_table(obs, w, dsl, mta, nowcast=nowcast)
    cols = [c for c in feature_cols(full) if c not in SAME_DAY_EXOG]
    for day in cutoffs(obs):
        o, m = hide(obs, mta, day, keep_morning=nowcast)
        trunc = make_table(o, w, dsl, m, nowcast=nowcast)
        a, b = full.loc[day, cols], trunc.loc[day, cols]
        diff = [c for c in cols if not np.isclose(a[c], b[c], equal_nan=True)]
        assert not diff, f"{pd.Timestamp(day).date()}: features leak future data: {diff}"


def test_nowcast_actually_uses_the_morning(data):
    """The nowcast test must be able to fail: hiding the morning changes morning features."""
    obs, w, dsl, mta = data
    full = make_table(obs, w, dsl, mta, nowcast=True)
    day = full.index[full.target_ok][-100]
    o, m = hide(obs, mta, day, keep_morning=False)
    assert np.isnan(make_table(o, w, dsl, m, nowcast=True).loc[day, "morning_ratio"])
    assert not np.isnan(full.loc[day, "morning_ratio"])


def test_target_does_change_when_hidden(data):
    obs, w, dsl, mta = data
    full = make_table(obs, w, dsl, mta)
    day = full.index[full.target_ok][-100]
    o, m = hide(obs, mta, day, keep_morning=False)
    assert np.isnan(make_table(o, w, dsl, m).loc[day, "ratio"])
