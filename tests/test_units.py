"""Unit tests on synthetic data: run anywhere (CI included), no downloads needed."""
import numpy as np
import pandas as pd
import pytest

from wim_forecast.build_dataset import logistics_calendar, schedule, sensor_ok
from wim_forecast.common import COST_IDLE, COST_SHORT, Q_OPT, dispatch_cost, folds, to_label
from wim_forecast.conformal import conf_q
from wim_forecast.evaluate import block_bootstrap_ci, diebold_mariano


@pytest.fixture
def weekly():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    base = np.where(idx.day_of_week < 5, 1000.0, 300.0)   # weekday vs weekend level
    return pd.Series(base + np.arange(70), index=idx)


def test_schedule_is_same_weekday_median(weekly):
    ok = pd.Series(True, index=weekly.index)
    cfg = pd.Series(2, index=weekly.index)
    s = schedule(weekly, ok, cfg)
    d = weekly.index[40]
    expected = np.median([weekly[d - pd.Timedelta(weeks=k)] for k in range(1, 5)])
    assert s[d] == expected
    assert s.iloc[:21].isna().all()          # needs >= 3 past weeks


def test_schedule_skips_invalid_days_and_other_lane_configs(weekly):
    ok = pd.Series(True, index=weekly.index)
    cfg = pd.Series(2, index=weekly.index)
    d = weekly.index[50]
    ok[d - pd.Timedelta(weeks=1)] = False     # outage last week
    s1 = schedule(weekly, ok, cfg)
    assert s1[d] == np.median([weekly[d - pd.Timedelta(weeks=k)] for k in (2, 3, 4, 5)])
    cfg2 = cfg.copy()
    cfg2[d - pd.Timedelta(weeks=2)] = 3       # different lane configuration two weeks ago
    s2 = schedule(weekly, ok, cfg2)
    assert s2[d] == np.median([weekly[d - pd.Timedelta(weeks=k)] for k in (3, 4, 5, 6)])


def test_labels_and_newsvendor_quantile():
    assert to_label([0.85, 0.95, 1.0, 1.05, 1.15]).tolist() == [0, 1, 1, 1, 2]
    assert Q_OPT == pytest.approx(COST_SHORT / (COST_SHORT + COST_IDLE))


def test_dispatch_cost_is_asymmetric():
    actual = np.array([100.0, 100.0])
    assert dispatch_cost(actual, np.array([90.0, 100.0])) == pytest.approx(10 * COST_SHORT)
    assert dispatch_cost(actual, np.array([110.0, 100.0])) == pytest.approx(10 * COST_IDLE)


def test_sensor_ok_drops_incomplete_days():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    d = pd.DataFrame({"hours_obs": 24, "lanes_alive": 2, "lanes_configured": 2, "cars": 40000.0}, index=idx)
    d.loc[idx[30], "hours_obs"] = 20
    d.loc[idx[31], "lanes_alive"] = 1
    d.loc[idx[32], "cars"] = 5000.0
    ok = sensor_ok(d)
    assert not ok[idx[30]] and not ok[idx[31]] and not ok[idx[32]]
    assert ok[idx[33]]


def test_logistics_calendar_known_dates():
    c = logistics_calendar(pd.date_range("2023-01-01", "2025-12-31"))
    assert c.index[c.black_friday == 1].strftime("%Y-%m-%d").tolist() == ["2023-11-24", "2024-11-29", "2025-11-28"]
    assert c.loc["2023-09-25", "jewish_holiday"] == 1      # Yom Kippur
    assert c.loc["2023-04-26", "jewish_holiday"] == 0      # Israeli Independence Day is excluded


def test_folds_are_time_ordered_and_disjoint():
    idx = pd.date_range("2021-01-01", "2026-09-30", freq="D")
    df = pd.DataFrame({"target_ok": True}, index=idx)
    prev_end = None
    for _, fit, es, test in folds(df):
        assert not (fit & es).any() and not (fit & test).any() and not (es & test).any()
        assert df.index[fit].max() < df.index[es].min() < df.index[test].min()
        if prev_end is not None:
            assert df.index[test].min() > prev_end
        prev_end = df.index[test].max()


def test_conformal_quantile_covers_on_average():
    """The guarantee is marginal: averaged over calibration draws, coverage >= 1 - alpha."""
    rng = np.random.default_rng(0)
    new = np.abs(rng.normal(size=20000))
    cov = [(new <= conf_q(np.abs(rng.normal(size=200)), 0.1)).mean() for _ in range(300)]
    assert 0.90 <= np.mean(cov) < 0.92


def test_diebold_mariano_and_bootstrap():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 1, 700)
    dates = pd.date_range("2024-01-01", periods=700)
    assert diebold_mariano(a, a + rng.normal(0, 1, 700)) > 0.01          # same expected loss
    assert diebold_mariano(a, a + 1 + rng.normal(0, 1, 700)) < 1e-6      # clearly different
    lo, hi = block_bootstrap_ci(a * 0.8 + rng.normal(0, 0.5, 700), a, dates)
    assert lo < -0.2 < hi < 0
