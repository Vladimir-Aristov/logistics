"""Build the daily modelling table.

Business framing
----------------
Think of the Queens-bound BQE truck flow as the outbound demand of a carrier.
Dispatch runs on a "schedule": for each day it plans as many trucks as went
out on the same weekday over the last 4 weeks (median). That schedule is the
baseline every model has to beat.

Target for day D (forecast made at the end of day D-1):
    ratio_D = actual_trucks_D / schedule_D
    label_D = below  if ratio < 1 - BAND
              above  if ratio > 1 + BAND
              normal otherwise
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

TRUCK_CLASSES = range(5, 14)      # FHWA 5-13: single-unit + combination trucks
COMBO_CLASSES = range(8, 14)      # tractor-trailers (fura)
CAR_CLASSES = (2, 3)              # used only as a sensor-health signal
BAND = 0.10                       # +-10% around schedule counts as "normal"
LABELS = ["below", "normal", "above"]
MORNING_HOURS = 7                 # nowcast is issued at 07:00 on day D


# ---------------------------------------------------------------- aggregation
def hourly(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.assign(
        truck=raw["class"].isin(TRUCK_CLASSES) * raw.n,
        combo=raw["class"].isin(COMBO_CLASSES) * raw.n,
        car=raw["class"].isin(CAR_CLASSES) * raw.n,
        truck_gvw=np.where(raw["class"].isin(TRUCK_CLASSES), raw.gvw_lb, 0.0),
    )
    h = raw.groupby("ts").agg(trucks=("truck", "sum"), combo=("combo", "sum"),
                              cars=("car", "sum"), truck_gvw_lb=("truck_gvw", "sum"))
    full = pd.date_range(h.index.min().floor("D"), h.index.max().ceil("D"), freq="h", inclusive="left")
    return h.reindex(full).rename_axis("ts")


def lane_health(raw: pd.DataFrame) -> pd.DataFrame:
    """Per day: how many lanes are alive, and whether that's a full set.

    A lane is alive on a day if its vehicle count is >= 20% of its own
    trailing 28-day median. Lanes are counted per day (not per hour) because a
    quiet lane can legitimately see zero vehicles at 3 a.m.
    """
    per = raw.groupby([raw.ts.dt.floor("D"), "lane"]).n.sum().unstack().sort_index()
    ref = per.rolling(28, min_periods=5).median().shift(1)
    alive = (per >= 0.2 * ref) & per.notna()
    # lanes carrying >= 5% of traffic in the last 14 days define the configuration
    share = per.fillna(0).div(per.sum(axis=1), axis=0)
    configured = (share >= 0.05).rolling(14, min_periods=1).max().astype(bool)
    out = pd.DataFrame({"lanes_alive": alive.sum(axis=1), "lanes_configured": configured.sum(axis=1)})
    out.index.name = "date"
    return out


def daily(h: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby(h.index.floor("D"))
    d = pd.DataFrame({
        "trucks": g.trucks.sum(min_count=1),
        "combo": g.combo.sum(min_count=1),
        "cars": g.cars.sum(min_count=1),
        "tons": g.truck_gvw_lb.sum(min_count=1) / 2000,
        "hours_obs": g.cars.count(),
        "night_trucks": h.trucks.where(h.index.hour < 6).groupby(h.index.floor("D")).sum(min_count=1),
    }).rename_axis("date")
    # first MORNING_HOURS hours of the day: what dispatch has seen by 07:00 (nowcast mode)
    m = h[h.index.hour < MORNING_HOURS]
    gm = m.groupby(m.index.floor("D"))
    d["morning_trucks"] = gm.trucks.sum(min_count=1)
    d["morning_combo"] = gm.combo.sum(min_count=1)
    d["morning_cars"] = gm.cars.sum(min_count=1)
    d["morning_hours"] = gm.cars.count()
    return d.join(lanes)


def sensor_ok(d: pd.DataFrame) -> pd.Series:
    """True when the WIM site recorded a complete day.

    Dropped: missing hours, any configured lane dark or far below normal, or
    car traffic (the control group: trucks may swing, cars should not) collapsed
    vs. its recent level.
    """
    full_day = d.hours_obs == 24
    all_lanes = d.lanes_alive >= d.lanes_configured
    car_ref = d.cars.where(full_day).rolling(28, min_periods=7).median().shift(1)
    cars_ok = (d.cars / car_ref).between(0.6, 1.5)
    return (full_day & all_lanes & cars_ok).fillna(False)


# ---------------------------------------------------------------- schedule
def schedule(series: pd.Series, ok: pd.Series, config: pd.Series, weeks: int = 4) -> pd.Series:
    """Median of the same weekday over the previous `weeks` valid weeks
    that had the same lane configuration as day D."""
    s = series.where(ok)
    lags = pd.concat([s.shift(7 * k).where(config.shift(7 * k) == config) for k in range(1, weeks + 3)], axis=1)
    # use the latest `weeks` valid observations (allow one gap)
    vals = lags.apply(lambda r: r.dropna().iloc[:weeks].median() if r.notna().sum() >= 3 else np.nan, axis=1)
    return vals


# ---------------------------------------------------------------- features
def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    us = holidays.country_holidays("US", subdiv="NY", years=range(idx.year.min(), idx.year.max() + 2))
    hol = pd.Series(pd.to_datetime(list(us.keys()))).sort_values().to_numpy()
    c = pd.DataFrame(index=idx)
    c["dow"] = idx.day_of_week
    c["month"] = idx.month
    c["doy_sin"] = np.sin(2 * np.pi * idx.day_of_year / 365.25)
    c["doy_cos"] = np.cos(2 * np.pi * idx.day_of_year / 365.25)
    c["dom"] = idx.day
    c["is_month_end"] = (idx + pd.Timedelta(days=3)).month != idx.month
    c["is_holiday"] = idx.isin(hol)
    pos = np.searchsorted(hol, idx.to_numpy())
    nxt = hol[np.clip(pos, 0, len(hol) - 1)]
    prv = hol[np.clip(np.searchsorted(hol, idx.to_numpy(), side="right") - 1, 0, len(hol) - 1)]
    c["days_to_hol"] = np.clip((nxt - idx.to_numpy()) / np.timedelta64(1, "D"), 0, 30)
    c["days_from_hol"] = np.clip((idx.to_numpy() - prv) / np.timedelta64(1, "D"), 0, 30)
    # day after / before a holiday, and the 2nd half of December (peak + holiday season)
    c["is_bridge"] = (c.days_to_hol == 1) | (c.days_from_hol == 1)
    c["late_december"] = (idx.month == 12) & (idx.day >= 15)
    c["covid_2020"] = (idx >= "2020-03-15") & (idx <= "2020-06-30")
    c["holiday_on_weekday"] = c.is_holiday.astype(bool) & (idx.day_of_week < 5)
    return c.astype(float)


LOGISTICS_CAL = ["black_friday", "cyber_monday", "thanksgiving_week", "xmas_eve_nye", "peak_season",
                 "quarter_end", "jewish_holiday", "jewish_holiday_eve"]


def logistics_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Dates that move freight but are not public holidays."""
    years = range(idx.year.min(), idx.year.max() + 2)
    c = pd.DataFrame(index=idx)
    # Thanksgiving = 4th Thursday of November
    tg = pd.DatetimeIndex([pd.Timestamp(y, 11, 1) + pd.offsets.WeekOfMonth(week=3, weekday=3) for y in years])
    tg = tg.where(tg.month == 11, tg - pd.Timedelta(weeks=1))
    c["black_friday"] = idx.isin(tg + pd.Timedelta(days=1))
    c["cyber_monday"] = idx.isin(tg + pd.Timedelta(days=4))
    c["thanksgiving_week"] = idx.isin(tg - pd.Timedelta(days=3)) | idx.isin(tg - pd.Timedelta(days=2)) | \
        idx.isin(tg - pd.Timedelta(days=1))
    c["xmas_eve_nye"] = ((idx.month == 12) & idx.day.isin([24, 31]))
    c["peak_season"] = ((idx.month == 11) & (idx.day >= 15)) | ((idx.month == 12) & (idx.day <= 23))
    # last 3 business days of a quarter: shippers push volume out before the books close
    bq = pd.bdate_range(idx.min(), idx.max() + pd.Timedelta(days=100))
    qe = pd.Series(bq, index=bq).groupby(bq.to_period("Q")).apply(lambda s: s.iloc[-3:]).to_numpy()
    c["quarter_end"] = idx.isin(qe)
    # Jewish holidays matter for Brooklyn freight (Israel's public-holiday calendar = the main ones)
    il_cal = holidays.country_holidays("IL", years=years, language="en_US")
    il = pd.to_datetime([d for d, name in il_cal.items() if "Independence" not in name])
    c["jewish_holiday"] = idx.isin(il)
    c["jewish_holiday_eve"] = idx.isin(il - pd.Timedelta(days=1))
    return c.astype(float)


MORNING_COLS = ["morning_trucks", "morning_combo", "morning_cars", "morning_hours"]
OBS_COLS = ["trucks", "combo", "cars", "tons", "hours_obs", "night_trucks", "lanes_alive", "lanes_configured",
            *MORNING_COLS]
MTA_FACILITIES = {"carey": "Hugh L. Carey Tunnel", "vz": "Verrazzano - Narrows Bridge"}


def mta_daily(m: pd.DataFrame) -> pd.DataFrame:
    """MTA hourly crossings -> daily truck counts (all facilities, the two next to the BQE, morning)."""
    t = m[m.cat == "Truck"]
    day = t.ts.dt.floor("D")
    out = pd.DataFrame({"mta_trucks": t.groupby(day).n.sum()})
    for k, name in MTA_FACILITIES.items():
        sel = t.facility == name
        out[f"mta_trucks_{k}"] = t[sel].groupby(day[sel]).n.sum()
    mor = t.ts.dt.hour < MORNING_HOURS
    out["mta_morning_trucks"] = t[mor].groupby(day[mor]).n.sum()
    out.index.name = "date"
    return out


def weekday_ratio(x: pd.Series, holiday: pd.Series) -> pd.Series:
    """x / (median of the same weekday over the previous 4 non-holiday weeks)."""
    ok = (x > 0) & (holiday == 0)
    const = pd.Series(1, index=x.index)
    return x / schedule(x, ok, const)
SAME_DAY_EXOG = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "snowfall_sum", "wind_speed_10m_max"]


def observe(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw WIM groups -> (hourly series, daily observations). No look-ahead logic here."""
    h = hourly(raw)
    return h, daily(h, lane_health(raw))


def make_table(obs: pd.DataFrame, weather: pd.DataFrame, diesel: pd.Series,
               mta: pd.DataFrame | None = None, nowcast: bool = False) -> pd.DataFrame:
    """Daily observations + exogenous data -> target and features.

    Contract (checked by tests/test_leakage.py): every feature for day D is a
    function of observations up to D-1 only. Exceptions: weather for day D
    (SAME_DAY_EXOG, in production the D-1 forecast) and, with nowcast=True,
    the first MORNING_HOURS hours of day D itself (forecast issued at 07:00).
    """
    d = obs.copy()
    d["ok"] = sensor_ok(d)
    cal = calendar(d.index)
    # lane configuration as of yesterday (closures are known to dispatch in advance)
    cfg = d.lanes_configured.shift(1)
    # planners know public holidays: keep them out of the schedule window
    d["schedule"] = schedule(d.trucks, d.ok & (cal.is_holiday == 0), cfg)
    d["ratio"] = d.trucks / d.schedule
    d["label"] = pd.cut(d.ratio, [-np.inf, 1 - BAND, 1 + BAND, np.inf], labels=LABELS)
    d["target_ok"] = d.ok & d.schedule.notna()

    # ---- features known at end of D-1 (everything is shifted by >= 1 day)
    f = cal.join(logistics_calendar(d.index))
    r_ok = d.ratio.where(d.ok)
    for k in (1, 2, 3, 7, 14):
        f[f"ratio_lag{k}"] = r_ok.shift(k)
    f["ratio_roll7"] = r_ok.shift(1).rolling(7, min_periods=4).mean()
    f["ratio_roll28"] = r_ok.shift(1).rolling(28, min_periods=14).mean()
    f["ratio_std28"] = r_ok.shift(1).rolling(28, min_periods=14).std()
    t_ok = d.trucks.where(d.ok)
    f["trucks_lag1"] = t_ok.shift(1)
    f["trucks_lag7"] = t_ok.shift(7)
    f["trend_7_28"] = t_ok.shift(1).rolling(7, min_periods=4).mean() / t_ok.shift(1).rolling(28, min_periods=14).mean()
    f["combo_share_lag1"] = (d.combo / d.trucks).where(d.ok).shift(1)
    f["tons_per_truck_lag1"] = (d.tons / d.trucks).where(d.ok).shift(1)
    f["night_share_lag1"] = (d.night_trucks / d.trucks).where(d.ok).shift(1)
    f["sched_vs_roll28"] = d.schedule / t_ok.shift(1).rolling(28, min_periods=14).mean()
    f["valid_days_last7"] = d.ok.astype(float).shift(1).rolling(7).sum()

    # weather for day D: in production use the D-1 forecast; here archive values stand in
    f = f.join(weather[SAME_DAY_EXOG])
    f["snow_lag1"] = weather.snowfall_sum.reindex(f.index).shift(1)

    # MTA bridges & tunnels: yesterday's truck traffic vs. its own weekday norm
    if mta is not None:
        mt = mta.reindex(d.index)
        hol = cal.is_holiday
        for col, name in [("mta_trucks", "mta"), ("mta_trucks_carey", "mta_carey"), ("mta_trucks_vz", "mta_vz")]:
            r = weekday_ratio(mt[col], hol)
            f[f"{name}_ratio_lag1"] = r.shift(1)
            if name == "mta":
                f["mta_ratio_roll7"] = r.shift(1).rolling(7, min_periods=4).mean()
                f["mta_ratio_lag7"] = r.shift(7)

    # nowcast: the morning of day D is already observed at 07:00
    if nowcast:
        mo = d.morning_hours == MORNING_HOURS
        mcar_ref = d.morning_cars.where(mo).rolling(28, min_periods=7).median().shift(1)
        mo &= (d.morning_cars / mcar_ref).between(0.5, 1.6)
        m_sched = schedule(d.morning_trucks, mo & (cal.is_holiday == 0), cfg)
        f["morning_ratio"] = (d.morning_trucks / m_sched).where(mo)
        f["morning_combo_share"] = (d.morning_combo / d.morning_trucks).where(mo)
        f["morning_share_of_sched"] = (d.morning_trucks / d.schedule).where(mo)
        if mta is not None:
            f["mta_morning_ratio"] = weekday_ratio(mta.mta_morning_trucks.reindex(d.index), cal.is_holiday)

    # diesel: weekly, published with a lag -> use the value at least 7 days old
    dsl = diesel.reindex(pd.date_range(diesel.index.min(), f.index.max())).ffill().shift(7)
    f["diesel"] = dsl.reindex(f.index)
    f["diesel_chg_8w"] = f.diesel / dsl.shift(56).reindex(f.index) - 1
    return d.join(f)


def load_exog() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    w = pd.read_parquet(RAW / "weather_nyc_daily.parquet").set_index("date")
    dsl = pd.read_parquet(RAW / "diesel_weekly.parquet").set_index("date").GASDESW
    mta = mta_daily(pd.read_parquet(RAW / "mta_hourly.parquet"))
    return w, dsl, mta


def load_weather_forecast() -> pd.DataFrame:
    """Weather forecast issued at D-1 for day D (columns as SAME_DAY_EXOG)."""
    return pd.read_parquet(RAW / "weather_forecast_d1.parquet").set_index("date")[SAME_DAY_EXOG]


def build() -> pd.DataFrame:
    raw = pd.read_parquet(RAW / "wim_QB_hour_lane_class.parquet")
    h, obs = observe(raw)
    PROC.mkdir(parents=True, exist_ok=True)
    h.to_parquet(PROC / "hourly.parquet")
    obs.to_parquet(PROC / "daily_obs.parquet")
    w, dsl, mta = load_exog()
    out = make_table(obs, w, dsl, mta)
    out.to_parquet(PROC / "daily.parquet")
    make_table(obs, w, dsl, mta, nowcast=True).to_parquet(PROC / "daily_nowcast.parquet")
    return out


FEATURES_EXCLUDE = {*OBS_COLS, "ok", "ratio", "label", "target_ok",
                    # dropped after ablation: diesel acted as a time proxy and hurt MAE (-1.7% without it)
                    "diesel", "diesel_chg_8w",
                    # dropped: sensor measurement drift, not demand (tons/truck 22.6 in 2019 -> 12.9 in 2026,
                    # tractor-trailer share 45% -> 27% as lanes were re-configured); -11% MAE in nowcast without them
                    "combo_share_lag1", "tons_per_truck_lag1", "night_share_lag1", "morning_combo_share"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE and not c.endswith("_f")]


if __name__ == "__main__":
    df = build()
    m = df.target_ok
    print(f"days: {len(df)}, valid targets: {m.sum()} ({m.mean():.0%})")
    print(df.loc[m, "label"].value_counts(normalize=True).round(3))
    print(df.loc[m, "ratio"].describe().round(3))
