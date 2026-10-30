"""Exogenous data.

* NYC daily weather: observed (Open-Meteo archive) and the forecast issued the day
  before (Open-Meteo previous-runs API; precipitation available from mid-2024).
* MTA Bridges & Tunnels hourly crossings (trucks and cars per facility).
* US weekly diesel price (FRED) - kept for the ablation record, not used as a feature.
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
START = "2019-09-01"
END = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def weather() -> pd.DataFrame:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 40.69, "longitude": -73.99,  # BQE, Brooklyn Heights
        "start_date": START, "end_date": END, "timezone": "America/New_York",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
    }, timeout=120)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    d["date"] = pd.to_datetime(d["date"])
    return d


def fred(series: str) -> pd.DataFrame:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}, timeout=120)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["date", series]
    d["date"] = pd.to_datetime(d["date"])
    d[series] = pd.to_numeric(d[series], errors="coerce")
    return d.dropna()


WEATHER_VARS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]


def weather_forecast_d1() -> pd.DataFrame:
    """Daily aggregates of the forecast issued one day ahead (what dispatch sees at D-1)."""
    r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
        "latitude": 40.69, "longitude": -73.99, "start_date": "2021-01-01", "end_date": END,
        "timezone": "America/New_York",
        "hourly": ",".join(f"{v}_previous_day1" for v in WEATHER_VARS),
    }, timeout=300)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["date"] = pd.to_datetime(h.time).dt.floor("D")
    g = h.groupby("date")
    d = pd.DataFrame({
        "temperature_2m_max": g.temperature_2m_previous_day1.max(),
        "temperature_2m_min": g.temperature_2m_previous_day1.min(),
        "precipitation_sum": g.precipitation_previous_day1.sum(min_count=24),
        "snowfall_sum": g.snowfall_previous_day1.sum(min_count=24),
        "wind_speed_10m_max": g.wind_speed_10m_previous_day1.max(),
    })
    return d.reset_index()


MTA = "https://data.ny.gov/resource/ebfx-2m7v.json"


def _mta_year(year: int) -> pd.DataFrame:
    cache = RAW_DIR / "mta" / f"mta_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    params = {
        "$select": "date,hour,facility,vehicle_class_category,sum(traffic_count) as n",
        "$where": f"date between '{year}-01-01' and '{year}-12-31T23:59:59' "
                  "and vehicle_class_category in ('Truck','Car')",
        "$group": "date,hour,facility,vehicle_class_category",
        "$limit": 1_000_000,
    }
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.get(MTA, params=params, timeout=(30, 900))
            r.raise_for_status()
            break
        except Exception as e:
            print(f"  retry MTA {year}: {type(e).__name__}", flush=True)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    d = pd.DataFrame(r.json())
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache, index=False)
    print(f"  MTA {year}: {len(d):,} rows in {time.time() - t0:.0f}s", flush=True)
    return d


def mta() -> pd.DataFrame:
    years = range(2019, pd.Timestamp.today().year + 1)
    with ThreadPoolExecutor(len(years)) as ex:
        d = pd.concat(list(ex.map(_mta_year, years)), ignore_index=True)
    d["ts"] = pd.to_datetime(d.date) + pd.to_timedelta(pd.to_numeric(d.hour), unit="h")
    d["n"] = pd.to_numeric(d.n)
    return d[["ts", "facility", "vehicle_class_category", "n"]].rename(columns={"vehicle_class_category": "cat"})


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    w = weather(); w.to_parquet(RAW_DIR / "weather_nyc_daily.parquet", index=False)
    print("weather", w.shape, w.date.min().date(), w.date.max().date())
    dsl = fred("GASDESW"); dsl.to_parquet(RAW_DIR / "diesel_weekly.parquet", index=False)
    print("diesel", dsl.shape, dsl.date.min().date(), dsl.date.max().date())
    wf = weather_forecast_d1(); wf.to_parquet(RAW_DIR / "weather_forecast_d1.parquet", index=False)
    print("weather forecast D-1", wf.shape, wf.dropna().date.min().date(), wf.date.max().date())
    m = mta(); m.to_parquet(RAW_DIR / "mta_hourly.parquet", index=False)
    print("MTA", m.shape, m.ts.min(), m.ts.max())


if __name__ == "__main__":
    main()
