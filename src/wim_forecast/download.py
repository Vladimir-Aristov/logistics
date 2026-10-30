"""Download NYC Weigh-in-Motion data (BQE, Queens-bound) aggregated server-side.

Each WIM record is one vehicle (~100M rows in total). Socrata has no index on
the date columns, so every query is a full scan that costs ~2-3 min no matter
how much it returns. We therefore ask for a whole year per query, aggregated to
(hour, lane, FHWA class), and run all years in parallel: ~8 queries, ~5 min.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.cityofnewyork.us/resource"
# dataset id -> (label, years covered)
DATASETS = {
    "n5c8-x7x2": ("historical", range(2019, 2025)),
    "4fwc-j3vn": ("from_2025", range(2025, pd.Timestamp.today().year + 1)),
}
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE = RAW_DIR / "years"
COLS = ["yearmonth", "day", "hour", "lane", "class", "n", "gvw_lb"]


def fetch_year(ds: str, year: int, direction: str) -> pd.DataFrame:
    cache = CACHE / f"{ds}_{direction}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    params = {
        "$select": "yearmonth,day,hour,lane,class,count(*) as n,sum(gvw) as gvw_lb",
        "$where": f"year='{year % 100:02d}' AND direction='{direction}'",
        "$group": "yearmonth,day,hour,lane,class",
        "$limit": 1_000_000,
    }
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.get(f"{BASE}/{ds}.json", params=params, timeout=(30, 900))
            r.raise_for_status()
            df = pd.DataFrame(r.json(), columns=COLS)
            break
        except Exception as e:
            print(f"  retry {ds} {year}: {type(e).__name__} after {time.time() - t0:.0f}s", flush=True)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    df["source"] = DATASETS[ds][0]
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  {ds} {year}: {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)
    return df


def main(direction: str = "QB") -> None:
    jobs = [(ds, y) for ds, (_, years) in DATASETS.items() for y in years]
    print(f"{len(jobs)} dataset-years to fetch (all in parallel)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(len(jobs)) as ex:
        parts = list(ex.map(lambda j: fetch_year(*j, direction), jobs))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for c in ["yearmonth", "day", "hour", "lane", "class", "n", "gvw_lb"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["yearmonth", "day", "hour", "lane", "class"]].isna().any(axis=1)
    print(f"dropping {bad.sum()} groups with missing keys ({df.n[bad].sum():,} vehicles)", flush=True)
    df = df[~bad]
    for c in ["yearmonth", "day", "hour", "lane", "class", "n"]:
        df[c] = df[c].astype("int32")
    df["gvw_lb"] = pd.to_numeric(df["gvw_lb"]).astype("float64")
    df["ts"] = pd.to_datetime(dict(year=df.yearmonth // 100, month=df.yearmonth % 100,
                                   day=df.day, hour=df.hour))
    df = df.drop(columns=["yearmonth", "day"])
    # If a month ever appears in both datasets, keep the newer source.
    df = df.sort_values("source").drop_duplicates(["ts", "lane", "class"], keep="first")
    out = RAW_DIR / f"wim_{direction}_hour_lane_class.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {len(df):,} rows -> {out.name} ({df.ts.min()} .. {df.ts.max()}), "
          f"vehicles={df.n.sum():,}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--direction", default="QB")
    main(p.parse_args().direction)
