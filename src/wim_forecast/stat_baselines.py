"""Classical time-series baselines (statsforecast) on the same backtest days.

The models forecast the daily truck count one day ahead with rolling origin
(refit every 30 days, state updated daily). Sensor-outage days are imputed
with the same weekday's previous valid value so the series is continuous;
predictions are scored only on valid days, like every other model.

    python stat_baselines.py                 # all models, one process each (in parallel)
    python stat_baselines.py --model ets     # a single model
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoETS

from .common import FOLD_STARTS, REP, folds, load_table, to_label

EXOG = ["is_holiday", "is_bridge", "late_december"]


def series(df: pd.DataFrame) -> pd.DataFrame:
    y = df.trucks.where(df.ok)
    # fill gaps with the same weekday's last valid value (then any leftovers by ffill)
    y = y.groupby(df.index.day_of_week).ffill().ffill().bfill()
    s = pd.DataFrame({"unique_id": "bqe", "ds": df.index, "y": np.log(y.to_numpy())})
    return pd.concat([s, df[EXOG].reset_index(drop=True)], axis=1)


MODELS = {
    "ets": lambda: AutoETS(season_length=7, alias="ETS (weekly)"),
    "mstl": lambda: MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL (weekly+yearly)"),
    "sarimax": lambda: AutoARIMA(season_length=7, alias="SARIMAX + holidays"),
}


def run_one(key: str):
    df, _ = load_table()
    s = series(df)
    if key != "sarimax":          # only ARIMA uses the holiday regressors
        s = s.drop(columns=EXOG)
    model = MODELS[key]()
    n_windows = int((df.index >= FOLD_STARTS[0]).sum())
    sf = StatsForecast(models=[model], freq="D", n_jobs=1)
    cv = sf.cross_validation(df=s, h=1, step_size=1, n_windows=n_windows, refit=30).set_index("ds")
    parts = []
    for k, _, _, test in folds(df):
        days = df.index[test]
        trucks = np.exp(cv.loc[days, model.alias].to_numpy())
        r = trucks / df.schedule.loc[days].to_numpy()
        parts.append(pd.DataFrame({"date": days, "fold": k, "model": model.alias, "ratio": r,
                                   "label": to_label(r), "plan": trucks}))
    (REP / "backtest").mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_parquet(REP / "backtest" / f"stat_{key}.parquet", index=False)
    print(f"{model.alias}: done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    a = ap.parse_args()
    if a.model:
        run_one(a.model)
    else:
        procs = [subprocess.Popen([sys.executable, "-m", "wim_forecast.stat_baselines", "--model", k]) for k in MODELS]
        sys.exit(max(p.wait() for p in procs))
