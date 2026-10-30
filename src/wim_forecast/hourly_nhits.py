"""N-HiTS on the HOURLY truck series (neuralforecast), aggregated to the daily forecast.

Daily data gives ~1.4k training points, which starves deep models. Hourly data gives
~60k. For each backtest fold the model is trained on everything before the fold and
then rolled through it day by day (no refit inside a fold):

* D-1      at 00:00 forecast the 24 hours of day D        -> daily = sum
* nowcast  at 07:00 forecast hours 07..23 of day D        -> daily = known morning + sum

Sensor-outage days are masked out of the loss (`available_mask`), not imputed.

    python hourly_nhits.py [--max-steps 1500]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from .build_dataset import MORNING_HOURS, PROC, calendar
from .common import REP, folds, load_table, to_label

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
FUTR = ["hour_sin", "hour_cos", "dow", "is_holiday", "is_bridge", "days_to_hol", "lanes_cfg"]


def hourly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(PROC / "hourly.parquet")
    day = h.index.floor("D")
    ok = daily.ok.reindex(day).fillna(False).to_numpy()
    cal = calendar(daily.index)
    x = pd.DataFrame({"unique_id": "bqe", "ds": h.index,
                      "y": np.where(ok, h.trucks.fillna(0).to_numpy(), 0.0),
                      "available_mask": ok.astype(float)})
    x["hour_sin"] = np.sin(2 * np.pi * h.index.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * h.index.hour / 24)
    for c in ["dow", "is_holiday", "is_bridge", "days_to_hol"]:
        x[c] = cal[c].reindex(day).to_numpy()
    x["lanes_cfg"] = daily.lanes_configured.shift(1).ffill().bfill().reindex(day).to_numpy()
    return x.reset_index(drop=True)


def run_task(x, daily, nowcast: bool, max_steps: int):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS

    h = 24 - MORNING_HOURS if nowcast else 24
    out = []
    for k, _, _, test in folds(daily):
        days = daily.index[test]
        start, end = days.min(), days.max()
        # series ends so that the last window covers day `end` fully
        sub = x[x.ds < end + pd.Timedelta(days=1)]
        n_windows = (end - start).days + 1
        model = NHITS(h=h, input_size=7 * 24, loss=MAE(), max_steps=max_steps, scaler_type="robust",
                      futr_exog_list=FUTR, learning_rate=1e-3, batch_size=64, windows_batch_size=256,
                      random_seed=42, enable_progress_bar=False, enable_model_summary=False, logger=False)
        nf = NeuralForecast(models=[model], freq="h")
        t0 = time.time()
        cv = nf.cross_validation(df=sub, n_windows=n_windows, step_size=24, refit=False)
        hours = set(cv.ds.dt.hour.unique())
        assert hours == set(range(24 - h, 24)), f"forecast windows misaligned: hours {sorted(hours)}"
        cv["day"] = cv.ds.dt.floor("D")
        daily_pred = cv.groupby("day").NHITS.sum()
        if nowcast:
            daily_pred = daily_pred + daily.morning_trucks.reindex(daily_pred.index)
        daily_pred = daily_pred.reindex(days)
        r = (daily_pred / daily.schedule.loc[days]).to_numpy()
        tag = "Nowcast 07:00" if nowcast else "D-1"
        out.append(pd.DataFrame({"date": days, "fold": k, "setup": tag, "model": "N-HiTS (hourly)",
                                 "ratio": r, "label": to_label(r), "plan": daily_pred.to_numpy()}))
        print(f"  [{tag}] fold {k}: {len(days)} days, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out)


def main(max_steps: int):
    daily, _ = load_table("daily_nowcast")
    x = hourly_frame(daily)
    res = pd.concat([run_task(x, daily, False, max_steps), run_task(x, daily, True, max_steps)])
    res.to_parquet(REP / "backtest" / "nhits_hourly.parquet", index=False)
    print("saved", len(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1500)
    main(ap.parse_args().max_steps)
