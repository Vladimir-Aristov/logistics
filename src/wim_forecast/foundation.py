"""Time-series foundation models (Amazon Chronos), zero-shot, on the same backtest days.

Every test day D becomes one series in a batch: daily truck counts up to D-1
(sensor-outage days left as NaN, which Chronos masks), forecast horizon 1 day.
No training or fine-tuning on our data.

Variants
* Chronos-Bolt (base)             univariate
* Chronos-2                       univariate
* Chronos-2 + covariates          past + known-future covariates (calendar, weather,
                                  lane configuration; MTA ratio as a past-only covariate)
* Chronos-2 + covariates (nowcast) + the morning of day D as a known-future covariate

The median is the point forecast; the 0.71 quantile is the dispatch plan
(same newsvendor quantile as the LightGBM planner); 5-95% is a 90% interval.

    python foundation.py [--device mps]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .common import FOLD_STARTS, Q_OPT, REP, folds, load_table, to_label

QL = [0.05, 0.5, round(Q_OPT, 2), 0.95]
CONTEXT = 1024  # days of history per series
FUTURE_COV = ["dow", "is_holiday", "is_bridge", "days_to_hol", "days_from_hol", "late_december", "peak_season",
              "black_friday", "jewish_holiday", "lanes_cfg", "temperature_2m_max", "precipitation_sum", "snowfall_sum"]
PAST_COV = ["mta_ratio"]


def frame(nowcast: bool) -> pd.DataFrame:
    df, _ = load_table("daily_nowcast" if nowcast else "daily")
    x = pd.DataFrame(index=df.index)
    x["y"] = df.trucks.where(df.ok)
    for c in FUTURE_COV:
        if c in df:
            x[c] = df[c]
    x["lanes_cfg"] = df.lanes_configured.shift(1)
    # MTA truck ratio on day D itself is only known after D -> past covariate (context only)
    x["mta_ratio"] = df.mta_ratio_lag1.shift(-1)
    if nowcast:
        x["morning"] = df.morning_trucks.where(df.morning_hours == 7)
    return x, df


def build_items(x: pd.DataFrame, test_idx: np.ndarray, cols: list[str], future_cols: list[str]):
    ctx_parts, fut_parts = [], []
    for i in test_idx:
        lo = max(0, i - CONTEXT)
        c = x.iloc[lo:i][["y", *cols]].copy()
        c["id"], c["ts"] = str(x.index[i].date()), x.index[lo:i]
        ctx_parts.append(c)
        if future_cols:
            f = x.iloc[[i]][future_cols].copy()
            f["id"], f["ts"] = str(x.index[i].date()), x.index[i]
            fut_parts.append(f)
    ctx = pd.concat(ctx_parts, ignore_index=True)
    fut = pd.concat(fut_parts, ignore_index=True) if fut_parts else None
    return ctx, fut


def predict_chronos2(pipe, x, test_idx, cols, future_cols):
    ctx, fut = build_items(x, test_idx, cols, future_cols)
    out = pipe.predict_df(ctx, future_df=fut, id_column="id", timestamp_column="ts", target="y",
                          prediction_length=1, quantile_levels=QL, batch_size=64)
    out = out.set_index("id").loc[[str(x.index[i].date()) for i in test_idx]]
    return {q: out[str(q)].to_numpy() if str(q) in out else out[q].to_numpy() for q in QL}


def predict_bolt(pipe, x, test_idx):
    ctxs = [torch.tensor(x.y.iloc[max(0, i - CONTEXT):i].to_numpy(), dtype=torch.float32) for i in test_idx]
    qs = []
    for b in range(0, len(ctxs), 128):
        q, _ = pipe.predict_quantiles(ctxs[b:b + 128], prediction_length=1, quantile_levels=QL)
        qs.append(q[:, 0, :].numpy())
    q = np.concatenate(qs)
    return {ql: q[:, j] for j, ql in enumerate(QL)}


def to_rows(df, test_idx, fold_of, setup, model, q):
    sched = df.schedule.to_numpy()[test_idx]
    r = q[0.5] / sched
    return pd.DataFrame({"date": df.index[test_idx], "fold": fold_of, "setup": setup, "model": model,
                         "ratio": r, "label": to_label(r), "plan": q[QL[2]],
                         "q05": q[0.05], "q95": q[0.95]})


def main(device: str):
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    t0 = time.time()
    c2 = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    bolt = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map=device)
    out = []
    for nowcast in (False, True):
        x, df = frame(nowcast)
        fl = list(folds(df))
        test_idx = np.concatenate([np.where(t)[0] for _, _, _, t in fl])
        fold_of = np.concatenate([[k] * int(t.sum()) for k, _, _, t in fl])
        setup = "Nowcast 07:00" if nowcast else "D-1"
        fut_cols = [c for c in FUTURE_COV if c in x] + (["morning"] if nowcast else [])
        if not nowcast:
            q_bolt = predict_bolt(bolt, x, test_idx)
            out.append(to_rows(df, test_idx, fold_of, setup, "Chronos-Bolt (zero-shot)", q_bolt))
            print(f"  bolt done {time.time() - t0:.0f}s", flush=True)
            out.append(to_rows(df, test_idx, fold_of, setup, "Chronos-2 (zero-shot)",
                               predict_chronos2(c2, x, test_idx, [], [])))
            print(f"  chronos-2 univariate done {time.time() - t0:.0f}s", flush=True)
        q = predict_chronos2(c2, x, test_idx, fut_cols + PAST_COV, fut_cols)
        out.append(to_rows(df, test_idx, fold_of, setup, "Chronos-2 + covariates", q))
        print(f"  chronos-2 covariates [{setup}] done {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(out)
    res.to_parquet(REP / "backtest" / "chronos.parquet", index=False)
    print(f"saved {len(res)} rows, folds from {FOLD_STARTS[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args().device)
