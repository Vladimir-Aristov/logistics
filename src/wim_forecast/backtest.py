"""Rolling-origin backtest for baselines and LightGBM (+ feature-group ablations).

Writes out-of-fold daily predictions to reports/backtest/<name>.parquet in long
format: date, fold, setup, model, ratio, label, plan. Evaluation lives in evaluate.py.

    python backtest.py --baselines                                   # schedule + naive
    python backtest.py --setup "D-1"                                 # LightGBM, all D-1 features
    python backtest.py --setup "D-1, phase-1 features" --drop logistics_cal mta
    python backtest.py --setup "D-1, weather forecast" --weather-forecast
    python backtest.py --setup "Nowcast 07:00" --table daily_nowcast
    python backtest.py --ablation [--table daily_nowcast]             # drop one group at a time
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .backtest_groups import FEATURE_GROUPS, slug
from .build_dataset import SAME_DAY_EXOG, load_weather_forecast
from .common import LGB_BASE, REP, fit_lgb, folds, load_table, to_label

OUT = REP / "backtest"

def rows(df, idx, fold, model, ratio, label, plan=None, setup="baseline"):
    sched = df.schedule.to_numpy()[idx]
    return pd.DataFrame({"date": df.index[idx], "fold": fold, "setup": setup, "model": model, "ratio": ratio[idx],
                         "label": label[idx], "plan": (ratio[idx] * sched) if plan is None else plan[idx]})


def run_lgb(df, cols, tag="LightGBM", setup="D-1", weather_fc: pd.DataFrame | None = None):
    """weather_fc: if given, test rows get the D-1 weather forecast instead of observed weather
    (the model is still trained on observed weather), and only days with a forecast are kept."""
    X = df[cols]
    X_pred, has_fc = X, np.ones(len(df), bool)
    if weather_fc is not None:
        fc = weather_fc.reindex(df.index)
        has_fc = fc.notna().all(axis=1).to_numpy()
        X_pred = X.copy()
        wc = [c for c in SAME_DAY_EXOG if c in cols]
        X_pred.loc[has_fc, wc] = fc.loc[has_fc, wc].to_numpy()
    y_log = np.log(df.ratio)
    y_lab = df.label.cat.codes
    out = []
    for k, fit, es, test in folds(df):
        idx = np.where(test & has_fc)[0]
        if len(idx) == 0:
            continue
        m = fit_lgb(X, y_log, y_lab, fit, es)
        r = np.exp(m["reg"].predict(X_pred))
        rq = np.exp(m["q"].predict(X_pred))
        out += [rows(df, idx, k, f"{tag} regressor", r, to_label(r), setup=setup),
                rows(df, idx, k, f"{tag} classifier", r, m["clf"].predict(X_pred), setup=setup),
                rows(df, idx, k, f"{tag} quantile plan", r, to_label(r), rq * df.schedule.to_numpy(), setup=setup)]
        print(f"  [{setup}] {tag} fold {k}: {len(idx)} days, iters reg={m['reg'].best_iteration_}", flush=True)
    return pd.concat(out)


def run_baselines(df):
    one = np.ones(len(df))
    lag7 = df.ratio_lag7.fillna(1).to_numpy()
    # simple nowcast rule: scale the whole day's schedule by how the morning compares to a usual morning
    now, _ = load_table("daily_nowcast")
    mr = now.morning_ratio.reindex(df.index).fillna(1).to_numpy()
    out = []
    for k, _, _, test in folds(df):
        idx = np.where(test)[0]
        out += [rows(df, idx, k, "Schedule", one, to_label(one)),
                rows(df, idx, k, "Naive: deviation 7d ago", lag7, to_label(lag7)),
                rows(df, idx, k, "Schedule × morning ratio", mr, to_label(mr), setup="Nowcast 07:00")]
    return pd.concat(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--setup", default="D-1")
    ap.add_argument("--table", default="daily", choices=["daily", "daily_nowcast"])
    ap.add_argument("--drop", nargs="*", default=[], choices=list(FEATURE_GROUPS))
    ap.add_argument("--weather-forecast", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="LightGBM seed (robustness checks)")
    a = ap.parse_args()
    if a.seed is not None:
        LGB_BASE["random_state"] = a.seed
    OUT.mkdir(parents=True, exist_ok=True)
    df, cols = load_table(a.table)
    cols = [c for c in cols if not any(c in FEATURE_GROUPS[g] for g in a.drop)]
    if a.baselines:
        run_baselines(df).to_parquet(OUT / "baselines.parquet", index=False)
    elif a.ablation:
        groups = [g for g in FEATURE_GROUPS if any(c in cols for c in FEATURE_GROUPS[g])]
        parts = [run_lgb(df, cols, "full", setup=a.table)]
        for g in groups:
            parts.append(run_lgb(df, [c for c in cols if c not in FEATURE_GROUPS[g]], f"no_{g}", setup=a.table))
        pd.concat(parts).to_parquet(OUT / "ablations" / f"ablation_{a.table}.parquet", index=False)
    else:
        wf = load_weather_forecast() if a.weather_forecast else None
        run_lgb(df, cols, setup=a.setup, weather_fc=wf).to_parquet(OUT / f"lgb_{slug(a.setup)}.parquet", index=False)
    print("done")
