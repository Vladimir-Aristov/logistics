"""Prediction intervals with coverage guarantees for the LightGBM forecasts.

For each backtest fold the last ES_DAYS valid days before the fold (also used for
early stopping) are the calibration set. Target coverage 90%.

* split   — split conformal on |log-ratio residual| of the point model
* cqr     — conformalized quantile regression: LightGBM 5% / 95% quantiles,
            widened/narrowed by the calibrated score (Romano et al., 2019)
* cqr+aci — CQR with adaptive conformal inference (Gibbs & Candès, 2021): the
            miscoverage level is updated daily from realised coverage, and new
            scores join the calibration set once the day is observed

    python conformal.py [--table daily_nowcast]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import LGB_BASE, REP, folds, load_table

ALPHA = 0.10
GAMMA = 0.01  # ACI step size


def conf_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, max(0.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return float(np.quantile(scores, level, method="higher"))


def fit_q(X, y, fit, es, alpha):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_BASE)
    m.fit(X[fit], y[fit], eval_set=[(X[es], y[es])], eval_metric="quantile",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(X)


def run(table: str) -> pd.DataFrame:
    df, cols = load_table(table)
    X, y = df[cols], np.log(df.ratio).to_numpy()
    sched = df.schedule.to_numpy()
    out = []
    for k, fit, es, test in folds(df):
        mid = fit_q(X, y, fit, es, 0.5)
        lo, hi = fit_q(X, y, fit, es, ALPHA / 2), fit_q(X, y, fit, es, 1 - ALPHA / 2)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        cal, idx = np.where(es)[0], np.where(test)[0]

        # split conformal
        q_abs = conf_q(np.abs(y[cal] - mid[cal]), ALPHA)
        # CQR
        s_cal = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q_cqr = conf_q(s_cal, ALPHA)
        # CQR + ACI, walking through the fold day by day
        a_t, scores = ALPHA, list(s_cal)
        aci_lo, aci_hi = np.empty(len(idx)), np.empty(len(idx))
        for j, i in enumerate(idx):
            q = conf_q(np.asarray(scores), a_t) if 0 < a_t < 1 else (np.inf if a_t <= 0 else 0.0)
            aci_lo[j], aci_hi[j] = lo[i] - q, hi[i] + q
            s_i = max(lo[i] - y[i], y[i] - hi[i])
            err = float(s_i > q)
            a_t += GAMMA * (ALPHA - err)
            scores.append(s_i)

        for name, L, H in [("split", mid[idx] - q_abs, mid[idx] + q_abs),
                           ("cqr", lo[idx] - q_cqr, hi[idx] + q_cqr),
                           ("cqr+aci", aci_lo, aci_hi),
                           ("raw quantiles (no conformal)", lo[idx], hi[idx])]:
            out.append(pd.DataFrame({"date": df.index[idx], "fold": k, "method": name,
                                     "lo": np.exp(L) * sched[idx], "hi": np.exp(H) * sched[idx],
                                     "y": df.trucks.to_numpy()[idx]}))
        print(f"  {table} fold {k}: q_abs={q_abs:.3f} q_cqr={q_cqr:+.3f}", flush=True)
    return pd.concat(out)


def summarize(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(cover=(p.y >= p.lo) & (p.y <= p.hi), width=p.hi - p.lo,
                 # Winkler / interval score: width + 2/alpha * miss distance
                 wink=(p.hi - p.lo) + 2 / ALPHA * ((p.lo - p.y).clip(lower=0) + (p.y - p.hi).clip(lower=0)))
    per_fold = p.groupby(["method", "fold"]).cover.mean().unstack()
    s = p.groupby("method").agg(coverage=("cover", "mean"), width=("width", "mean"), winkler=("wink", "mean"))
    s["worst_fold_coverage"] = per_fold.min(axis=1)
    s["best_fold_coverage"] = per_fold.max(axis=1)
    return s.sort_values("winkler")


def plot(p: pd.DataFrame, table: str):
    g = p[(p.method == "cqr+aci") & (p.date >= "2026-03-01")].set_index("date").sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.vlines(g.index, g.lo, g.hi, color="#1f77b4", alpha=.35, lw=3.2, label="90% interval (CQR + ACI)")
    miss = (g.y < g.lo) | (g.y > g.hi)
    ax.plot(g.index, g.y, ".", color="#222", ms=4, label="actual trucks")
    ax.plot(g.index[miss], g.y[miss], "o", mfc="none", color="#d62728", ms=7, label="outside interval")
    ax.set_title(f"Prediction intervals, {table}, Mar–Sep 2026 ({miss.mean():.0%} outside)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(REP / "figures" / f"intervals_{table}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="daily", choices=["daily", "daily_nowcast"])
    a = ap.parse_args()
    p = run(a.table)
    out_dir = REP / "backtest" / "intervals"
    out_dir.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out_dir / f"intervals_{a.table}.parquet", index=False)
    s = summarize(p)
    s.round(4).to_csv(REP / f"intervals_{a.table}_summary.csv")
    print(s.round(3).to_string())
    plot(p, a.table)
