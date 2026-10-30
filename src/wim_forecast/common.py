"""Shared pieces: data loading, CV folds, LightGBM models, metrics, business cost."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .build_dataset import BAND, PROC, ROOT, feature_cols

REP = ROOT / "reports"
SEED = 42

# Asymmetric dispatch cost: a missing truck is hired on the spot market at 2.5x,
# an idle scheduled truck costs 1x. Optimal plan = 2.5/3.5 ~ 71% quantile.
COST_SHORT, COST_IDLE = 2.5, 1.0
Q_OPT = COST_SHORT / (COST_SHORT + COST_IDLE)

# Rolling-origin backtest: expanding train window, 6-month test folds.
FOLD_STARTS = ["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]
ES_DAYS = 120  # last valid days before a fold are used for early stopping


def load_table(table: str = "daily") -> tuple[pd.DataFrame, list[str]]:
    """table: 'daily' (forecast at end of D-1) or 'daily_nowcast' (forecast at 07:00 on D)."""
    df = pd.read_parquet(PROC / f"{table}.parquet")
    return df, feature_cols(df)


def folds(df: pd.DataFrame):
    """Yield (fold_id, fit_mask, es_mask, test_mask) over valid-target days."""
    ok = df.target_ok.to_numpy()
    ends = FOLD_STARTS[1:] + [str(df.index.max() + pd.Timedelta(days=1))]
    for i, (start, end) in enumerate(zip(FOLD_STARTS, ends)):
        before = ok & (df.index < start)
        es_from = df.index[before][-ES_DAYS]
        es = before & (df.index >= es_from)
        yield i, before & ~es, es, ok & (df.index >= start) & (df.index < end)


def to_label(ratio) -> np.ndarray:
    ratio = np.asarray(ratio)
    return np.where(ratio < 1 - BAND, 0, np.where(ratio > 1 + BAND, 2, 1))


# ---------------------------------------------------------------- LightGBM
LGB_BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_estimators=3000, random_state=SEED, verbose=-1)


def fit_lgb(X, y_log, y_lab, fit, es, params: dict | None = None) -> dict:
    """Fit regressor (huber on log-ratio), quantile planner and balanced classifier."""
    import lightgbm as lgb  # lazy: keeps lightgbm's OpenMP out of the PyTorch process
    p = {**LGB_BASE, **(params or {})}
    cb = [lgb.early_stopping(100, verbose=False)]
    reg = lgb.LGBMRegressor(objective="huber", alpha=0.1, **p)
    reg.fit(X[fit], y_log[fit], eval_set=[(X[es], y_log[es])], callbacks=cb)
    q = lgb.LGBMRegressor(objective="quantile", alpha=Q_OPT, **p)
    q.fit(X[fit], y_log[fit], eval_set=[(X[es], y_log[es])], eval_metric="quantile", callbacks=cb)
    clf = lgb.LGBMClassifier(objective="multiclass", class_weight="balanced", **p)
    clf.fit(X[fit], y_lab[fit], eval_set=[(X[es], y_lab[es])], callbacks=cb)
    return {"reg": reg, "q": q, "clf": clf}


# ---------------------------------------------------------------- metrics
def dispatch_cost(actual, plan) -> float:
    short = np.clip(actual - plan, 0, None)
    idle = np.clip(plan - actual, 0, None)
    return float((COST_SHORT * short + COST_IDLE * idle).sum())


def score(y_ratio, sched, p_ratio, p_label, plan=None) -> dict:
    """All metrics for one model on one set of days."""
    y_lab = to_label(y_ratio)
    actual = y_ratio * sched
    plan = p_ratio * sched if plan is None else plan
    rec = recall_score(y_lab, p_label, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_lab, p_label, average="macro", labels=[0, 1, 2], zero_division=0),
        "recall_below": rec[0], "recall_above": rec[2],
        "mae_trucks": float(np.mean(np.abs(actual - p_ratio * sched))),
        "dispatch_cost": dispatch_cost(actual, plan),
    }
