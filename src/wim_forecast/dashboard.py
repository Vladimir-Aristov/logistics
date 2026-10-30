"""Render the shadow-monitoring dashboard to docs/index.html (static, GitHub Pages)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dataset import ROOT
from .production import MODE_NAMES, MODES, MONITOR

DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
ROLL = 28
TEMPLATE = Path(__file__).with_name("templates") / "dashboard.html"


def _mode_payload(mode: str) -> dict:
    log = pd.read_csv(MONITOR / f"shadow_{mode}.csv", parse_dates=["date"])
    log = log[log.schedule.notna()].sort_values("date")
    obs = log[log.observed].copy()
    obs["roll_model"] = obs.abs_err.rolling(ROLL, min_periods=14).mean()
    obs["roll_sched"] = obs.abs_err_schedule.rolling(ROLL, min_periods=14).mean()
    drift = pd.read_csv(MONITOR / f"drift_{mode}.csv")

    def col(s, nd=0):
        return [None if pd.isna(v) else round(float(v), nd) for v in s]

    return {
        "name": MODE_NAMES[mode],
        "days": log.date.dt.strftime("%Y-%m-%d").tolist(),
        "actual": col(log.actual.where(log.observed)), "schedule": col(log.schedule), "forecast": col(log.forecast),
        "plan": col(log.plan), "lo": col(log.lo), "hi": col(log.hi),
        "roll_days": obs.date.dt.strftime("%Y-%m-%d").tolist(),
        "roll_model": col(obs.roll_model, 1), "roll_sched": col(obs.roll_sched, 1),
        "drift": [{"feature": r.feature, "share": round(r.importance_share, 3),
                   "psi": None if pd.isna(r.psi) else round(r.psi, 2)} for r in drift.head(8).itertuples()],
    }


def _backtest_rows() -> list[dict]:
    want = [("D-1", "backtest_summary.csv", ["Schedule", "SARIMAX + holidays", "LightGBM quantile plan · D-1",
                                             "Chronos-2 + covariates · D-1", "Ensemble LGB+GRU+Chronos · D-1"]),
            ("Nowcast 07:00", "backtest_summary_nowcast.csv",
             ["Schedule × morning ratio · Nowcast 07:00", "LightGBM quantile plan · Nowcast 07:00",
              "N-HiTS (hourly) · Nowcast 07:00", "Ensemble LGB+GRU+Chronos · Nowcast 07:00"])]
    rows = []
    for setup, f, models in want:
        p = REPORTS / f
        if not p.exists():
            continue
        t = pd.read_csv(p, index_col=0)
        for m in models:
            if m in t.index:
                rows.append({"setup": setup, "model": m.split(" · ")[0], "mae": t.loc[m, "MAE trucks"],
                             "cost": t.loc[m, "Cost, pooled [95% CI]"]})
    return rows


def render() -> str:
    status = json.loads((MONITOR / "status.json").read_text())
    data = {"status": status, "modes": {m: _mode_payload(m) for m in MODES}, "backtest": _backtest_rows()}
    blob = json.dumps(data, allow_nan=False, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    blob = blob.replace("</", "<\\/")
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(TEMPLATE.read_text().replace("__DATA__", blob))
    return f"dashboard -> {out.relative_to(ROOT)}"
