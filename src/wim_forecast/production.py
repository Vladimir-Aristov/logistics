"""Production model: train once at a cutoff, freeze, score new data in shadow mode.

The WIM portal publishes data monthly, so there is no live feed to forecast from.
Instead the model is frozen at a cutoff date and, every time new data lands, it is
scored on the days it has never seen ("shadow mode"): what it would have predicted
each day vs. what happened, with the same procedure as in the backtest.

Per forecast mode (D-1, nowcast 07:00) the bundle holds LightGBM models for
* the point forecast (huber on log-ratio),
* the dispatch plan (quantile 0.71),
* the 5% / 95% quantiles, conformalized (CQR) and adapted online (ACI).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .build_dataset import ROOT
from .common import ES_DAYS, LGB_BASE, Q_OPT, SEED, load_table, to_label
from .conformal import ALPHA, GAMMA, conf_q

MODELS = ROOT / "models"
MONITOR = ROOT / "monitoring"
MODES = {"d1": "daily", "nowcast": "daily_nowcast"}
MODE_NAMES = {"d1": "D-1 (evening before)", "nowcast": "Nowcast 07:00"}


@dataclass
class Bundle:
    mode: str
    cutoff: str
    features: list[str]
    models: dict
    cal_scores: list[float]
    train_days: int
    data_end: str

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> Bundle:
        return joblib.load(path)


def _fit(objective: str, X, y, fit, es, **kw):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective=objective, **{**LGB_BASE, **kw})
    m.fit(X[fit], y[fit], eval_set=[(X[es], y[es])],
          eval_metric="quantile" if objective == "quantile" else None,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def train(mode: str, cutoff: str) -> Bundle:
    df, cols = load_table(MODES[mode])
    ok = df.target_ok.to_numpy() & (df.index <= cutoff)
    es_from = df.index[ok][-ES_DAYS]
    es = ok & (df.index >= es_from)
    fit = ok & ~es
    X, y = df[cols], np.log(df.ratio).to_numpy()
    models = {
        "point": _fit("huber", X, y, fit, es, alpha=0.1),
        "plan": _fit("quantile", X, y, fit, es, alpha=Q_OPT),
        "lo": _fit("quantile", X, y, fit, es, alpha=ALPHA / 2),
        "hi": _fit("quantile", X, y, fit, es, alpha=1 - ALPHA / 2),
    }
    lo, hi = models["lo"].predict(X[es]), models["hi"].predict(X[es])
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    scores = np.maximum(lo - y[es], y[es] - hi)
    return Bundle(mode=mode, cutoff=cutoff, features=cols, models=models, cal_scores=scores.tolist(),
                  train_days=int(ok.sum()), data_end=str(df.index.max().date()))


def bundle_path(mode: str, cutoff: str) -> Path:
    return MODELS / f"lgbm_{mode}_{cutoff}.joblib"


def latest_bundle(mode: str) -> Bundle:
    paths = sorted(MODELS.glob(f"lgbm_{mode}_*.joblib"))
    if not paths:
        raise FileNotFoundError(f"no trained model for mode {mode!r}; run `wim train`")
    return Bundle.load(paths[-1])


def shadow(b: Bundle) -> pd.DataFrame:
    """Score every day after the cutoff. Interval level adapts online (ACI) on observed days."""
    df, _ = load_table(MODES[b.mode])
    new = df[df.index > b.cutoff]
    X = new[b.features]
    point = np.exp(b.models["point"].predict(X))
    plan = np.exp(b.models["plan"].predict(X))
    lo_raw, hi_raw = b.models["lo"].predict(X), b.models["hi"].predict(X)
    lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)
    sched = new.schedule.to_numpy()
    y = np.log(new.ratio.to_numpy())
    observed = new.target_ok.to_numpy()

    a_t, scores = ALPHA, list(b.cal_scores)
    lo, hi, alpha_used = np.full(len(new), np.nan), np.full(len(new), np.nan), np.full(len(new), np.nan)
    for i in range(len(new)):
        if np.isnan(sched[i]):
            continue
        q = conf_q(np.asarray(scores), a_t) if 0 < a_t < 1 else (np.inf if a_t <= 0 else 0.0)
        lo[i], hi[i], alpha_used[i] = lo_raw[i] - q, hi_raw[i] + q, a_t
        if observed[i]:                       # the day's outcome becomes known -> update ACI
            s = max(lo_raw[i] - y[i], y[i] - hi_raw[i])
            a_t += GAMMA * (ALPHA - float(s > q))
            scores.append(s)

    out = pd.DataFrame({
        "date": new.index, "mode": b.mode, "observed": observed,
        "actual": new.trucks.where(new.target_ok).to_numpy(), "schedule": sched,
        "forecast": point * sched, "plan": plan * sched,
        "lo": np.exp(lo) * sched, "hi": np.exp(hi) * sched, "aci_alpha": alpha_used,
        "label_forecast": to_label(point), "label_actual": np.where(observed, to_label(new.ratio.to_numpy()), -1),
    })
    from .common import COST_IDLE, COST_SHORT
    a = out.actual
    out["abs_err"] = (a - out.forecast).abs()
    out["abs_err_schedule"] = (a - out.schedule).abs()
    out["cost_plan"] = COST_SHORT * (a - out.plan).clip(lower=0) + COST_IDLE * (out.plan - a).clip(lower=0)
    out["cost_schedule"] = COST_SHORT * (a - out.schedule).clip(lower=0) + COST_IDLE * (out.schedule - a).clip(lower=0)
    out["covered"] = np.where(out.observed, (a >= out.lo) & (a <= out.hi), np.nan)
    return out


# ---------------------------------------------------------------- drift
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index on reference-quantile bins."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


CALENDAR_FEATURES = {"dow", "month", "doy_sin", "doy_cos", "dom", "is_month_end", "is_holiday", "days_to_hol",
                     "days_from_hol", "is_bridge", "late_december", "covid_2020", "holiday_on_weekday"}


def drift(b: Bundle, top: int = 12) -> pd.DataFrame:
    """PSI of the most important non-calendar features: shadow period vs. the same
    calendar window one year earlier (a whole-year reference flags plain seasonality)."""
    df, _ = load_table(MODES[b.mode])
    imp = pd.Series(b.models["point"].booster_.feature_importance("gain"), index=b.features)
    imp = imp.drop([c for c in imp.index if c in CALENDAR_FEATURES or c.startswith(("black_", "cyber_", "jewish_",
                                                                                     "peak_", "quarter_", "xmas_",
                                                                                     "thanksgiving_"))])
    feats = imp.sort_values(ascending=False).index[:top]
    cur = df[df.index > b.cutoff]
    year = pd.Timedelta(days=364)  # 52 weeks: keeps weekdays aligned
    ref = df[(df.index > pd.Timestamp(b.cutoff) - year) & (df.index <= cur.index.max() - year)]
    rows = [{"feature": f, "importance_share": imp[f] / imp.sum(),
             "psi": psi(ref[f].to_numpy(float), cur[f].to_numpy(float))} for f in feats]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- health
THRESHOLDS = {"mae_vs_schedule_max": 0.0, "coverage_min": 0.80, "psi_max": 0.25, "min_days": 28}


def health(log: pd.DataFrame, dr: pd.DataFrame) -> dict:
    """Retraining policy. Performance decides (last 28 observed days must beat the schedule and keep
    interval coverage); feature drift alone only raises a warning — it is often seasonal or benign."""
    obs = log[log.observed].sort_values("date").tail(THRESHOLDS["min_days"])
    if len(obs) < THRESHOLDS["min_days"] // 2:
        return {"status": "warming up", "reasons": [f"only {len(obs)} observed days since cutoff"]}
    mae_rel = obs.abs_err.sum() / obs.abs_err_schedule.sum() - 1
    cov = obs.covered.mean()
    drifted = dr[(dr.psi > THRESHOLDS["psi_max"]) & (dr.importance_share > 0.05)].feature.tolist()
    reasons, warnings = [], []
    if mae_rel > THRESHOLDS["mae_vs_schedule_max"]:
        reasons.append(f"last {len(obs)} days: MAE {mae_rel:+.0%} vs schedule")
    if cov < THRESHOLDS["coverage_min"]:
        reasons.append(f"interval coverage {cov:.0%} < {THRESHOLDS['coverage_min']:.0%}")
    if drifted:
        warnings.append("feature drift (PSI > 0.25 vs. same season last year): " + ", ".join(drifted))
    status = "retrain recommended" if reasons else ("drift warning" if warnings else "healthy")
    return {"status": status, "reasons": reasons, "warnings": warnings,
            "last_days": len(obs), "mae_vs_schedule": float(mae_rel), "coverage": float(cov)}


def run_shadow() -> dict:
    MONITOR.mkdir(parents=True, exist_ok=True)
    report = {}
    for mode in MODES:
        b = latest_bundle(mode)
        log = shadow(b)
        dr = drift(b)
        log.to_csv(MONITOR / f"shadow_{mode}.csv", index=False)
        dr.round(4).to_csv(MONITOR / f"drift_{mode}.csv", index=False)
        obs = log[log.observed]
        report[mode] = {
            "name": MODE_NAMES[mode], "cutoff": b.cutoff, "data_end": str(log.date.max().date()),
            "observed_days": int(len(obs)),
            "mae": float(obs.abs_err.mean()), "mae_schedule": float(obs.abs_err_schedule.mean()),
            "cost_vs_schedule": float(obs.cost_plan.sum() / obs.cost_schedule.sum() - 1),
            "coverage": float(obs.covered.mean()),
            "health": health(log, dr),
        }
    report["generated"] = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M %Z")
    (MONITOR / "status.json").write_text(json.dumps(report, indent=2))
    return report


def train_all(cutoff: str) -> None:
    for mode in MODES:
        b = train(mode, cutoff)
        b.save(bundle_path(mode, cutoff))
        card = {
            "model": f"LightGBM quantile/huber bundle, mode={mode} ({MODE_NAMES[mode]})",
            "cutoff": cutoff, "train_days": b.train_days, "features": b.features,
            "targets": "log(actual / schedule); point = huber, plan = quantile 0.71, interval = CQR 5-95% + ACI",
            "params": {k: v for k, v in LGB_BASE.items() if k != "n_estimators"},
            "best_iterations": {k: int(m.best_iteration_) for k, m in b.models.items()},
            "seed": SEED,
        }
        (MODELS / f"lgbm_{mode}_{cutoff}.json").write_text(json.dumps(card, indent=2))
        print(f"trained {mode}: {b.train_days} days up to {cutoff}, iterations {card['best_iterations']}")
