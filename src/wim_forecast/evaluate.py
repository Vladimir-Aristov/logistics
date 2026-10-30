"""Evaluate all backtest predictions: per-fold metrics, significance, ablations, figures.

Reads reports/backtest/*.parquet, writes
    reports/backtest_summary.csv   mean +- std across folds, pooled significance
    reports/ablation_summary.csv   change vs. full LightGBM when a feature group is dropped
    reports/figures/backtest_*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .common import COST_IDLE, COST_SHORT, FOLD_STARTS, REP, load_table, score, to_label

BT = REP / "backtest"
FIG = REP / "figures"
BASE = "Schedule"


def attach_truth(p: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    t = df.loc[p.date, ["ratio", "schedule", "trucks"]].reset_index(drop=True)
    p = p.reset_index(drop=True)
    p["y_ratio"], p["sched"], p["actual"] = t.ratio, t.schedule, t.trucks
    p["abs_err"] = (p.actual - p.ratio * p.sched).abs()
    short = (p.actual - p.plan).clip(lower=0)
    p["cost"] = COST_SHORT * short + COST_IDLE * (p.plan - p.actual).clip(lower=0)
    return p


def per_fold(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, k), g in p.groupby(["model", "fold"]):
        s = score(g.y_ratio.to_numpy(), g.sched.to_numpy(), g.ratio.to_numpy(), g.label.to_numpy(), g.plan.to_numpy())
        rows.append({"model": m, "fold": k, "days": len(g), **s})
    f = pd.DataFrame(rows)
    base = f[f.model == BASE].set_index("fold").dispatch_cost
    f["cost_vs_schedule"] = f.dispatch_cost / f.fold.map(base) - 1
    return f


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> float:
    """Two-sided DM test p-value, H0: equal expected loss. Newey-West variance (weekly lag)."""
    d = loss_a - loss_b
    n, mu = len(d), d.mean()
    dc = d - mu
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    dm = mu / np.sqrt(var / n)
    return float(2 * stats.t.sf(abs(dm), df=n - 1))


def block_bootstrap_ci(loss_m, loss_b, dates, n=2000, seed=0) -> tuple[float, float]:
    """95% CI of the relative improvement sum(m)/sum(b)-1, resampling whole weeks."""
    wk = pd.Series(pd.to_datetime(dates)).dt.to_period("W").to_numpy()
    uw, inv = np.unique(wk, return_inverse=True)
    sm, sb = np.bincount(inv, loss_m), np.bincount(inv, loss_b)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uw), size=(n, len(uw)))
    rel = sm[pick].sum(1) / sb[pick].sum(1) - 1
    return float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))


def pooled_tests(p: pd.DataFrame) -> pd.DataFrame:
    base = p[p.model == BASE].set_index("date")
    rows = []
    for m, g in p.groupby("model"):
        g = g.set_index("date")
        common = base.index.intersection(g.index)
        g, b = g.loc[common], base.loc[common]
        lo, hi = block_bootstrap_ci(g.cost.to_numpy(), b.cost.to_numpy(), g.index)
        rows.append({
            "model": m,
            "mae_vs_schedule": g.abs_err.sum() / b.abs_err.sum() - 1,
            "dm_p_mae": np.nan if m == BASE else diebold_mariano(g.abs_err.to_numpy(), b.abs_err.to_numpy()),
            "cost_vs_schedule_pooled": g.cost.sum() / b.cost.sum() - 1,
            "cost_ci95_lo": lo, "cost_ci95_hi": hi,
            "dm_p_cost": np.nan if m == BASE else diebold_mariano(g.cost.to_numpy(), b.cost.to_numpy()),
        })
    return pd.DataFrame(rows).set_index("model")


def summarize(f: pd.DataFrame) -> pd.DataFrame:
    cols = ["macro_f1", "recall_below", "recall_above", "mae_trucks", "cost_vs_schedule"]
    agg = f.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg


def fmt(summary: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    def ms(c, pct=False, d=2):
        m, s = summary[f"{c}_mean"], summary[f"{c}_std"]
        if pct:
            return (m * 100).map(lambda x: f"{x:+.0f}%") + " ± " + (s * 100).map(lambda x: f"{x:.0f}")
        return m.map(lambda x: f"{x:.{d}f}") + " ± " + s.map(lambda x: f"{x:.{d}f}")
    t = pd.DataFrame({
        "Macro-F1": ms("macro_f1"), "Recall below": ms("recall_below"), "Recall above": ms("recall_above"),
        "MAE trucks": ms("mae_trucks", d=0), "Cost vs schedule (per fold)": ms("cost_vs_schedule", pct=True),
        "Cost, pooled [95% CI]": tests.apply(lambda r: f"{r.cost_vs_schedule_pooled*100:+.1f}% "
                                             f"[{r.cost_ci95_lo*100:+.1f}, {r.cost_ci95_hi*100:+.1f}]", axis=1),
        "DM p (MAE)": tests.dm_p_mae.map(lambda x: "—" if np.isnan(x) else f"{x:.3g}"),
    })
    return t.sort_values("MAE trucks", key=lambda s: s.str.split().str[0].astype(float))


ENSEMBLES = {
    "Ensemble LGB+GRU": ["LightGBM regressor", "GRU (5 seeds)"],
    "Ensemble LGB+GRU+Chronos": ["LightGBM regressor", "GRU (5 seeds)", "Chronos-2 + covariates"],
}


def ensemble(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per setup: geometric mean of member ratio forecasts (equal weights, no tuning)."""
    out = []
    for setup, g in raw.groupby("setup"):
        for name, members in ENSEMBLES.items():
            parts = [g[g.model == m].set_index("date") for m in members]
            if any(x.empty for x in parts):
                continue
            idx = parts[0].index
            for x in parts[1:]:
                idx = idx.intersection(x.index)
            r = np.exp(np.mean([np.log(x.loc[idx, "ratio"].to_numpy()) for x in parts], axis=0))
            out.append(pd.DataFrame({"date": idx, "fold": parts[0].loc[idx, "fold"].to_numpy(), "setup": setup,
                                     "model": name, "ratio": r, "label": to_label(r),
                                     "plan": r * df.schedule.loc[idx].to_numpy()}))
    return pd.concat(out) if out else raw.iloc[:0]


def load_predictions(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "fold", "setup", "model", "ratio", "label", "plan"]
    parts = []
    for f in sorted(BT.glob("*.parquet")):
        x = pd.read_parquet(f)
        if "setup" not in x:
            x["setup"] = "baseline"
        parts.append(x[keep])
    raw = pd.concat(parts)
    raw = pd.concat([raw, ensemble(raw, df)])
    p = attach_truth(raw, df)
    p["model_setup"] = p.model
    p["model"] = np.where(p.setup == "baseline", p.model, p.model + " · " + p.setup)
    return p


def setup_table(p: pd.DataFrame, setup: str) -> pd.DataFrame:
    q = p[p.setup.isin(["baseline", setup])]
    q = q[q.date.isin(q[q.setup == setup].date)]
    return fmt(summarize(per_fold(q)), pooled_tests(q))


def compare_setups(p: pd.DataFrame) -> pd.DataFrame:
    """Key models across setups, pooled over the days each pair of setups shares."""
    rows = []
    sched = p[p.model == BASE].set_index("date")
    for setup, g in p[p.setup != "baseline"].groupby("setup"):
        days = g.date.unique()
        r = {"setup": setup, "days": len(days)}
        b = sched.loc[days]
        for name, col in [("LightGBM regressor", "abs_err"), ("Ensemble LGB+GRU", "abs_err"),
                          ("LightGBM quantile plan", "cost")]:
            x = g[g.model_setup == name].set_index("date")
            if len(x):
                metric = "MAE" if col == "abs_err" else "cost"
                r[f"{name}: {metric} vs schedule"] = x[col].sum() / b.loc[x.index, col].sum() - 1
        gru = g[g.model_setup == "GRU (5 seeds)"]
        if len(gru):
            s = score(gru.y_ratio.to_numpy(), gru.sched.to_numpy(), gru.ratio.to_numpy(), gru.label.to_numpy())
            r["GRU recall below"], r["GRU recall above"] = s["recall_below"], s["recall_above"]
            r["GRU macro-F1"] = s["macro_f1"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("setup")


def ablation_table(path, df) -> pd.DataFrame:
    a = attach_truth(pd.read_parquet(path), df)
    a["variant"] = a.model.str.split().str[0]
    a["head"] = a.model.str.split(n=1).str[1]
    reg = a[a["head"] == "regressor"].groupby("variant").abs_err.mean()
    plan = a[a["head"] == "quantile plan"].groupby("variant").cost.sum()
    full_reg = a[(a.variant == "full") & (a["head"] == "regressor")].set_index("date").abs_err
    rows = []
    for v in reg.index:
        g = a[(a.variant == v) & (a["head"] == "regressor")].set_index("date").abs_err.loc[full_reg.index]
        rows.append({"variant": v, "mae": reg[v], "mae_vs_full": reg[v] / reg["full"] - 1,
                     "plan_cost_vs_full": plan[v] / plan["full"] - 1,
                     "dm_p_vs_full": np.nan if v == "full" else diebold_mariano(g.to_numpy(), full_reg.to_numpy())})
    return pd.DataFrame(rows).set_index("variant").sort_values("mae_vs_full", ascending=False)


def plot_folds(p: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    f = per_fold(p)
    show = [BASE, "SARIMAX + holidays", "Chronos-2 + covariates · D-1", "LightGBM regressor · D-1",
            "Ensemble LGB+GRU+Chronos · D-1", "N-HiTS (hourly) · Nowcast 07:00", "LightGBM regressor · Nowcast 07:00",
            "Ensemble LGB+GRU+Chronos · Nowcast 07:00"]
    piv = f.pivot(index="fold", columns="model", values="mae_trucks")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in [x for x in show if x in piv]:
        ax.plot(piv.index, piv[m], marker="o", lw=2.4 if "Nowcast" in m or m == BASE else 1.3, label=m)
    ax.set_xticks(piv.index, [s[:7] for s in FOLD_STARTS])
    ax.set_ylabel("MAE, trucks/day"); ax.set_xlabel("fold start (6-month test window)")
    ax.set_title("Rolling-origin backtest: error by fold"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "backtest_mae_by_fold.png", dpi=140); plt.close(fig)


def main():
    df, _ = load_table()
    p = load_predictions(df)
    pd.set_option("display.width", 250)
    for setup, fname in [("D-1", "backtest_summary.csv"), ("Nowcast 07:00", "backtest_summary_nowcast.csv")]:
        if (p.setup == setup).any():
            t = setup_table(p, setup)
            t.to_csv(REP / fname)
            print(f"\n=== {setup}: {p[(p.setup == setup)].date.nunique()} days ===")
            print(t.to_string())
    cmp_ = compare_setups(p)
    cmp_.round(4).to_csv(REP / "setup_comparison.csv")
    print("\n=== setups (pooled, relative to schedule on the same days) ===")
    print(cmp_.round(3).to_string())

    # weather: observed vs forecast on the days that have a forecast
    wf = p[p.setup == "D-1, weather forecast"]
    if len(wf):
        days = wf.date.unique()
        rows = []
        for setup in ["D-1", "D-1, weather forecast"]:
            g = p[(p.setup == setup) & p.date.isin(days)]
            for name in ["LightGBM regressor", "LightGBM quantile plan"]:
                x = g[g.model_setup == name]
                rows.append({"weather input": "observed (archive)" if setup == "D-1" else "D-1 forecast",
                             "model": name, "MAE": x.abs_err.mean(), "cost": x.cost.sum()})
        w = pd.DataFrame(rows)
        w.round(2).to_csv(REP / "weather_forecast_check.csv", index=False)
        print(f"\n=== weather input, {len(days)} days with a D-1 forecast ===")
        print(w.round(1).to_string(index=False))

    for path in sorted((BT / "ablations").glob("*.parquet")):
        abl = ablation_table(path, df)
        abl.round(4).to_csv(REP / f"{path.stem}_summary.csv")
        print(f"\n=== {path.stem} (LightGBM, pooled over folds) ===")
        print(abl.round(3).to_string())
    # coverage of Chronos' own 90% intervals (vs. conformal ones in reports/intervals_*_summary.csv)
    ch = BT / "chronos.parquet"
    if ch.exists():
        c = pd.read_parquet(ch)
        c["y"] = df.trucks.reindex(c.date).to_numpy()
        c["cover"] = (c.y >= c.q05) & (c.y <= c.q95)
        c["width"] = c.q95 - c.q05
        cc = c.groupby(["setup", "model"]).agg(coverage=("cover", "mean"), width=("width", "mean"))
        cc["worst_fold_coverage"] = c.groupby(["setup", "model", "fold"]).cover.mean().groupby(["setup", "model"]).min()
        cc.round(4).to_csv(REP / "intervals_chronos_summary.csv")
        print("\n=== Chronos native 90% intervals ===")
        print(cc.round(3).to_string())
    plot_folds(p[p.date.isin(p[p.setup == "D-1"].date)])


if __name__ == "__main__":
    main()
