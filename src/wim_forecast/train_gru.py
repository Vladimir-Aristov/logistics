"""GRU model (PyTorch) for next-day truck demand vs. schedule.

Runs in its own process (LightGBM and PyTorch ship clashing OpenMP runtimes on macOS).

    python -m wim_forecast.train_gru [--table daily_nowcast] [--setup NAME] [--drop logistics_cal mta]
        rolling folds -> reports/backtest/gru_<setup>.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

from .backtest_groups import FEATURE_GROUPS, slug
from .common import REP, SEED, folds, load_table

SEQ_LEN = 28


class GRUNet(nn.Module):
    """GRU over the last 28 days + MLP over day-D known features; two heads."""

    def __init__(self, seq_dim, stat_dim, hidden=64):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=1)
        self.stat = nn.Sequential(nn.Linear(stat_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.reg = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, 3)

    def forward(self, seq, stat):
        _, h = self.gru(seq)
        z = self.trunk(torch.cat([h[-1], self.stat(stat)], 1))
        return self.reg(z).squeeze(1), self.cls(z)


def gru_arrays(df, cols, train_mask):
    ok = df.ok.to_numpy().astype(float)
    logr = np.nan_to_num(np.log(df.ratio.to_numpy()), nan=0.0) * ok
    logt = np.log(df.trucks.where(df.ok)).to_numpy()
    mu = np.nanmean(logt[train_mask]); sd = np.nanstd(logt[train_mask])
    logt = np.nan_to_num((logt - mu) / sd) * ok
    dow = np.eye(7)[df.index.day_of_week]
    hol = df.is_holiday.to_numpy()[:, None]
    seq_feat = np.column_stack([logr, logt, ok, dow, hol]).astype(np.float32)

    S = df[cols].to_numpy(dtype=float)
    mask = np.isnan(S)
    trm = train_mask
    smu, ssd = np.nanmean(S[trm], 0), np.nanstd(S[trm], 0) + 1e-6
    S = np.nan_to_num((S - smu) / ssd)
    stat = np.column_stack([S, mask]).astype(np.float32)

    n = len(df)
    seq = np.zeros((n, SEQ_LEN, seq_feat.shape[1]), np.float32)
    for i in range(SEQ_LEN, n):
        seq[i] = seq_feat[i - SEQ_LEN:i]   # strictly before day i
    return seq, stat


def fit_gru(df, cols, fit_mask, es_mask, epochs=200, patience=20, seed=SEED):
    """Train on fit_mask days, early-stop on es_mask days. Masks are boolean arrays."""
    torch.manual_seed(seed); np.random.seed(seed)
    seq, stat = gru_arrays(df, cols, fit_mask)
    y_log = np.log(df.ratio.to_numpy()).astype(np.float32)
    y_lab = df.label.cat.codes.to_numpy()
    idx = np.arange(len(df))
    tr = idx[fit_mask & (idx >= SEQ_LEN)]
    va = idx[es_mask]
    T = lambda a: torch.tensor(a)
    net = GRUNet(seq.shape[2], stat.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    w = torch.tensor(len(tr) / (3 * np.bincount(y_lab[tr], minlength=3)), dtype=torch.float32)
    ce, hub = nn.CrossEntropyLoss(weight=w), nn.HuberLoss(delta=0.1)

    def loss_on(ii):
        r, c = net(T(seq[ii]), T(stat[ii]))
        return hub(r, T(y_log[ii])) * 10 + ce(c, T(y_lab[ii]).long())

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        for b in np.array_split(np.random.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            loss = loss_on(b)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_on(va).item()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        r, c = net(T(seq), T(stat))
    return np.exp(r.numpy()), torch.softmax(c, 1).numpy(), ep + 1



def ensemble(df, cols, fit, es, n_seeds=5):
    """Average 5 seeds: a single small GRU on ~1.4k days is noisy."""
    runs = [fit_gru(df, cols, fit, es, seed=SEED + k) for k in range(n_seeds)]
    r = np.exp(np.mean([np.log(x[0]) for x in runs], 0))
    return r, np.mean([x[1] for x in runs], 0), [x[2] for x in runs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="daily")
    ap.add_argument("--setup", default="D-1")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--backtest", action="store_true", help="kept for compatibility; always on")
    a = ap.parse_args()
    torch.set_num_threads(4)
    df, cols = load_table(a.table)
    cols = [c for c in cols if not any(c in FEATURE_GROUPS[g] for g in a.drop)]
    parts = []
    for k, fit, es, test in folds(df):
        r, prob, eps = ensemble(df, cols, fit, es)
        idx = np.where(test)[0]
        sched = df.schedule.to_numpy()
        parts.append(pd.DataFrame({"date": df.index[idx], "fold": k, "setup": a.setup, "model": "GRU (5 seeds)",
                                   "ratio": r[idx], "label": prob.argmax(1)[idx], "plan": r[idx] * sched[idx],
                                   "p_below": prob[idx, 0], "p_normal": prob[idx, 1], "p_above": prob[idx, 2]}))
        print(f"  [{a.setup}] GRU fold {k}: {len(idx)} days, epochs {eps}", flush=True)
    (REP / "backtest").mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_parquet(REP / "backtest" / f"gru_{slug(a.setup)}.parquet", index=False)
