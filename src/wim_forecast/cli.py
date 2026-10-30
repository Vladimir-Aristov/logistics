"""Command line entry point: `wim <command>`.

    wim update               refresh the current year of WIM / MTA data + weather, diesel
    wim build                rebuild daily feature tables
    wim train --cutoff DATE  train and freeze the production models
    wim shadow               score the frozen models on days after the cutoff, check drift
    wim dashboard            render docs/index.html
    wim monthly              update + build + shadow + dashboard (what CI runs)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def update() -> None:
    from . import download, external
    year = pd.Timestamp.today().year
    stale = [year, year - 1] if pd.Timestamp.today().month <= 2 else [year]
    for y in stale:  # past years are immutable and stay cached; the current one is refetched
        for p in list(download.CACHE.glob(f"*_{y}.parquet")) + list((external.RAW_DIR / "mta").glob(f"*_{y}.parquet")):
            p.unlink()
    download.main()
    external.main()


def build() -> None:
    from .build_dataset import build as _build
    df = _build()
    print(f"built {len(df)} days, last {df.index.max().date()}, valid targets {int(df.target_ok.sum())}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update"); sub.add_parser("build"); sub.add_parser("shadow")
    sub.add_parser("dashboard"); sub.add_parser("monthly")
    t = sub.add_parser("train"); t.add_argument("--cutoff", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "update":
        update()
    elif a.cmd == "build":
        build()
    elif a.cmd == "train":
        from .production import train_all
        train_all(a.cutoff)
    elif a.cmd == "shadow":
        from .production import run_shadow
        rep = run_shadow()
        for m in ("d1", "nowcast"):
            r = rep[m]
            print(f"{r['name']}: {r['observed_days']} days, MAE {r['mae']:.0f} vs schedule {r['mae_schedule']:.0f}, "
                  f"cost {r['cost_vs_schedule']:+.0%}, coverage {r['coverage']:.0%} -> {r['health']['status']}")
    elif a.cmd == "dashboard":
        from .dashboard import render
        print(render())
    elif a.cmd == "monthly":
        update(); build()
        main(["shadow"]); main(["dashboard"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

