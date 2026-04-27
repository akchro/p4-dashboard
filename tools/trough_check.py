"""Re-test E2 (divergence-trough) on the new log."""
from __future__ import annotations
import json
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "2026-04-26_01-25-29.log"

SIGNAL_STRIKES = [5300, 5400, 5500]
TARGETS = ["VEV_5000", "VEV_5100", "VEV_5200"]


def main():
    with open(LOG) as f:
        df = pd.read_csv(StringIO(json.load(f)["activitiesLog"]), sep=";")
    keep = TARGETS + [f"VEV_{k}" for k in SIGNAL_STRIKES]
    df = df[df["product"].isin(keep)].copy()
    for c in ["bid_volume_1", "ask_volume_1"]:
        df[c] = df[c].fillna(0).astype(int)
    df["t"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    bids = df.pivot(index="t", columns="product", values="bid_volume_1").sort_index()
    asks = df.pivot(index="t", columns="product", values="ask_volume_1").sort_index()
    mids = df.pivot(index="t", columns="product", values="mid_price").sort_index()
    sig_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
    bull = sum(bids[s].fillna(0) - asks[s].fillna(0) for s in sig_syms).to_numpy(dtype=float)
    n = len(bull)
    print(f"ticks={n:,}")

    print(f"\n{'lb':>4} {'sm':>4} {'min':>5} {'decay':>6} {'fires':>6}  "
          f"{'fwd5':>7} {'fwd20':>7} {'fwd50':>7} {'fwd100':>7}")
    for tgt in TARGETS:
        m = mids[tgt].ffill().to_numpy(dtype=float)
        for lb, sm, minb, dec in [
            (20, 10, -10, 0.5),  # symmetric to current E1
            (20, 10, -15, 0.5),
            (20, 10, -10, 0.7),
            (20, 10, -10, 0.3),
            (30, 10, -10, 0.5),
            (50, 10, -10, 0.5),
        ]:
            # smoothed bull at each tick (rolling mean window=sm)
            cumsum = np.concatenate([[0.0], np.cumsum(bull)])
            def smooth(i):
                s = max(0, i - sm + 1)
                return (cumsum[i + 1] - cumsum[s]) / (i + 1 - s)
            fires = np.zeros(n, dtype=bool)
            bs_arr = np.array([smooth(i) for i in range(n)])
            for i in range(lb, n):
                bs_min = bs_arr[i - lb + 1: i + 1].min()
                bs_now = bs_arr[i]
                mid_chg = m[i] - m[i - lb]
                if mid_chg < 0 and bs_min <= minb and bs_now >= dec * bs_min:
                    fires[i] = True
            cnt = int(fires.sum())
            row = []
            for k in (5, 20, 50, 100):
                fwd = np.full(n, np.nan)
                fwd[:-k] = m[k:] - m[:-k]
                valid = fires & ~np.isnan(fwd)
                row.append(fwd[valid].mean() if valid.sum() else float("nan"))
            print(f"  [{tgt}] lb={lb:>2} sm={sm:>2} min={minb:>4} dec={dec:>4}  "
                  f"n={cnt:>4}  "
                  f"k=5:{row[0]:+.3f} k=20:{row[1]:+.3f} "
                  f"k=50:{row[2]:+.3f} k=100:{row[3]:+.3f}")
        print()


if __name__ == "__main__":
    main()
