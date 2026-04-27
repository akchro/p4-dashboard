"""
Diagnose why PEAK_BULL_MIN appears inert. Replay 2026-04-26_01-25-29.log
through the same logic but break down which condition (C1 / E1 / both) fires.
"""
from __future__ import annotations
import json, sys
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "2026-04-26_01-25-29.log"

SIGNAL_STRIKES = [5300, 5400, 5500]
TARGETS = ["VEV_5000", "VEV_5100", "VEV_5200"]
ALL = TARGETS + [f"VEV_{k}" for k in SIGNAL_STRIKES]


def load(path):
    with open(path) as f:
        obj = json.load(f)
    return pd.read_csv(StringIO(obj["activitiesLog"]), sep=";")


def replay(bull_min, div_lookback, div_smooth, div_min_bull, div_decay,
           bull_hist_len, mids, bull_arr, lookback_warm=200):
    """Returns (c1_only, e1_only, both, fwd_arrs) per voucher."""
    out = {}
    n = len(bull_arr)
    for tgt in TARGETS:
        if tgt not in mids.columns:
            continue
        m_arr = mids[tgt].ffill().to_numpy(dtype=float)
        c1 = np.zeros(n, dtype=bool)
        e1 = np.zeros(n, dtype=bool)
        for i in range(n):
            if i < lookback_warm:
                continue
            # C1
            if i >= 2:
                b2, b1, b0 = bull_arr[i-2], bull_arr[i-1], bull_arr[i]
                if max(b2, b1) >= bull_min and (b1 - b2) >= 0 and (b0 - b1) < 0:
                    c1[i] = True
            # E1
            if i >= div_lookback:
                bh = bull_arr[max(0, i - bull_hist_len + 1):i + 1]
                # smoothed bull at tick j: mean over [j-smooth+1 .. j] of bh
                # convert to absolute indices into bull_arr.
                def sm(j):
                    s = max(0, j - div_smooth + 1)
                    return bull_arr[s:j + 1].mean()
                bs_now = sm(i)
                bs_max = max(sm(j) for j in range(i - div_lookback + 1, i + 1))
                mid_chg = m_arr[i] - m_arr[i - div_lookback]
                if (mid_chg > 0 and bs_max >= div_min_bull
                        and bs_now <= div_decay * bs_max):
                    e1[i] = True
        out[tgt] = (c1, e1, m_arr)
    return out


def fwd_stats(fires, m_arr, ks=(5, 20, 50, 100)):
    rows = []
    for k in ks:
        fwd = np.full_like(m_arr, np.nan)
        fwd[:-k] = m_arr[k:] - m_arr[:-k]
        valid = fires & ~np.isnan(fwd)
        n = int(valid.sum())
        mu = fwd[valid].mean() if n else float("nan")
        rows.append((k, n, mu))
    return rows


def main():
    df = load(LOG)
    df = df[df["product"].isin(ALL)].copy()
    for c in ["bid_volume_1", "ask_volume_1"]:
        df[c] = df[c].fillna(0).astype(int)
    df["t"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    bids = df.pivot(index="t", columns="product", values="bid_volume_1").sort_index()
    asks = df.pivot(index="t", columns="product", values="ask_volume_1").sort_index()
    mids = df.pivot(index="t", columns="product", values="mid_price").sort_index()
    sig_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
    bull_arr = sum(bids[s].fillna(0) - asks[s].fillna(0) for s in sig_syms).to_numpy(dtype=int)

    print(f"=== {LOG.name} === ticks={len(bull_arr):,}")
    print(f"bull stats: mean={bull_arr.mean():.1f} std={bull_arr.std():.1f} "
          f"min={bull_arr.min()} max={bull_arr.max()}")

    # Fixed E1 params, vary BULL_MIN.
    print("\n--- C1 / E1 fire rates as PEAK_BULL_MIN varies ---")
    print(f"{'bull_min':>9} {'C1 only':>9} {'E1 only':>9} {'both':>6} {'either':>7}")
    for bm in (0, 5, 20, 50, 100):
        # only need to compute once per bm for any voucher; use VEV_5000.
        out = replay(bm, 20, 10, 10, 0.5, 30, mids, bull_arr)
        c1, e1, _ = out["VEV_5000"]
        both = (c1 & e1).sum()
        either = (c1 | e1).sum()
        print(f"{bm:>9} {(c1 & ~e1).sum():>9} {(e1 & ~c1).sum():>9} {both:>6} {either:>7}")

    # Now: C1-only fwd ret vs E1-only vs both.
    print("\n--- Per-component fwd return (bull_min=5) ---")
    out = replay(5, 20, 10, 10, 0.5, 30, mids, bull_arr)
    for tgt in TARGETS:
        c1, e1, m_arr = out[tgt]
        only_c1 = c1 & ~e1
        only_e1 = e1 & ~c1
        both = c1 & e1
        any_ = c1 | e1
        print(f"\n[{tgt}]")
        for label, fires in [("C1 only", only_c1), ("E1 only", only_e1),
                             ("both",    both),    ("either",  any_)]:
            n = int(fires.sum())
            if n == 0:
                print(f"  {label:>9}  n=0")
                continue
            rows = fwd_stats(fires, m_arr)
            f = " ".join(f"k={k}:{mu:+.3f}" for k, _, mu in rows)
            print(f"  {label:>9}  n={n:>4}  {f}")

    # Sweep E1 thresholds on VEV_5000 to see what tightens it.
    print("\n--- E1 sensitivity sweep on VEV_5000 (C1 disabled by bull_min=999) ---")
    print(f"{'min_bull':>9} {'decay':>6} {'lookback':>9} {'fires':>6}  "
          f"{'fwd5':>7} {'fwd20':>7} {'fwd50':>7} {'fwd100':>7}")
    for min_b, decay, lb in [
        (10, 0.5, 20),  # current
        (15, 0.5, 20),
        (20, 0.5, 20),
        (25, 0.5, 20),
        (10, 0.3, 20),
        (10, 0.7, 20),
        (10, 0.5, 30),
        (10, 0.5, 50),
        (15, 0.3, 30),
        (20, 0.3, 30),
    ]:
        out = replay(999, lb, 10, min_b, decay, max(30, lb), mids, bull_arr)
        c1, e1, m_arr = out["VEV_5000"]
        n = int(e1.sum())
        rows = fwd_stats(e1, m_arr)
        muvals = [mu for _, _, mu in rows]
        print(f"{min_b:>9} {decay:>6} {lb:>9} {n:>6}  "
              f"{muvals[0]:>+7.3f} {muvals[1]:>+7.3f} {muvals[2]:>+7.3f} {muvals[3]:>+7.3f}")


if __name__ == "__main__":
    main()
