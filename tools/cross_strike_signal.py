"""
Verify: does aggregate (bid_vol_1 - ask_vol_1) on VEV_5300/5400/5500 predict
forward returns on OTHER strikes (5000-5200, 4000-4500, VFE)?
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HIST = REPO / "historical" / "ROUND_3"

SIGNAL_STRIKES = [5300, 5400, 5500]
TARGETS = ["VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
           "VEV_5300", "VEV_5400", "VELVETFRUIT_EXTRACT"]
ALL = TARGETS + [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" not in TARGETS]


def load(d):
    df = pd.read_csv(HIST / f"prices_round_3_day_{d}.csv", sep=";")
    df = df[df["product"].isin(ALL + [f"VEV_{k}" for k in SIGNAL_STRIKES])].copy()
    for c in ["bid_volume_1", "ask_volume_1"]:
        df[c] = df[c].fillna(0).astype(int)
    return df


def piv(df, val):
    return df.pivot(index="timestamp", columns="product", values=val).sort_index()


def cross_signal(day):
    df = load(day)
    bids = piv(df, "bid_volume_1")
    asks = piv(df, "ask_volume_1")
    mids = piv(df, "mid_price")

    # Aggregate signal across signal strikes
    signal_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
    sig = sum(bids[s] - asks[s] for s in signal_syms)

    print(f"\n=== Day {day} — cross-strike OTM signal vs target fwd return ===")
    print(f"  Signal = sum(b1 − a1) across {signal_syms}")
    print(f"  Signal stats: mean={sig.mean():.1f}  std={sig.std():.1f}  median={sig.median():.0f}")
    pos_frac = (sig > 0).mean()
    print(f"  Signal > 0 fraction: {pos_frac:.1%}")
    print(f"\n{'target':>22} {'k':>3} {'corr(sig,ret)':>14} "
          f"{'fwd|sig>0':>11} {'fwd|sig<0':>11} {'spread':>8}")
    for tgt in TARGETS:
        if tgt not in mids.columns: continue
        m = mids[tgt].ffill()
        for k in (1, 5, 10, 50):
            fwd = m.shift(-k) - m
            mask = sig.notna() & fwd.notna()
            if mask.sum() < 50 or fwd[mask].std() == 0: continue
            c = np.corrcoef(sig[mask], fwd[mask])[0, 1]
            pos = fwd[mask & (sig > 0)].mean()
            neg = fwd[mask & (sig < 0)].mean()
            print(f"{tgt:>22} {k:>3} {c:>+14.4f} {pos:>+11.4f} {neg:>+11.4f} "
                  f"{pos - neg:>+8.4f}")


def per_day_summary():
    """Show whether bullish-signal periods would have given net-positive
    long entries on each strike for each day (cross-day reliability)."""
    print("\n=== Mean fwd return by signal sign — per day, all targets ===")
    print(f"  (k=10 ticks horizon)")
    for d in (0, 1, 2):
        df = load(d)
        bids = piv(df, "bid_volume_1")
        asks = piv(df, "ask_volume_1")
        mids = piv(df, "mid_price")
        signal_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
        sig = sum(bids[s] - asks[s] for s in signal_syms)
        print(f"\n  Day {d}:")
        print(f"  {'target':>22} {'fwd|sig>0':>11} {'fwd|sig<0':>11} {'spread':>9}")
        for tgt in TARGETS:
            if tgt not in mids.columns: continue
            m = mids[tgt].ffill()
            fwd = m.shift(-10) - m
            mask = sig.notna() & fwd.notna()
            if mask.sum() < 50 or fwd[mask].std() == 0: continue
            pos = fwd[mask & (sig > 0)].mean()
            neg = fwd[mask & (sig < 0)].mean()
            print(f"  {tgt:>22} {pos:>+11.4f} {neg:>+11.4f} {pos - neg:>+9.4f}")


if __name__ == "__main__":
    for d in (0, 1, 2):
        cross_signal(d)
    per_day_summary()
