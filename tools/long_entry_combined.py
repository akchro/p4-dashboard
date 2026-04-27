"""
Combine both signals: bottom-decile-of-recent + high-bid-volume.
Hypothesis: dip-buy alone fails on strong trend days, but dip + bidder-bot
bullish (high bid_vol_1) might be more selective and avoid bear traps.

Also: per-strike PnL math. If we just go max-long every day, what's the result?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HIST = REPO / "historical" / "ROUND_3"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]
PRODUCTS = [f"VEV_{k}" for k in STRIKES] + ["VELVETFRUIT_EXTRACT"]
LIMIT = {f"VEV_{k}": 300 for k in STRIKES}
LIMIT["VELVETFRUIT_EXTRACT"] = 200


def load(day: int) -> pd.DataFrame:
    df = pd.read_csv(HIST / f"prices_round_3_day_{day}.csv", sep=";")
    df = df[df["product"].isin(PRODUCTS)].copy()
    for col in ["bid_volume_1", "ask_volume_1", "bid_price_1", "ask_price_1"]:
        df[col] = df[col].fillna(0)
    return df


def piv(df, value):
    return df.pivot(index="timestamp", columns="product", values=value).sort_index()


def buy_and_hold_pnl(days=(0, 1, 2)) -> None:
    """Naive: max long at open mid, mark to EOD mid. Per-strike PnL."""
    print("\n=== Naive max-long-at-open strategy: per-strike per-day PnL ===")
    print("(realized = (close - open) × LIMIT, no entry/exit cost beyond mid)")
    print(f"{'product':>22} ", end="")
    for d in days: print(f"{'D'+str(d)+'_chg':>9} {'D'+str(d)+'_pnl':>10}", end="")
    print(f" {'3d_pnl':>10}")
    totals = {p: 0 for p in PRODUCTS}
    for sym in PRODUCTS:
        print(f"{sym:>22} ", end="")
        for d in days:
            mids = piv(load(d), "mid_price")
            if sym not in mids.columns:
                print(f"{'-':>9} {'-':>10}", end=""); continue
            m = mids[sym].ffill().dropna()
            chg = m.iloc[-1] - m.iloc[0]
            pnl = chg * LIMIT[sym]
            totals[sym] += pnl
            print(f"{chg:>+9.2f} {int(pnl):>+10}", end="")
        print(f" {int(totals[sym]):>+10}")


def combined_signal(day: int, sym: str = "VEV_5300",
                    lookback: int = 200, holding: int = 200) -> None:
    """For symbol on day: bottom-decile entry, then split by bid_vol_1 quintile.
    Show forward return per joint bucket.
    """
    df = load(day)
    mids = piv(df, "mid_price")
    bids = piv(df, "bid_volume_1")
    if sym not in mids.columns: return
    m = mids[sym].ffill()
    if m.std() == 0: return
    rp = m.rolling(lookback, min_periods=lookback).rank(pct=True)
    fwd = m.shift(-holding) - m
    b = bids[sym] if sym in bids.columns else None
    print(f"\n=== Day {day} — {sym}: bot-decile entry × bid_vol_1 quintile ===")
    df_q = pd.DataFrame({"rp": rp, "b": b, "fwd": fwd}).dropna()
    df_q = df_q[df_q["rp"] <= 0.10]
    if len(df_q) < 50: print("  (not enough)"); return
    df_q["q"] = pd.qcut(df_q["b"], 5, labels=False, duplicates="drop")
    print(f"  Total bot-decile entries: {len(df_q)}, mean fwd={df_q['fwd'].mean():+.3f}")
    for q, sub in df_q.groupby("q"):
        print(f"  b1_q={int(q)} (b1={sub['b'].mean():>5.1f}): n={len(sub):>4}  "
              f"mean_fwd={sub['fwd'].mean():>+7.3f}  win={(sub['fwd'] > 0).mean():.0%}")


def per_day_summary(days=(0, 1, 2)) -> None:
    """For each day, summarize: VFE move, naive long PnL per strike."""
    print("\n=== VFE day moves and naive long delta-exposed PnL ===")
    print("(if we held LIMIT contracts long all day at open mid)")
    print(f"{'metric':>30} " + " ".join(f"{'D'+str(d):>10}" for d in days))
    moves = {}
    for d in days:
        mids = piv(load(d), "mid_price")
        vfe = mids["VELVETFRUIT_EXTRACT"].ffill().dropna()
        moves[d] = vfe.iloc[-1] - vfe.iloc[0]
    print(f"{'VFE move (ticks)':>30} " +
          " ".join(f"{moves[d]:>+10.2f}" for d in days))
    for sym in PRODUCTS:
        per_day = []
        for d in days:
            mids = piv(load(d), "mid_price")
            if sym not in mids.columns:
                per_day.append("-"); continue
            m = mids[sym].ffill().dropna()
            chg = m.iloc[-1] - m.iloc[0]
            per_day.append(f"{int(chg * LIMIT[sym]):>+10}")
        print(f"{sym + ' max-long':>30} " + " ".join(f"{x:>10}" for x in per_day))


def compounded_naive(days=(0, 1, 2)) -> None:
    """If we max-long EVERYTHING (every voucher + VFE) every day, total PnL?"""
    total = 0
    for d in days:
        mids = piv(load(d), "mid_price")
        day_pnl = 0
        for sym in PRODUCTS:
            if sym not in mids.columns: continue
            m = mids[sym].ffill().dropna()
            chg = m.iloc[-1] - m.iloc[0]
            day_pnl += chg * LIMIT[sym]
        total += day_pnl
        print(f"  Day {d}: max-long ALL longs PnL = {int(day_pnl):>+10}  (cumulative {int(total):>+10})")
    print(f"  3-day total max-long-everything PnL = {int(total)}")


if __name__ == "__main__":
    buy_and_hold_pnl()
    per_day_summary()
    compounded_naive()
    print("\n--- combined signal across days/strikes ---")
    for d in (0, 1, 2):
        for sym in ("VEV_5000", "VEV_5300", "VEV_5400"):
            combined_signal(d, sym)
