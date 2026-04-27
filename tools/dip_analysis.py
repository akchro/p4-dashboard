"""
Long-only entry timing for OTM vouchers.

Q1: Do voucher mids mean-revert? (Buy-the-dip is only profitable if so.)
Q2: Conditional on a recent drop, what's the forward return distribution?
Q3: Does bid_vol_1 ≥ some threshold improve the dip entry?
Q4: Per-strike: which strike has the best dip-recovery edge?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HIST = REPO / "historical" / "ROUND_3"

ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = [f"VEV_{k}" for k in ALL_STRIKES] + ["VELVETFRUIT_EXTRACT"]


def load(day: int) -> pd.DataFrame:
    df = pd.read_csv(HIST / f"prices_round_3_day_{day}.csv", sep=";")
    df = df[df["product"].isin(PRODUCTS)].copy()
    for col in ["bid_volume_1", "ask_volume_1"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


def piv(df, value):
    return df.pivot(index="timestamp", columns="product", values=value).sort_index()


def autocorr(day: int) -> None:
    """Autocorr of returns at various lags. Negative = mean reverting."""
    df = load(day)
    mids = piv(df, "mid_price")
    print(f"\n=== Day {day} — return autocorrelation (negative = mean reverting) ===")
    print(f"{'product':>22} {'σ_per_tick':>11} {'ac(1)':>8} {'ac(5)':>8} {'ac(10)':>8} {'ac(50)':>8}")
    for sym in PRODUCTS:
        if sym not in mids.columns: continue
        m = mids[sym].ffill()
        r = m.diff().dropna()
        if r.std() == 0 or len(r) < 200: continue
        acs = []
        for L in (1, 5, 10, 50):
            r_lag = r.shift(L)
            mask = r.notna() & r_lag.notna()
            if mask.sum() < 50: acs.append(np.nan); continue
            acs.append(np.corrcoef(r[mask], r_lag[mask])[0, 1])
        print(f"{sym:>22} {r.std():>11.3f} " + " ".join(f"{a:>+8.4f}" for a in acs))


def dip_entry(day: int, lookback: int = 10, holding: int = 50,
              dip_thresh_ticks: float = -1.0) -> None:
    """For each tick t, check return over [t-lookback, t]. If <= dip_thresh,
    enter long, hold for `holding` ticks, record forward return.
    """
    df = load(day)
    mids = piv(df, "mid_price")
    print(f"\n=== Day {day} — dip-buying: enter when {lookback}-tick return ≤ {dip_thresh_ticks}, hold {holding} ticks ===")
    print(f"{'product':>22} {'n_dips':>7} {'mean_fwd':>10} {'std_fwd':>9} {'pos_frac':>9} {'sharpe_proxy':>13}")
    for sym in PRODUCTS:
        if sym not in mids.columns: continue
        m = mids[sym].ffill()
        if m.std() == 0: continue
        past_ret = m - m.shift(lookback)
        fwd_ret = m.shift(-holding) - m
        signal = (past_ret <= dip_thresh_ticks)
        sub = fwd_ret[signal & past_ret.notna() & fwd_ret.notna()]
        if len(sub) < 20: continue
        mean = sub.mean(); std = sub.std()
        pos = (sub > 0).mean()
        sharpe = mean / std if std else np.nan
        print(f"{sym:>22} {len(sub):>7} {mean:>+10.3f} {std:>9.3f} {pos:>+9.1%} {sharpe:>+13.4f}")


def percentile_entry(day: int, sym: str, lookback: int = 200,
                     holding: int = 200, pct_thresh: float = 0.10) -> None:
    """Cleaner entry: enter long when current mid is in bottom `pct_thresh`
    of last `lookback` ticks. Compare to top quantile (where you'd lose).
    """
    df = load(day)
    mids = piv(df, "mid_price")
    if sym not in mids.columns: return
    m = mids[sym].ffill()
    if m.std() == 0: return
    rolling_pct = m.rolling(lookback, min_periods=lookback).rank(pct=True)
    fwd = m.shift(-holding) - m
    print(f"\n=== Day {day} — {sym}: rolling-rank entry "
          f"(lookback={lookback}, hold={holding}) ===")
    for label, mask in [("BOT decile (entry)", rolling_pct <= pct_thresh),
                        ("MID 50%",            (rolling_pct > 0.25) & (rolling_pct < 0.75)),
                        ("TOP decile (sell)",  rolling_pct >= 1 - pct_thresh)]:
        sub = fwd[mask & rolling_pct.notna() & fwd.notna()]
        if len(sub) < 20: continue
        print(f"  {label:>20}: n={len(sub):>5}  mean_fwd={sub.mean():>+8.3f}  "
              f"pos_frac={(sub > 0).mean():.1%}  std={sub.std():>5.3f}")


def percentile_grid(day: int, sym: str = "VEV_5300") -> None:
    """Sweep (lookback, holding) for the bottom-decile entry strategy."""
    df = load(day)
    mids = piv(df, "mid_price")
    if sym not in mids.columns: print(f"no {sym}"); return
    m = mids[sym].ffill()
    if m.std() == 0: print(f"flat {sym}"); return
    print(f"\n=== Day {day} — {sym}: bottom-decile entry grid ===")
    print(f"{'lookback':>9} {'holding':>8} {'n':>6} {'mean_fwd':>10} "
          f"{'top_dec_fwd':>12} {'spread':>8}")
    for lookback in (50, 100, 200, 500, 1000):
        rp = m.rolling(lookback, min_periods=lookback).rank(pct=True)
        for holding in (50, 100, 500, 1000):
            fwd = m.shift(-holding) - m
            bot = fwd[(rp <= 0.10) & rp.notna() & fwd.notna()]
            top = fwd[(rp >= 0.90) & rp.notna() & fwd.notna()]
            if len(bot) < 20 or len(top) < 20: continue
            print(f"{lookback:>9} {holding:>8} {len(bot):>6} "
                  f"{bot.mean():>+10.3f} {top.mean():>+12.3f} "
                  f"{bot.mean() - top.mean():>+8.3f}")


def dip_with_bid_filter(day: int, sym: str = "VEV_5300",
                        lookback: int = 50, holding: int = 100) -> None:
    """Dip-buy, but only when bid_vol_1 is in the top N quintiles.
    Tests whether the bidder-bot signal improves dip entries."""
    df = load(day)
    mids = piv(df, "mid_price")
    bids = piv(df, "bid_volume_1")
    if sym not in mids.columns or sym not in bids.columns: return
    m = mids[sym].ffill()
    b = bids[sym]
    past = m - m.shift(lookback)
    fwd = m.shift(-holding) - m
    sig_dip = (past <= -1.0)
    print(f"\n=== Day {day} — {sym}: dip-buy filtered by bid_vol_1 quintile ===")
    print(f"  (lookback={lookback}, holding={holding})")
    print(f"{'b1_quintile':>13} {'n':>6} {'mean_b1':>9} {'mean_fwd':>10} {'pos_frac':>9}")
    df_q = pd.DataFrame({"b": b, "past": past, "fwd": fwd}).dropna()
    df_q = df_q[df_q["past"] <= -1.0]
    if len(df_q) < 20: print("  (not enough dips)"); return
    df_q["q"] = pd.qcut(df_q["b"], 5, labels=False, duplicates="drop")
    for q, sub in df_q.groupby("q"):
        print(f"{int(q):>13} {len(sub):>6} {sub['b'].mean():>9.1f} "
              f"{sub['fwd'].mean():>+10.3f} {(sub['fwd'] > 0).mean():>+9.1%}")
    # Combined: top-2-quintile bid AND dip
    top2 = df_q[df_q["q"] >= 3]["fwd"]
    bot1 = df_q[df_q["q"] == 0]["fwd"]
    if len(top2) > 20:
        print(f"\n  COMBINED: dip + b1 in top 2 quintiles: n={len(top2)}, "
              f"mean_fwd={top2.mean():+.3f}, win_rate={(top2 > 0).mean():.1%}")
    if len(bot1) > 20:
        print(f"  COMBINED: dip + b1 in bottom quintile: n={len(bot1)}, "
              f"mean_fwd={bot1.mean():+.3f}, win_rate={(bot1 > 0).mean():.1%}")


def buy_and_hold_eod(day: int) -> None:
    """Simplest strategy: buy at random tick, hold to EOD. Check if positive."""
    df = load(day)
    mids = piv(df, "mid_price")
    print(f"\n=== Day {day} — buy-anywhere, mark-to-EOD-mid (entry @mid) ===")
    print(f"  (positive = naked long has positive return on this day)")
    print(f"{'product':>22} {'mid_open':>10} {'mid_close':>10} {'change':>9}")
    for sym in PRODUCTS:
        if sym not in mids.columns: continue
        m = mids[sym].ffill().dropna()
        if len(m) < 100: continue
        print(f"{sym:>22} {m.iloc[0]:>10.2f} {m.iloc[-1]:>10.2f} "
              f"{m.iloc[-1] - m.iloc[0]:>+9.2f}")


if __name__ == "__main__":
    days = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2]
    for d in days:
        buy_and_hold_eod(d)
        autocorr(d)
        # Bottom-decile entry strategy
        for sym in ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400",
                    "VELVETFRUIT_EXTRACT"]:
            percentile_entry(d, sym, lookback=200, holding=200)
        percentile_grid(d, "VEV_5300")
