"""Round 5 lead-lag scan across all 50 products at multiple lags.

The competition hint: 'consistent timing gaps between related goods are
actionable intelligence'. My round5_edge_analysis only checked lag-1 corr
and found nothing (max 0.022). This script checks larger lags.

Approach:
  1. Build a tick-aligned mid table for all 50 products across 3 days.
  2. Compute Δmid (1-tick) returns for each.
  3. For every ordered pair (A, B) and lag k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        rho_{A->B}(k) = corr(ret_A[t], ret_B[t+k])
     (positive lag k means: A leads B by k ticks)
  4. Surface the top pairs by abs(rho).
  5. For top hits, verify CONSISTENCY across days 2/3/4 separately —
     a real lead-lag should hold OOS.
"""
from __future__ import annotations
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def load_pivot() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = f"historical/ROUND_5/prices_round_5_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * 1_000_000 + df["timestamp"]
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    pivot = p.pivot_table(index="t", columns="product", values="mid_price", aggfunc="first").sort_index()
    return pivot


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """corr(x[t], y[t+k]) — positive k means x leads y by k ticks."""
    if k > 0:
        a = x[:-k]; b = y[k:]
    elif k < 0:
        a = x[-k:]; b = y[:k]
    else:
        a = x; b = y
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]; b = b[mask]
    if len(a) < 100 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scan_pair(rets_A: np.ndarray, rets_B: np.ndarray, lags=LAGS) -> Dict[int, float]:
    out = {}
    for k in lags:
        c = lagged_corr(rets_A, rets_B, k)
        out[k] = c
    return out


def main():
    print("loading…", file=sys.stderr)
    pivot = load_pivot()
    pivot = pivot.dropna(how="any")
    print(f"  pivot shape: {pivot.shape}", file=sys.stderr)
    products = list(pivot.columns)

    # 1-tick changes
    rets = pivot.diff().dropna()
    rets_arr = {p: rets[p].to_numpy(dtype=float) for p in products}

    # 50-tick aggregated changes (smoother, less microstructure noise)
    rets_50 = pivot.diff(50).dropna()
    rets_50_arr = {p: rets_50[p].to_numpy(dtype=float) for p in products}

    # Scan all ordered pairs at all lags. Use 50-tick aggregated returns for
    # signal robustness, and report the lag with the largest |correlation|.
    print(f"scanning {len(products)*len(products)} pairs x {len(LAGS)} lags…", file=sys.stderr)
    results = []
    for a in products:
        for b in products:
            if a == b:
                continue
            cors = scan_pair(rets_50_arr[a], rets_50_arr[b])
            best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
            best = cors[best_k]
            if not np.isnan(best) and abs(best) > 0.05:
                results.append((a, b, best_k, best, cors))

    results.sort(key=lambda r: -abs(r[3]))

    # Print top 30 and verify per-day consistency
    print(f"\nTop 30 lag pairs (50-tick returns; lag k = ticks A leads B):")
    print(f"{'A':28s} {'B':28s} {'k':>5s}  {'rho':>7s}  per-lag profile (1/2/5/10/20/50/100/200/500)")
    for a, b, k, rho, cors in results[:30]:
        prof = "  ".join(f"{cors[L]:+.2f}" if not np.isnan(cors[L]) else " nan " for L in LAGS)
        print(f"{a:28s} {b:28s} {k:>5d}  {rho:+.3f}  {prof}")

    # Per-day check on top 12
    print(f"\nPer-day consistency check on top 12 pairs:")
    print(f"{'A':28s} {'B':28s} {'k':>5s}  {'d2':>7s} {'d3':>7s} {'d4':>7s}")
    for a, b, k, rho, cors in results[:12]:
        per_day = []
        for d in DAYS:
            sub = pivot[pivot.index // 1_000_000 == (d - DAYS[0])]
            sub_rets = sub.diff(50).dropna()
            if a not in sub_rets.columns or b not in sub_rets.columns:
                per_day.append(float("nan"))
                continue
            x = sub_rets[a].to_numpy(); y = sub_rets[b].to_numpy()
            per_day.append(lagged_corr(x, y, k))
        d_str = "  ".join(f"{c:+.3f}" if not np.isnan(c) else " nan  " for c in per_day)
        print(f"{a:28s} {b:28s} {k:>5d}   {d_str}")


if __name__ == "__main__":
    main()
