"""High-resolution cross-correlation scan on RETURNS.

For each ordered pair (A, B), compute CCF(k) = corr(ret_A[t], ret_B[t+k])
at every lag k in [-300, 300]. Look for pairs where:
  - the peak |CCF(k)| occurs at k != 0
  - peak is at SAME sign across all 3 days
  - peak magnitude is at least 0.05 (not noise)
  - peak is meaningfully higher than CCF(0) (the contemporaneous baseline)
"""
from __future__ import annotations
import sys
from typing import Tuple

import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

# Use coarse-but-wide lag grid first to find candidates, then refine
LAGS = list(range(-300, 301, 5))


def load_pivot() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = f"historical/ROUND_5/prices_round_5_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    pivot = p.pivot_table(index="t", columns="product", values="mid_price", aggfunc="first").sort_index()
    return pivot.dropna(how="any")


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
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


def main():
    pivot = load_pivot()
    products = list(pivot.columns)
    rets = pivot.diff().dropna()
    print(f"products: {len(products)}, ret rows: {len(rets)}", file=sys.stderr)

    print(f"scanning {len(products)*(len(products)-1)} pairs at {len(LAGS)} lags…", file=sys.stderr)
    out = []
    for a in products:
        ax = rets[a].to_numpy(dtype=float)
        for b in products:
            if a == b:
                continue
            bx = rets[b].to_numpy(dtype=float)
            cors = {k: lagged_corr(ax, bx, k) for k in LAGS}
            # find peak |corr|
            best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
            best = cors[best_k]
            c0 = cors[0]
            if np.isnan(best) or abs(best) < 0.05:
                continue
            if best_k == 0:
                continue
            uplift = abs(best) - abs(c0)
            if uplift < 0.02:
                continue
            out.append((a, b, best_k, best, c0, uplift))

    out.sort(key=lambda r: -abs(r[3]))
    print(f"\ncandidates: {len(out)}\n")
    print(f"{'A (lead?)':28s} {'B (lag?)':28s} {'k':>5s}  {'peak':>7s}  {'@0':>7s}  {'uplift':>7s}")
    for a, b, k, peak, c0, up in out[:25]:
        print(f"{a:28s} {b:28s} {k:>5d}   {peak:+.3f}   {c0:+.3f}   {up:+.3f}")

    # Per-day verification on top 10
    print("\nPer-day verification (top 10):")
    print(f"{'A':28s} {'B':28s} {'k':>5s}  {'all':>7s}  {'d2':>7s} {'d3':>7s} {'d4':>7s}")
    for a, b, k, peak, c0, up in out[:10]:
        per_day = []
        for d in DAYS:
            sub = pivot[(pivot.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])]
            srets = sub.diff().dropna()
            x = srets[a].to_numpy(); y = srets[b].to_numpy()
            per_day.append(lagged_corr(x, y, k))
        d_str = "  ".join(f"{c:+.3f}" if not np.isnan(c) else " nan " for c in per_day)
        print(f"{a:28s} {b:28s} {k:>5d}   {peak:+.3f}    {d_str}")


if __name__ == "__main__":
    main()
