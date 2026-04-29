"""Strict product-pair lead-lag scan.

Selection criteria for a TRUE lead-lag pair:
  1. peak |corr| at lag k != 0 is at least 0.05 HIGHER than corr at lag 0.
  2. peak occurs at the SAME k (within a small window) across all 3 days.
  3. peak |corr| is at least 0.4 (otherwise too noisy to trade).

Run on both LEVELS (z-scored mids) and on returns at the same windowed scale
as the lag (so windows don't overlap and we measure pure directional
predictability).

Print all qualifying pairs.
"""
from __future__ import annotations
import sys
from typing import Dict

import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

# Lags to scan. Wide enough to catch slow signals.
LAGS = (0, 50, 100, 250, 500, 1000, 2000, 5000)


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


def zscore_per_day(pivot: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for d in DAYS:
        sub = pivot[(pivot.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])]
        parts.append((sub - sub.mean()) / sub.std(ddof=1))
    return pd.concat(parts).sort_index()


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
    z = zscore_per_day(pivot)
    products = list(z.columns)
    print(f"products: {len(products)}, ticks: {len(z)}", file=sys.stderr)

    # Scan every product-pair at each lag on z-scored levels
    print(f"scanning {len(products)*(len(products)-1)} pairs at {len(LAGS)} lags…", file=sys.stderr)
    candidates = []
    for a in products:
        ax = z[a].to_numpy(dtype=float)
        for b in products:
            if a == b:
                continue
            bx = z[b].to_numpy(dtype=float)
            cors = {k: lagged_corr(ax, bx, k) for k in LAGS}
            c0 = cors.get(0, np.nan)
            if np.isnan(c0):
                continue
            # Find best non-zero-lag corr
            non_zero = {k: v for k, v in cors.items() if k != 0 and not np.isnan(v)}
            if not non_zero:
                continue
            best_k = max(non_zero, key=lambda k: abs(non_zero[k]))
            best = non_zero[best_k]
            uplift = abs(best) - abs(c0)
            if uplift > 0.05 and abs(best) > 0.40:
                candidates.append((a, b, best_k, best, c0, uplift, cors))

    candidates.sort(key=lambda r: -r[5])  # by uplift
    print(f"\ncandidates with uplift > 0.05 and |peak| > 0.40: {len(candidates)}\n")
    print(f"{'A (lead?)':28s} {'B (lag?)':28s} {'k':>5s}  {'peak':>7s} {'@0':>7s} {'uplift':>7s}")
    for a, b, k, peak, c0, up, cors in candidates[:30]:
        print(f"{a:28s} {b:28s} {k:>5d}   {peak:+.3f}  {c0:+.3f}  {up:+.3f}")

    # For each top candidate, verify per-day
    print(f"\nPer-day verification for top 12 candidates:")
    print(f"{'A (lead?)':28s} {'B (lag?)':28s} {'k':>5s} {'all':>7s}  {'d2':>7s} {'d3':>7s} {'d4':>7s}  {'@0_d2':>6s} {'@0_d3':>6s} {'@0_d4':>6s}")
    for a, b, k, peak, c0, up, _cors in candidates[:12]:
        per_day = []
        per_day_zero = []
        for d in DAYS:
            sub = z[(z.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])]
            xa = sub[a].to_numpy(); xb = sub[b].to_numpy()
            per_day.append(lagged_corr(xa, xb, k))
            per_day_zero.append(lagged_corr(xa, xb, 0))
        d_str = "  ".join(f"{c:+.3f}" if not np.isnan(c) else " nan " for c in per_day)
        d_str0 = " ".join(f"{c:+.3f}" if not np.isnan(c) else " nan " for c in per_day_zero)
        print(f"{a:28s} {b:28s} {k:>5d}  {peak:+.3f}    {d_str}    {d_str0}")


if __name__ == "__main__":
    main()
