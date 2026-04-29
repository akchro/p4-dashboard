"""Category-level lead-lag scan for Round 5.

The competition hint emphasizes 'one category moves first while another
follows with a measurable delay'. This script:

1. Builds a category-level index for each of the 10 categories (mean of
   z-scored mids of the 5 products in that category).
2. Computes 1-tick changes of each category index.
3. Scans all 100 ordered (A, B) category pairs at lags
   {1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000}.
4. Reports the lag at which |rho| peaks AND verifies day-by-day consistency.
"""
from __future__ import annotations
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
LAGS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000)
TIMESTAMP_PER_DAY = 1_000_000

CATEGORIES = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS"],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH"],
    "PANEL": ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"],
    "PEBBLES": ["PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"],
    "ROBOT": ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"],
    "SLEEP_POD": [
        "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA"],
    "TRANSLATOR": [
        "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"],
    "UV_VISOR": [
        "UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_YELLOW"],
}


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


def build_category_indices(pivot: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted z-scored mid index per category. Returns a 10-column DF."""
    out = {}
    for cat, prods in CATEGORIES.items():
        # z-score each product per-day to remove cross-product level differences
        # while preserving within-day variation
        z_dfs = []
        for d in DAYS:
            sub = pivot[(pivot.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])]
            sub = sub[prods]
            z = (sub - sub.mean()) / sub.std(ddof=1)
            z_dfs.append(z)
        z_all = pd.concat(z_dfs).sort_index()
        out[cat] = z_all.mean(axis=1)
    return pd.DataFrame(out).dropna()


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """corr(x[t], y[t+k]) — positive k: x leads y by k ticks."""
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
    print(f"pivot shape: {pivot.shape}", file=sys.stderr)
    cat_idx = build_category_indices(pivot)
    print(f"category index shape: {cat_idx.shape}", file=sys.stderr)

    # Scan with 1-tick z-score level changes (since we already z-scored, the
    # diff is in z-units). Also compute on the LEVEL itself to look for
    # slow propagation.
    rets = cat_idx.diff().dropna()
    levels = cat_idx
    cats = list(cat_idx.columns)

    for label, df in (("RETURNS (Δ z-score)", rets), ("LEVELS (z-score)", levels)):
        print(f"\n========== {label} ==========")
        results = []
        for a in cats:
            for b in cats:
                if a == b:
                    continue
                ax = df[a].to_numpy()
                bx = df[b].to_numpy()
                cors = {k: lagged_corr(ax, bx, k) for k in LAGS}
                # Best lag by |corr|
                best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
                best = cors[best_k]
                if not np.isnan(best):
                    results.append((a, b, best_k, best, cors))
        results.sort(key=lambda r: -abs(r[3]))

        print(f"{'A (leader)':14s} {'B (follower)':14s} {'k':>5s}  {'rho':>7s}  per-lag profile (1/5/10/25/50/100/250/500/1k/2k/5k)")
        for a, b, k, rho, cors in results[:20]:
            prof = "  ".join(f"{cors[L]:+.2f}" if not np.isnan(cors[L]) else " nan" for L in LAGS)
            print(f"{a:14s} {b:14s} {k:>5d}  {rho:+.3f}  {prof}")

    # Per-day consistency on the top RETURNS pairs at non-trivial lags (>=5)
    print("\nPer-day consistency on top non-trivial-lag RETURN pairs:")
    rets = cat_idx.diff().dropna()
    cats = list(cat_idx.columns)
    long_lag_results = []
    for a in cats:
        for b in cats:
            if a == b: continue
            ax = rets[a].to_numpy(); bx = rets[b].to_numpy()
            cors = {k: lagged_corr(ax, bx, k) for k in LAGS if k >= 5}
            best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
            best = cors[best_k]
            if not np.isnan(best):
                long_lag_results.append((a, b, best_k, best))
    long_lag_results.sort(key=lambda r: -abs(r[3]))

    print(f"{'A (leader)':14s} {'B (follower)':14s} {'k':>5s}  {'all':>7s}  {'d2':>7s} {'d3':>7s} {'d4':>7s}")
    for a, b, k, rho in long_lag_results[:15]:
        per_day = []
        for d in DAYS:
            sub = cat_idx[(cat_idx.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])]
            srets = sub.diff().dropna()
            x = srets[a].to_numpy(); y = srets[b].to_numpy()
            per_day.append(lagged_corr(x, y, k))
        d_str = "  ".join(f"{c:+.3f}" if not np.isnan(c) else " nan  " for c in per_day)
        print(f"{a:14s} {b:14s} {k:>5d}   {rho:+.3f}    {d_str}")


if __name__ == "__main__":
    main()
