"""Two scans:
1. CIRCLE → every other product (cross-category): does CIRCLE predict
   anything outside MICROCHIP at any lag?
2. Every product → CIRCLE: what predicts CIRCLE itself? Plus CIRCLE's
   own AR structure.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

LAGS_FINE = list(range(0, 301, 5))   # ±300 in 5-step
LAGS_AR  = (1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)


def load_pivot() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        df = pd.read_csv(f"historical/ROUND_5/prices_round_5_day_{d}.csv", sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    return p.pivot_table(index="t", columns="product", values="mid_price",
                         aggfunc="first").sort_index().dropna(how="any")


def lagged_corr(x, y, k):
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


def per_day_corr(pivot, a_name, b_name, k):
    out = []
    for d in DAYS:
        sub = pivot[(pivot.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])].diff().dropna()
        x = sub[a_name].to_numpy(); y = sub[b_name].to_numpy()
        out.append(lagged_corr(x, y, k))
    return out


def main():
    pivot = load_pivot()
    rets = pivot.diff().dropna()
    products = list(rets.columns)
    CIRCLE = "MICROCHIP_CIRCLE"
    print(f"loaded {len(rets)} ret rows, {len(products)} products", file=sys.stderr)

    # Spreads (median per product)
    print("\n=== Spread context (median) ===")
    half_spreads = {}
    for prod in products:
        prod_df = pd.read_csv("historical/ROUND_5/prices_round_5_day_2.csv", sep=";")
        prod_df = prod_df[prod_df["product"] == prod]
        if not prod_df.empty:
            spr = (prod_df["ask_price_1"] - prod_df["bid_price_1"]).median()
            half_spreads[prod] = spr / 2
    # Print 10 tightest
    sorted_hs = sorted(half_spreads.items(), key=lambda x: x[1])[:10]
    print("Tightest half-spreads:")
    for p, hs in sorted_hs:
        print(f"  {p:30s}  half-spread = {hs:5.2f}")

    # === Scan 1: CIRCLE → every other product ===
    print("\n=== Scan 1: CIRCLE → every other product (lag 0..300, step 5) ===")
    ax = rets[CIRCLE].to_numpy()
    candidates = []
    for b in products:
        if b == CIRCLE:
            continue
        bx = rets[b].to_numpy()
        cors = {k: lagged_corr(ax, bx, k) for k in LAGS_FINE}
        # find best |corr|
        best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
        best = cors[best_k]
        c0 = cors.get(0, np.nan)
        if np.isnan(best):
            continue
        if abs(best) > 0.03 and best_k != 0:
            candidates.append((b, best_k, best, c0))

    candidates.sort(key=lambda r: -abs(r[2]))
    print(f"\nTop candidates ({len(candidates)} total with |peak| > 0.03 at lag != 0):")
    print(f"{'follower':30s} {'k':>5s}  {'rho':>7s}  {'@0':>7s}  d2/d3/d4  half-spr  pred(σ_C)")
    sig_C = rets[CIRCLE].std()
    for b, k, rho, c0 in candidates[:20]:
        per = per_day_corr(pivot, CIRCLE, b, k)
        d_str = '/'.join(f'{c:+.3f}' if not np.isnan(c) else 'nan' for c in per)
        sig_F = rets[b].std()
        pred_per_sigma = abs(rho) * sig_F  # |β| * σ_C = |corr| * σ_F when standardized
        hs = half_spreads.get(b, float('nan'))
        flag = "" if all(np.sign(c) == np.sign(rho) for c in per if not np.isnan(c)) else " (FLIPS)"
        print(f"{b:30s} {k:>5d}  {rho:+.4f}  {c0:+.4f}  {d_str}  {hs:5.2f}     {pred_per_sigma:.3f}{flag}")

    # === Scan 2: every product → CIRCLE (what predicts CIRCLE?) ===
    print("\n\n=== Scan 2: every product → CIRCLE (lag 0..300, step 5) ===")
    bx = rets[CIRCLE].to_numpy()
    cand2 = []
    for a in products:
        if a == CIRCLE:
            continue
        ax_l = rets[a].to_numpy()
        cors = {k: lagged_corr(ax_l, bx, k) for k in LAGS_FINE}
        best_k = max(cors, key=lambda k: abs(cors[k]) if not np.isnan(cors[k]) else 0)
        best = cors[best_k]
        c0 = cors.get(0, np.nan)
        if np.isnan(best):
            continue
        if abs(best) > 0.03 and best_k != 0:
            cand2.append((a, best_k, best, c0))
    cand2.sort(key=lambda r: -abs(r[2]))
    print(f"\nTop candidates ({len(cand2)}):")
    print(f"{'leader':30s} {'k':>5s}  {'rho':>7s}  {'@0':>7s}  d2/d3/d4")
    for a, k, rho, c0 in cand2[:20]:
        per = per_day_corr(pivot, a, CIRCLE, k)
        d_str = '/'.join(f'{c:+.3f}' if not np.isnan(c) else 'nan' for c in per)
        flag = "" if all(np.sign(c) == np.sign(rho) for c in per if not np.isnan(c)) else " (FLIPS)"
        print(f"{a:30s} {k:>5d}  {rho:+.4f}  {c0:+.4f}  {d_str}{flag}")

    # === Scan 3: CIRCLE's own AR structure ===
    print("\n\n=== Scan 3: CIRCLE's own AR structure ===")
    ax = rets[CIRCLE].to_numpy()
    print(f"{'lag':>5s}  {'corr':>7s}  d2/d3/d4")
    for k in LAGS_AR:
        c = lagged_corr(ax, ax, k)
        per = per_day_corr(pivot, CIRCLE, CIRCLE, k)
        d_str = '/'.join(f'{cc:+.3f}' if not np.isnan(cc) else 'nan' for cc in per)
        print(f"{k:>5d}  {c:+.4f}  {d_str}")


if __name__ == "__main__":
    main()
