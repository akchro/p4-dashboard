"""Test the cross-leg hedged residual strategy on MICROCHIP.

Idea: at each tick t, target_position[follower] = + sign(beta_F * ret_CIRCLE(t - LAG_F)) * size_F
      target_position[CIRCLE] = -1 * sum(target_position[follower] * hedge_ratio_F)

This isolates the predictable echo of past CIRCLE in each follower while
neutralizing exposure to NEW CIRCLE moves that haven't propagated yet.

Variants tested:
  v1: directional, sign-based, fixed size, no hedge (taker)
  v2: directional, sign-based, fixed size, with CIRCLE hedge (taker)
  v3: scaled by signal magnitude, with hedge (taker)
  v4: scaled by signal magnitude, MAKER (post quotes near mid)

Costs assumed: 1-tick spread cost per round-trip per leg.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

CIRCLE = "MICROCHIP_CIRCLE"
FOLLOWERS = {
    "MICROCHIP_OVAL": 50,
    "MICROCHIP_SQUARE": 100,
    "MICROCHIP_RECTANGLE": 150,
    "MICROCHIP_TRIANGLE": 200,
}
BETAS = {
    "MICROCHIP_OVAL": 0.0676,
    "MICROCHIP_SQUARE": 0.1372,
    "MICROCHIP_RECTANGLE": 0.0866,
    "MICROCHIP_TRIANGLE": 0.0840,
}
POS_LIMIT = 10


def load_pivot() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = f"historical/ROUND_5/prices_round_5_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    return p.pivot_table(index="t", columns="product", values="mid_price",
                         aggfunc="first").sort_index().dropna(how="any")


def run_strategy(pivot, name, target_fn, cost_per_unit=1.0):
    """target_fn(t, ret_C_lagged_dict, current_pos) → dict[product → target_position].
    cost_per_unit: spread cost per unit of trade."""
    products = [CIRCLE] + list(FOLLOWERS.keys())
    rets = pivot.diff()
    pos = {p: 0 for p in products}
    cash = {p: 0.0 for p in products}
    n_trades = 0

    # Pre-compute lagged CIRCLE returns (via shift)
    ret_C = rets[CIRCLE]
    lagged_C = {LAG: ret_C.shift(LAG).fillna(0).to_numpy()
                for LAG in set(FOLLOWERS.values())}

    mids = {p: pivot[p].to_numpy() for p in products}
    n = len(pivot)

    for i in range(1, n):
        ret_C_at_lags = {LAG: lagged_C[LAG][i] for LAG in set(FOLLOWERS.values())}
        target = target_fn(i, ret_C_at_lags, pos, mids)
        for p, tgt in target.items():
            tgt = max(-POS_LIMIT, min(POS_LIMIT, int(tgt)))
            d = tgt - pos[p]
            if d != 0:
                cash[p] -= d * mids[p][i]
                cash[p] -= abs(d) * cost_per_unit
                pos[p] = tgt
                n_trades += 1

    # Mark to last mid
    pnl = {p: cash[p] + pos[p] * mids[p][-1] for p in products}
    total = sum(pnl.values())
    return {"name": name, "total_pnl": total, "by_product": pnl,
            "n_trades": n_trades, "final_pos": dict(pos)}


def main():
    pivot = load_pivot()
    print(f"data: {len(pivot)} ticks")

    # Per-day evaluation
    by_day = {d: pivot[(pivot.index // TIMESTAMP_PER_DAY) == (d - DAYS[0])] for d in DAYS}

    sig_C = pivot[CIRCLE].diff().std()

    def make_v1_directional_unhedged(size, threshold=0):
        # Take direction based on sign of beta * ret_C lagged
        def fn(i, ret_lags, pos, mids):
            tgt = {CIRCLE: 0}
            for f, lag in FOLLOWERS.items():
                signal = BETAS[f] * ret_lags[lag]
                if abs(signal) < threshold:
                    tgt[f] = 0
                else:
                    tgt[f] = int(size * np.sign(signal))
            return tgt
        return fn

    def make_v2_directional_hedged(size, threshold=0, hedge=1.0):
        def fn(i, ret_lags, pos, mids):
            tgt = {}
            net_F = 0
            for f, lag in FOLLOWERS.items():
                signal = BETAS[f] * ret_lags[lag]
                if abs(signal) < threshold:
                    tgt[f] = 0
                else:
                    tgt[f] = int(size * np.sign(signal))
                net_F += tgt[f]
            tgt[CIRCLE] = int(-hedge * net_F)
            tgt[CIRCLE] = max(-POS_LIMIT, min(POS_LIMIT, tgt[CIRCLE]))
            return tgt
        return fn

    def make_v3_scaled_hedged(scale, hedge=1.0, max_size=POS_LIMIT):
        def fn(i, ret_lags, pos, mids):
            tgt = {}
            net_F = 0
            for f, lag in FOLLOWERS.items():
                signal = BETAS[f] * ret_lags[lag]
                # Position scaled by signal magnitude
                p = int(round(scale * signal))
                p = max(-max_size, min(max_size, p))
                tgt[f] = p
                net_F += p
            tgt[CIRCLE] = max(-POS_LIMIT, min(POS_LIMIT, int(-hedge * net_F)))
            return tgt
        return fn

    print(f"\n{'strategy':45s} {'d2':>9s} {'d3':>9s} {'d4':>9s} {'total':>9s} {'trades':>7s}")
    strategies = []

    for size in (1, 3, 5, 10):
        strategies.append((f"v1 directional unhedged size={size} thr=0", make_v1_directional_unhedged(size)))

    for size in (1, 3, 5, 10):
        for thr in (0, 0.3, 0.5, 1.0):
            strategies.append((f"v1 directional unhedged size={size} thr={thr}",
                              make_v1_directional_unhedged(size, threshold=thr)))

    for size in (3, 5, 10):
        for hedge in (0.5, 1.0, 1.5):
            strategies.append((f"v2 hedged size={size} hedge={hedge}",
                              make_v2_directional_hedged(size, hedge=hedge)))

    for scale in (5.0, 10.0, 20.0, 50.0):
        strategies.append((f"v3 scaled hedged scale={scale}",
                          make_v3_scaled_hedged(scale)))

    for name, fn in strategies:
        per_day_pnl = []
        for d in DAYS:
            sub = by_day[d]
            r = run_strategy(sub, name, fn)
            per_day_pnl.append(r["total_pnl"])
        total = sum(per_day_pnl)
        # Count trades over full data
        r = run_strategy(pivot, name, fn)
        print(f"{name:45s} {per_day_pnl[0]:>9.0f} {per_day_pnl[1]:>9.0f} {per_day_pnl[2]:>9.0f} "
              f"{total:>9.0f} {r['n_trades']:>7d}")


if __name__ == "__main__":
    main()
