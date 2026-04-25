"""Vol-surface arbitrage and regime diagnostics.

Subcommands:
  realized   realized σ of underlying at multiple sampling frequencies vs IV
  sticky     test whether IV(K) or IV(m) tracks S — sticky-strike vs sticky-delta
  convex     call-surface convexity test (butterfly arb): C(K-h) − 2C(K) + C(K+h) ≥ 0
  monotone   call monotonicity test: C(K1) > C(K2) for K1 < K2

Convention: IV is per-√day to match utils/options.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import historical_loader
from utils import options as opts


LIQUID_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]


def _panel(activities, day, tte):
    df = activities[activities["day"] == day].copy()
    products = [opts.UNDERLYING] + [f"VEV_{K}" for K in opts.STRIKES]
    df = df[df["product"].isin(products)]
    keep = ["timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"]
    df = df[keep].rename(columns={
        "mid_price": "mid", "bid_price_1": "bid", "ask_price_1": "ask",
    })
    pivot = df.pivot(index="timestamp", columns="product",
                     values=["mid", "bid", "ask"])
    pivot.columns = [f"{prod}__{field}" for field, prod in pivot.columns]
    pivot = pivot.sort_index()
    pivot["T"] = np.maximum(tte - pivot.index.values / opts.TIMESTAMP_PER_DAY, 1e-6)
    return pivot


# -----------------------------------------------------------------
# realized
# -----------------------------------------------------------------

def cmd_realized(args):
    """Realized σ at multiple sampling frequencies vs mean IV."""
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]

    intervals = [1, 5, 10, 50, 100, 500, 1000]  # in 100-ts units
    days = [args.day] if args.day is not None else sorted(acts["day"].unique())

    print(f"Realized vol of {opts.UNDERLYING} mid (per √day units, ×1e-2)")
    print(f"  Interval is in 100-ts ticks. Day has 10000 ticks.")
    print(f"  {'day':>4} {'IV_mean':>9} " +
          "".join(f"{'σ_'+str(s):>9}" for s in intervals))

    for day in days:
        S = acts[(acts["day"] == day) & (acts["product"] == opts.UNDERLYING)] \
            .sort_values("timestamp")["mid_price"].dropna().to_numpy()
        if len(S) < 100:
            continue
        log_S = np.log(S)
        row = []
        for k in intervals:
            r = log_S[k::k] - log_S[:-k:k]
            r = r[np.isfinite(r)]
            if len(r) < 5:
                row.append(float("nan"))
                continue
            std_per = r.std(ddof=1)
            n_per_day = 10000 / k
            sigma_perday = std_per * np.sqrt(n_per_day)
            row.append(sigma_perday)

        # Mean IV across liquid strikes
        tte = opts.tte_for_day(args.round, day)
        panel = _panel(acts, day, tte)
        S_arr = panel[f"{opts.UNDERLYING}__mid"].to_numpy()
        T_arr = panel["T"].to_numpy()
        ivs = []
        for K in LIQUID_STRIKES:
            col = f"VEV_{K}__mid"
            if col not in panel.columns:
                continue
            iv = opts.implied_vol(panel[col].to_numpy(), S_arr, float(K), T_arr)
            iv = iv[np.isfinite(iv)]
            ivs.append(iv.mean())
        iv_mean = float(np.mean(ivs)) if ivs else float("nan")

        print(f"  {day:>4} {iv_mean*1e2:>8.4f}  " +
              "".join(f"{r*1e2:>8.4f}" for r in row))

    print()
    print("Notes:")
    print("  · σ_1   uses every tick (100-ts spacing). Inflated by bid-ask bounce.")
    print("  · σ_50+ samples every ≥5000 ts → near-true realized vol.")
    print("  · IV_mean averaged across strikes 5000–5500.")
    print("  · IV >> realized → surface rich → sell vol; IV << realized → buy vol.")


# -----------------------------------------------------------------
# sticky
# -----------------------------------------------------------------

def cmd_sticky(args):
    """Test sticky-strike (IV at K invariant to S) vs sticky-delta (IV at m invariant)."""
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _panel(acts, args.day, tte)

    S = panel[f"{opts.UNDERLYING}__mid"].to_numpy()
    T = panel["T"].to_numpy()

    print(f"Sticky-strike vs sticky-delta · day {args.day} · TTE@start={tte}")
    print(f"  N = {len(S)} ticks · S range [{np.nanmin(S):.0f}, {np.nanmax(S):.0f}]"
          f" std {np.nanstd(S):.2f}")
    print()
    print("Per-strike: regress IV_K on S → slope dIV/dS, correlation, σ(IV) explained")
    print(f"  {'K':>5} {'IV_mean':>9} {'σ_IV':>9} "
          f"{'dIV/dS·1e6':>12} {'corr(IV,S)':>11} {'R²':>7}")
    by_K = {}
    for K in LIQUID_STRIKES:
        col = f"VEV_{K}__mid"
        if col not in panel.columns:
            continue
        iv = opts.implied_vol(panel[col].to_numpy(), S, float(K), T)
        ok = np.isfinite(iv) & np.isfinite(S)
        if ok.sum() < 100:
            continue
        s, i = S[ok], iv[ok]
        slope, intercept = np.polyfit(s, i, 1)
        corr = np.corrcoef(s, i)[0, 1]
        r2 = corr ** 2
        by_K[K] = (i, s, slope, corr)
        print(f"  {K:>5} {i.mean()*100:>8.4f} {i.std()*100:>8.4f}  "
              f"{slope*1e6:>+11.3f}  {corr:>+10.3f}  {r2:>6.3f}")

    # Now interpolate to constant moneyness m and re-do.
    # For each tick, we have a smile across strikes; pick a fixed m and
    # interpolate IV(m) from the fitted quadratic.
    print()
    print("At fixed normalized moneyness m=ln(K/S)/√T:")
    print(f"  {'m':>7} {'IV_mean':>9} {'σ_IV':>9} "
          f"{'dIV/dS·1e6':>12} {'corr(IV,S)':>11} {'R²':>7}")
    # Collect (m, IV) pairs per tick at all strikes; fit quadratic; sample.
    iv_grid = np.full((len(panel), len(LIQUID_STRIKES)), np.nan)
    for j, K in enumerate(LIQUID_STRIKES):
        col = f"VEV_{K}__mid"
        if col not in panel.columns:
            continue
        iv_grid[:, j] = opts.implied_vol(panel[col].to_numpy(), S, float(K), T)
    m_grid = np.log(np.array(LIQUID_STRIKES, float)[None, :] /
                    S[:, None]) / np.sqrt(T[:, None])
    iv_at_m = {}
    for m_target in (-0.02, -0.01, 0.0, 0.01, 0.02):
        out = np.full(len(panel), np.nan)
        for i in range(len(panel)):
            mask = np.isfinite(iv_grid[i]) & np.isfinite(m_grid[i])
            if mask.sum() < 3:
                continue
            X = np.column_stack([np.ones(mask.sum()),
                                 m_grid[i, mask], m_grid[i, mask] ** 2])
            try:
                beta, *_ = np.linalg.lstsq(X, iv_grid[i, mask], rcond=None)
            except np.linalg.LinAlgError:
                continue
            out[i] = beta[0] + beta[1] * m_target + beta[2] * m_target ** 2
        ok = np.isfinite(out) & np.isfinite(S)
        if ok.sum() < 100:
            continue
        s, i = S[ok], out[ok]
        slope, _ = np.polyfit(s, i, 1)
        corr = np.corrcoef(s, i)[0, 1]
        iv_at_m[m_target] = (i, s, slope, corr)
        print(f"  {m_target:>+7.3f} {i.mean()*100:>8.4f} {i.std()*100:>8.4f}  "
              f"{slope*1e6:>+11.3f}  {corr:>+10.3f}  {corr**2:>6.3f}")

    print()
    print("Interpretation:")
    print("  · sticky-strike: IV(K) flat as S moves → slope and corr ≈ 0 in K-table,")
    print("    big slope in m-table (because at fixed m, K shifts as S moves).")
    print("  · sticky-delta: opposite — IV(m) flat, IV(K) drifts.")
    print("  · neither: surface re-shapes.")


# -----------------------------------------------------------------
# convex (butterfly arb)
# -----------------------------------------------------------------

def cmd_convex(args):
    """Call-surface convexity: C(K-h) - 2C(K) + C(K+h) must be ≥ 0.

    We compute it for evenly-spaced strike triplets (h=100), per-tick, using:
      - mid prices: 'is the smile convex on average?'
      - bid/ask combo for an executable butterfly:
          buy K-h at ask, sell K twice at bid, buy K+h at ask
          payoff = C_ask(K-h) - 2*C_bid(K) + C_ask(K+h)
        if this is ≤ 0 you have a free butterfly.
    """
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _panel(acts, args.day, tte)

    triplets = [
        (5000, 5100, 5200), (5100, 5200, 5300),
        (5200, 5300, 5400), (5300, 5400, 5500),
    ]

    print(f"Call butterfly convexity · day {args.day} · TTE@start={tte}d")
    print("  Mid butterfly = C(K-h) − 2C(K) + C(K+h). Must be > 0 (convex surface).")
    print(f"  {'triplet':>20} {'mid_mean':>9} {'mid_min':>8} {'<=0_pct':>8} "
          f"{'exec_mean':>10} {'exec_min':>9} {'exec<=0_pct':>11}")

    for K1, K2, K3 in triplets:
        c1 = panel[f"VEV_{K1}__mid"].to_numpy()
        c2 = panel[f"VEV_{K2}__mid"].to_numpy()
        c3 = panel[f"VEV_{K3}__mid"].to_numpy()
        bm = c1 - 2 * c2 + c3
        bm_ok = bm[np.isfinite(bm)]

        # Executable cost: cost to BUILD this long butterfly (long wings, short body × 2)
        # cost = ASK(K1) + ASK(K3) - 2*BID(K2). If cost < 0 → free money.
        a1 = panel[f"VEV_{K1}__ask"].to_numpy()
        a3 = panel[f"VEV_{K3}__ask"].to_numpy()
        b2 = panel[f"VEV_{K2}__bid"].to_numpy()
        exec_cost = a1 - 2 * b2 + a3
        exec_ok = exec_cost[np.isfinite(exec_cost)]

        print(f"  {f'{K1}/{K2}/{K3}':>20} "
              f"{bm_ok.mean():>8.3f} {bm_ok.min():>8.3f} "
              f"{(bm_ok <= 0).mean()*100:>7.2f}% "
              f"{exec_ok.mean():>9.3f} {exec_ok.min():>9.3f} "
              f"{(exec_ok <= 0).mean()*100:>10.2f}%")

    print()
    print("  · mid butterfly = the smooth-surface check. Positive = surface convex at K.")
    print("    Lower number = the smile dips at K (K is locally cheap).")
    print("  · exec butterfly = the actual cash you'd pay to OPEN the long fly.")
    print("    Negative or zero = direct arb. Positive = cost; lower is cheaper to enter.")


# -----------------------------------------------------------------
# monotone
# -----------------------------------------------------------------

def cmd_monotone(args):
    """Test C(K_low) > C(K_high) for K_low < K_high. Violations = arb."""
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _panel(acts, args.day, tte)

    print(f"Call monotonicity · day {args.day} · TTE@start={tte}d")
    print(f"  Adjacent strike pairs: C(K1) - C(K2) where K1 < K2. Must be > 0.")
    print(f"  exec_check: ask(K2) - bid(K1) ≥ 0?  Negative = sell K1 buy K2 risk-free.")
    print(f"  {'pair':>16} {'mid_diff_mean':>14} {'mid_diff_min':>13} "
          f"{'mid_inv_pct':>12} {'exec_min':>9} {'exec<0_pct':>11}")

    pairs = list(zip(LIQUID_STRIKES[:-1], LIQUID_STRIKES[1:]))
    for K1, K2 in pairs:
        c1 = panel[f"VEV_{K1}__mid"].to_numpy()
        c2 = panel[f"VEV_{K2}__mid"].to_numpy()
        d = c1 - c2
        d_ok = d[np.isfinite(d)]
        a2 = panel[f"VEV_{K2}__ask"].to_numpy()
        b1 = panel[f"VEV_{K1}__bid"].to_numpy()
        exec_diff = b1 - a2  # if positive, you can sell K1 at bid and buy K2 at ask cheaper
        exec_ok = exec_diff[np.isfinite(exec_diff)]
        print(f"  {f'{K1}/{K2}':>16} "
              f"{d_ok.mean():>13.3f} {d_ok.min():>13.3f} "
              f"{(d_ok < 0).mean()*100:>11.2f}% "
              f"{exec_ok.max():>9.3f} {(exec_ok > 0).mean()*100:>10.2f}%")

    print()
    print("  exec column polarity flipped: bid(K1) - ask(K2) > 0 means sell low-strike at "
          "bid, buy high-strike at ask, pocket the difference (and own a vertical spread for free).")


# -----------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description="Vol-surface arb / regime diagnostics.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp, require_day=True):
        sp.add_argument("--round", default="ROUND_3")
        if require_day:
            sp.add_argument("--day", type=int, required=True)
        else:
            sp.add_argument("--day", type=int)
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("realized", help="realized σ at multiple frequencies vs IV")
    _common(sp, require_day=False)
    sp.set_defaults(fn=cmd_realized)

    sp = sub.add_parser("sticky", help="sticky-strike vs sticky-delta test")
    _common(sp)
    sp.set_defaults(fn=cmd_sticky)

    sp = sub.add_parser("convex", help="call butterfly convexity (no-arb)")
    _common(sp)
    sp.set_defaults(fn=cmd_convex)

    sp = sub.add_parser("monotone", help="call monotonicity (no-arb)")
    _common(sp)
    sp.set_defaults(fn=cmd_monotone)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
