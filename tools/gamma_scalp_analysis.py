"""Gamma-scalping diagnostic for VEV vouchers.

Three questions answered per (round, day, strike):

(1) Is realized σ > implied σ at a subsample rate that strips bid-ask bounce?
    Tick-level realized is dominated by quote noise. We compute σ at sample
    steps {1, 5, 20, 100, 500} ticks. The plateau where σ stabilizes is the
    debiased realized vol — the number to compare against IV.

(2) For each strike, what's the *theoretical* Γ-PnL minus Θ paid if you held
    1 voucher across the day, evaluated at that debiased σ_R?
        Γ-PnL ≈ ½·Γ(t)·ΔS²·dt summed
        Θ-PnL ≈  Θ(t)·dt summed
    Net should be > 0 iff σ_R > IV.

(3) What's the *practical* PnL after hedging cost? Simulate:
        - Buy 1 voucher at ask (or market mid + spread/2)
        - At every Δt-th tick, rebalance VFE delta. Each hedge crosses
          the VFE half-spread.
        - Track cumulative VFE-crossing costs against scalp gain.

Usage:
    python3 tools/gamma_scalp_analysis.py --round ROUND_4 --day 2
    python3 tools/gamma_scalp_analysis.py --round ROUND_4
    python3 tools/gamma_scalp_analysis.py --round ROUND_4 --day 2 --strike 5300
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import historical_loader
from utils import options as opts


SAMPLE_STEPS = [1, 5, 20, 100, 500]  # in ticks (1 tick = 100 timestamp units)
HEDGE_STEPS = [1, 5, 20, 100, 500]


def realized_sigma_subsampled(prices, timestamps, step):
    """Realized σ per √day from log-returns sampled every `step`-th observation."""
    if len(prices) < 2 * step:
        return np.nan
    p = prices[::step]
    t = timestamps[::step]
    if len(p) < 5:
        return np.nan
    dlog = np.diff(np.log(p))
    dt = np.diff(t) / opts.TIMESTAMP_PER_DAY
    valid = dt > 0
    if not valid.any():
        return np.nan
    var_per_day = np.mean(dlog[valid] ** 2 / dt[valid])
    return float(np.sqrt(var_per_day))


def build_panel(activities, day, tte_start):
    """Wide df: timestamp, S, S_bid, S_ask, C_K, mid_K, bid_K, ask_K per strike."""
    df = activities[activities["day"] == day].copy()
    ve = df[df["product"] == opts.UNDERLYING][[
        "timestamp", "mid_price", "bid_price_1", "ask_price_1"
    ]].rename(columns={
        "mid_price": "S", "bid_price_1": "S_bid", "ask_price_1": "S_ask",
    })

    panel = ve.copy()
    for K in opts.STRIKES:
        v = df[df["product"] == f"VEV_{K}"][[
            "timestamp", "mid_price", "bid_price_1", "ask_price_1"
        ]].rename(columns={
            "mid_price": f"C_{K}",
            "bid_price_1": f"Cbid_{K}",
            "ask_price_1": f"Cask_{K}",
        })
        panel = panel.merge(v, on="timestamp", how="left")

    panel = panel.dropna(subset=["S"]).reset_index(drop=True)
    panel["T"] = opts.time_to_expiry(panel["timestamp"].values, tte_start)
    return panel


def median_iv_per_strike(panel, strikes):
    out = {}
    for K in strikes:
        col = f"C_{K}"
        if col not in panel.columns:
            continue
        C = panel[col].values
        S = panel["S"].values
        T = panel["T"].values
        # Skip floor-pinned obs (no IV info there).
        mask = C > 0.5 + 1e-3
        if mask.sum() < 50:
            out[K] = np.nan
            continue
        iv = opts.implied_vol(C[mask], S[mask], np.full(mask.sum(), float(K)), T[mask])
        out[K] = float(np.nanmedian(iv))
    return out


def theoretical_scalp_pnl(panel, K, sigma_r):
    """Long 1 voucher, continuously delta-hedged at σ_R. No spread cost.
    Returns (gamma_pnl, theta_pnl, net)."""
    S = panel["S"].values
    T = panel["T"].values
    K_arr = np.full(len(panel), float(K))
    g = opts.greeks(S, K_arr, T, np.full(len(panel), sigma_r))
    gamma = g["gamma"]
    theta = g["theta"]

    dS = np.diff(S)
    dt = np.diff(panel["timestamp"].values) / opts.TIMESTAMP_PER_DAY
    gamma_step = 0.5 * gamma[:-1] * dS * dS  # per 1 voucher
    theta_step = theta[:-1] * dt
    return float(gamma_step.sum()), float(theta_step.sum()), float(
        gamma_step.sum() + theta_step.sum()
    )


def practical_scalp_pnl(panel, K, sigma_r, hedge_step):
    """Long 1 voucher entered at ask, delta-hedge VFE every `hedge_step` ticks
    by crossing the VFE half-spread. Returns dict with breakdown."""
    cask = panel.get(f"Cask_{K}")
    cbid = panel.get(f"Cbid_{K}")
    cmid = panel.get(f"C_{K}")
    if cask is None or cmid is None:
        return None
    if cmid.dropna().empty:
        return None

    # Use first valid timestamp as entry.
    valid_idx = cask.first_valid_index()
    if valid_idx is None:
        return None
    entry_C = float(panel[f"Cask_{K}"].iloc[valid_idx])
    entry_S = float(panel["S"].iloc[valid_idx])
    entry_T = float(panel["T"].iloc[valid_idx])
    if not np.isfinite(entry_C) or entry_C <= 0:
        return None

    # Greeks evaluated at σ_R.
    S_arr = panel["S"].values
    T_arr = panel["T"].values
    g = opts.greeks(
        S_arr, np.full(len(panel), float(K)), T_arr, np.full(len(panel), sigma_r)
    )
    delta = g["delta"]

    # VFE half-spread per timestamp.
    s_mid = panel["S"].values
    s_ask = panel["S_ask"].values if "S_ask" in panel.columns else None
    s_bid = panel["S_bid"].values if "S_bid" in panel.columns else None
    if s_ask is None or s_bid is None:
        half_spread = np.full(len(panel), 0.5)
    else:
        spread = s_ask - s_bid
        # Default 1-tick half-spread when book is missing
        half_spread = np.where(np.isfinite(spread) & (spread > 0), spread / 2.0, 0.5)

    # Hedge schedule: rebalance every hedge_step rows starting from valid_idx.
    hedge_idx = np.arange(valid_idx, len(panel), hedge_step)
    if len(hedge_idx) < 2:
        return None

    # Track delta position. Start hedge at entry to be delta-flat.
    pos_vfe = 0.0
    hedge_cost = 0.0
    n_hedges = 0
    for i in hedge_idx:
        target = -delta[i]  # short delta·S to flatten
        adj = target - pos_vfe
        # Fractional VFE shares are fine in this idealized analysis;
        # cost = |adj| * half_spread (crossing).
        if abs(adj) > 1e-9:
            hedge_cost += abs(adj) * half_spread[i]
            pos_vfe = target
            n_hedges += 1

    # End-of-day mark: voucher mid - entry_ask
    last_idx = panel[f"C_{K}"].last_valid_index()
    if last_idx is None:
        return None
    exit_C = float(panel[f"C_{K}"].iloc[last_idx])
    voucher_pnl = exit_C - entry_C  # held long 1 voucher

    # VFE position contributes via held delta * (S_now - S_at_last_hedge),
    # but if we hedged at every hedge_idx and rebalanced to target, the residual
    # is small. For simplicity, compute final mark-to-market on residual:
    # PnL_VFE = pos_vfe * (S_last - S_entry) approximately - already partially
    # captured in incremental rebalances; simpler: cumulate hedge gains.
    # We'll compute this properly:
    pnl_vfe = 0.0
    pos_vfe_run = 0.0
    last_S = entry_S
    for i in hedge_idx:
        target = -delta[i]
        # Realize PnL on existing position before changing
        pnl_vfe += pos_vfe_run * (s_mid[i] - last_S)
        pos_vfe_run = target
        last_S = s_mid[i]
    # Final close at last underlying mid
    pnl_vfe += pos_vfe_run * (s_mid[last_idx] - last_S)

    return {
        "entry_C": entry_C,
        "exit_C": exit_C,
        "voucher_pnl": voucher_pnl,
        "vfe_pnl": pnl_vfe,
        "hedge_cost": hedge_cost,
        "n_hedges": n_hedges,
        "net": voucher_pnl + pnl_vfe - hedge_cost,
    }


def analyze_day(round_name, day, strikes_filter=None):
    bundle = historical_loader.load_round(round_name)
    tte_start = opts.tte_for_day(round_name, day)
    panel = build_panel(bundle["activities"], day, tte_start)
    if panel.empty:
        print(f"\n{round_name} day {day}: no underlying data, skipping")
        return

    print(f"\n{'=' * 78}")
    print(f"{round_name} day {day}  ·  TTE@start={tte_start:.1f}d  ·  N={len(panel)} rows")
    print('=' * 78)

    # -------- (1) Subsampled realized σ --------
    S = panel["S"].values
    ts = panel["timestamp"].values
    print("\n[1] Subsampled realized σ on VFE underlying")
    print(f"    {'step':>6}  {'σ (per√day)':>12}  {'note':<40}")
    rv = {}
    for step in SAMPLE_STEPS:
        s = realized_sigma_subsampled(S, ts, step)
        rv[step] = s
        note = ""
        if step == 1:
            note = "raw tick (bid-ask bounce inflated)"
        elif step >= 100:
            note = "should plateau here if MR-noise dominates"
        print(f"    {step:>6}  {s:>12.5f}  {note:<40}")

    # Pick a debiased estimate: median across steps 100 and 500 (or 20 if 500 unavailable).
    candidates = [rv[s] for s in (100, 500) if not np.isnan(rv.get(s, np.nan))]
    if not candidates:
        candidates = [rv[s] for s in (20,) if not np.isnan(rv.get(s, np.nan))]
    sigma_r = float(np.mean(candidates)) if candidates else np.nan
    print(f"\n    σ_R (debiased, mean of step≥100): {sigma_r:.5f}")

    # -------- (2) Median IV per strike --------
    strikes = [K for K in opts.STRIKES if (strikes_filter is None or K in strikes_filter)]
    iv_med = median_iv_per_strike(panel, strikes)

    print("\n[2] Median IV vs σ_R per strike")
    print(f"    {'K':>6}  {'IV_med':>9}  {'σ_R':>9}  {'gap (σ_R−IV)':>14}  {'sign':>6}")
    gaps = {}
    for K in strikes:
        ivm = iv_med.get(K, np.nan)
        if np.isnan(ivm):
            print(f"    {K:>6}  {'(n/a)':>9}  {sigma_r:>9.5f}  {'—':>14}  {'—':>6}")
            continue
        gap = sigma_r - ivm
        gaps[K] = gap
        sign = "long Γ" if gap > 0 else "short Γ" if gap < 0 else "—"
        print(f"    {K:>6}  {ivm:>9.5f}  {sigma_r:>9.5f}  {gap:>+14.5f}  {sign:>6}")

    # -------- (3) Theoretical scalp PnL per strike --------
    print(f"\n[3] Theoretical Γ-Θ PnL across day at σ_R={sigma_r:.5f}  (per 1 voucher held)")
    print(f"    {'K':>6}  {'Γ-PnL':>10}  {'Θ-PnL':>10}  {'net':>10}  {'×300':>10}")
    for K in strikes:
        gp, tp, net = theoretical_scalp_pnl(panel, K, sigma_r)
        print(f"    {K:>6}  {gp:>+10.2f}  {tp:>+10.2f}  {net:>+10.2f}  {net * 300:>+10.0f}")

    # -------- (4) Practical scalp PnL with VFE hedge cost --------
    print(f"\n[4] Practical scalp PnL (long 1 voucher entered at ask, hedged at VFE half-spread)")
    print(f"    {'K':>5} {'h-step':>7} {'voucher':>9} {'VFE':>9} {'hcost':>8} "
          f"{'net':>9} {'×300':>9} {'#hedges':>9}")
    for K in strikes:
        ivm = iv_med.get(K, np.nan)
        if np.isnan(ivm):
            continue
        for hs in HEDGE_STEPS:
            res = practical_scalp_pnl(panel, K, sigma_r, hs)
            if res is None:
                continue
            print(f"    {K:>5} {hs:>7} {res['voucher_pnl']:>+9.2f} "
                  f"{res['vfe_pnl']:>+9.2f} {res['hedge_cost']:>8.2f} "
                  f"{res['net']:>+9.2f} {res['net'] * 300:>+9.0f} {res['n_hedges']:>9}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="ROUND_4")
    ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--strike", type=int, nargs="*", default=None,
                    help="optional list of strikes to filter")
    args = ap.parse_args()

    days = [args.day] if args.day is not None else sorted(
        historical_loader.get_available_days(args.round)
    )
    for d in days:
        analyze_day(args.round, d, args.strike)


if __name__ == "__main__":
    main()
