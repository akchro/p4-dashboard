"""IV mean-reversion analysis — does fading deviations from a frozen smile work?

Inspired by the prior-round champion's IV scalping strategy:
  1. Fit a *static* quadratic smile IV(m) once from historical data
     where m = ln(K/S)/√TTE (normalized log-moneyness).
  2. At each timestamp compute theoretical price from that smile, then
     residual = market_mid - theo_price.
  3. EMA-smooth the residual to get a "drift" mean; the deviation from drift
     is the trade signal — fade large deviations, expect reversion.

This script measures whether the deviations actually do revert.

Outputs per (round, day, strike):
  (1) Static smile coefficients fitted on R3 days 0/1/2 combined (or --fit-on)
  (2) Distribution of theo_diff (residual)
  (3) Autocorrelation + AR(1) half-life of theo_diff
  (4) "Reversion test": after |theo_diff - EMA20(theo_diff)| > threshold,
       what's the average forward return over N ticks? Positive = reversion.
  (5) Simple backtest: fade deviations > THR_OPEN=0.5 @ best_bid/ask, close
       at THR_CLOSE=0 with VFE half-spread hedge cost.

Usage:
    python3 tools/iv_meanrev_analysis.py --fit-on ROUND_3 --test-on ROUND_4
    python3 tools/iv_meanrev_analysis.py --test-on ROUND_4 --day 2 --strikes 5300 5400
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import historical_loader
from utils import options as opts


FLOOR_PRICE = 0.5
FLOOR_EPS = 1e-3

THEO_NORM_WINDOW = 20      # EMA window for residual drift (champion: 20)
IV_SCALPING_WINDOW = 100   # EMA window for residual volatility (champion: 100)
IV_SCALPING_THR = 0.7      # Only trade when regime vol exceeds this (champion)
THR_OPEN = 0.5             # Open-fade threshold (champion: 0.5 in price)
THR_CLOSE = 0.0            # Close at mean


def _build_panel(activities, day, tte_start):
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


def fit_static_smile(round_name, days_filter=None):
    """Fit IV(m) = c2·m² + c1·m + c0 across all days/strikes/timestamps in `round_name`.
    Returns the np.poly1d coefficients in descending order (c2, c1, c0)."""
    bundle = historical_loader.load_round(round_name)
    days = sorted(historical_loader.get_available_days(round_name))
    if days_filter is not None:
        days = [d for d in days if d in days_filter]

    all_m = []
    all_iv = []
    for d in days:
        tte_start = opts.tte_for_day(round_name, d)
        panel = _build_panel(bundle["activities"], d, tte_start)
        if panel.empty:
            continue
        S = panel["S"].values
        T = panel["T"].values
        for K in opts.STRIKES:
            col = f"C_{K}"
            if col not in panel.columns:
                continue
            C = panel[col].values
            mask = (
                np.isfinite(C) & (C > FLOOR_PRICE + FLOOR_EPS) &
                np.isfinite(S) & np.isfinite(T) & (T > 1e-6)
            )
            if mask.sum() < 50:
                continue
            iv = opts.implied_vol(C[mask], S[mask], np.full(mask.sum(), float(K)), T[mask])
            ok = np.isfinite(iv)
            if ok.sum() < 50:
                continue
            m = np.log(K / S[mask][ok]) / np.sqrt(T[mask][ok])
            all_m.append(m)
            all_iv.append(iv[ok])

    m_all = np.concatenate(all_m)
    iv_all = np.concatenate(all_iv)
    coeffs = np.polyfit(m_all, iv_all, 2)  # returns [c2, c1, c0]
    return coeffs, len(m_all)


def static_iv(coeffs, K, S, T):
    """Evaluate fitted IV at given S/T per strike K."""
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    m = np.log(K / S) / np.sqrt(T)
    return np.poly1d(coeffs)(m)


def static_call(coeffs, K, S, T):
    """Theoretical call price from the static smile."""
    iv = static_iv(coeffs, K, S, T)
    return opts.bs_call(S, np.full_like(S, float(K)), T, iv)


def ema(series, window):
    """Match champion's EMA: alpha = 2/(window+1), seeded at 0."""
    alpha = 2.0 / (window + 1.0)
    out = np.zeros_like(series, dtype=float)
    state = 0.0
    for i, v in enumerate(series):
        if not np.isfinite(v):
            out[i] = state
            continue
        state = alpha * v + (1 - alpha) * state
        out[i] = state
    return out


def autocorr_at_lags(series, lags):
    s = pd.Series(series).dropna()
    if len(s) < max(lags) + 5:
        return {lag: np.nan for lag in lags}
    return {lag: s.autocorr(lag=lag) for lag in lags}


def fit_ar1_halflife(series):
    s = pd.Series(series).dropna()
    if len(s) < 50:
        return np.nan, np.nan
    y = s.values[1:]
    x = s.values[:-1]
    var_x = np.var(x)
    if var_x == 0:
        return np.nan, np.nan
    beta = np.cov(x, y, ddof=0)[0, 1] / var_x
    if beta <= 0 or beta >= 1:
        return beta, np.nan
    return beta, np.log(0.5) / np.log(beta)


def reversion_curve(theo_diff, drift, threshold, horizons):
    """For each row where |theo_diff - drift| > threshold, measure forward
    Δ(theo_diff) at each horizon. If reversion holds, the sign is opposite to
    the trigger sign, so the *signed* return after a positive deviation should
    be negative.

    Returns dict {horizon: avg(-sign(dev) * future_return)} — positive = reverts.
    """
    dev = theo_diff - drift
    valid = np.isfinite(dev)
    flagged = valid & (np.abs(dev) > threshold)
    n = flagged.sum()
    out = {"n_triggered": int(n)}
    for h in horizons:
        gains = []
        idxs = np.where(flagged)[0]
        for i in idxs:
            if i + h >= len(theo_diff):
                continue
            future = theo_diff[i + h] - theo_diff[i]
            gain = -np.sign(dev[i]) * future
            if np.isfinite(gain):
                gains.append(gain)
        if gains:
            out[h] = float(np.mean(gains))
        else:
            out[h] = np.nan
    return out


def backtest_fade(panel, K, coeffs, position_cap=300, delta_hedge=True,
                   hedge_step=100):
    """Champion-corrected fade backtest with VFE delta hedging.

    Champion checks (best_bid − theo) − EMA20(mid − theo) > threshold
    rather than (mid − theo) > threshold. That's the fill-edge after
    spread crossing. If we sell at best_bid, we want best_bid > theo +
    EMA20-drift + threshold (so even after crossing, we're at a richer-
    than-typical price).

    - Open short (sell at bid) when (bid − theo) > drift + THR_OPEN
    - Open long (buy at ask) when (ask − theo) < drift − THR_OPEN
    - Close at THR_CLOSE crossing
    - Optionally rebalance VFE delta every `hedge_step` rows
    """
    bid_col, ask_col, mid_col = f"Cbid_{K}", f"Cask_{K}", f"C_{K}"
    if mid_col not in panel.columns:
        return None
    mid = panel[mid_col].values
    bid = panel[bid_col].values if bid_col in panel.columns else mid - 0.5
    ask = panel[ask_col].values if ask_col in panel.columns else mid + 0.5

    S = panel["S"].values
    T = panel["T"].values
    s_bid = panel.get("S_bid", pd.Series(S - 0.5)).values
    s_ask = panel.get("S_ask", pd.Series(S + 0.5)).values
    s_half = np.where(
        np.isfinite(s_ask) & np.isfinite(s_bid) & (s_ask - s_bid > 0),
        (s_ask - s_bid) / 2.0, 0.5,
    )

    # Theoretical price from static smile (used for residual signal)
    iv_static = static_iv(coeffs, K, S, T)
    theo = opts.bs_call(S, np.full_like(S, float(K)), T, iv_static)
    # But hedge with MARKET-IV delta (more accurate)
    iv_market = opts.implied_vol(mid, S, np.full_like(S, float(K)), T)
    iv_for_delta = np.where(np.isfinite(iv_market), iv_market, iv_static)
    g = opts.greeks(S, np.full_like(S, float(K)), T, iv_for_delta)
    delta_theo = g["delta"]

    valid = (
        np.isfinite(mid) & (mid > FLOOR_PRICE + FLOOR_EPS) &
        np.isfinite(theo) & np.isfinite(S) & np.isfinite(delta_theo)
    )
    if valid.sum() < 200:
        return None
    theo_diff = np.where(valid, mid - theo, np.nan)
    # Edge = (best_bid - theo) for selling, (theo - best_ask) for buying
    sell_edge = np.where(valid, bid - theo, np.nan)  # > drift + thr → sell rich
    buy_edge = np.where(valid, ask - theo, np.nan)   # < drift - thr → buy cheap
    finite_mask = np.isfinite(theo_diff)
    drift = ema(np.where(finite_mask, theo_diff, 0.0), THEO_NORM_WINDOW)
    abs_dev = np.where(finite_mask, np.abs(theo_diff - drift), 0.0)
    regime_vol = ema(abs_dev, IV_SCALPING_WINDOW)

    pos = 0           # voucher position
    pos_vfe = 0.0     # underlying hedge
    cash = 0.0
    n_trades = 0
    n_hedges = 0
    n_in_regime = 0
    hedge_cost_total = 0.0
    closing_pnls = []
    open_price = None
    open_side = 0

    last_hedge_idx = -hedge_step  # ensure first valid row triggers hedge

    for i in range(len(panel)):
        if not finite_mask[i]:
            continue

        in_regime = regime_vol[i] >= IV_SCALPING_THR
        if not in_regime:
            if pos != 0:
                cash += pos * mid[i]
                if open_price is not None:
                    closing_pnls.append(open_side * (mid[i] - open_price))
                pos = 0
                open_price = None
                open_side = 0
                n_trades += 1
            # rebalance hedge to 0 too
            if delta_hedge and pos_vfe != 0:
                target = 0.0
                cash -= (target - pos_vfe) * S[i]  # close at mid for simplicity
                pos_vfe = 0.0
            continue

        n_in_regime += 1
        dev = theo_diff[i] - drift[i]
        # Champion's effective edge calculation
        sell_signal = sell_edge[i] - drift[i]  # >= THR_OPEN ⇒ sell at bid is rich
        buy_signal = buy_edge[i] - drift[i]    # <= -THR_OPEN ⇒ buy at ask is cheap

        if sell_signal >= THR_OPEN and pos > -position_cap:
            qty = min(position_cap - abs(pos), 50)
            cash += qty * bid[i]
            pos -= qty
            if open_price is None:
                open_price = bid[i]
                open_side = -1
            n_trades += 1
        elif buy_signal <= -THR_OPEN and pos < position_cap:
            qty = min(position_cap - abs(pos), 50)
            cash -= qty * ask[i]
            pos += qty
            if open_price is None:
                open_price = ask[i]
                open_side = 1
            n_trades += 1
        elif pos > 0 and dev > THR_CLOSE:
            cash += pos * bid[i]
            if open_price is not None:
                closing_pnls.append(open_side * (bid[i] - open_price))
            pos = 0
            open_price = None
            open_side = 0
            n_trades += 1
        elif pos < 0 and dev < -THR_CLOSE:
            cash -= -pos * ask[i]
            if open_price is not None:
                closing_pnls.append(open_side * (ask[i] - open_price))
            pos = 0
            open_price = None
            open_side = 0
            n_trades += 1

        # Delta hedge VFE every `hedge_step` rows
        if delta_hedge and (i - last_hedge_idx) >= hedge_step:
            target = -pos * delta_theo[i]
            adj = target - pos_vfe
            if abs(adj) > 1e-6:
                # Crossing direction
                px = s_ask[i] if adj > 0 else s_bid[i]
                if not np.isfinite(px):
                    px = S[i] + (0.5 if adj > 0 else -0.5)
                cash -= adj * px
                pos_vfe = target
                hedge_cost_total += abs(adj) * s_half[i]
                n_hedges += 1
                last_hedge_idx = i

    # Mark to market at end on last valid row
    last_idx = np.where(finite_mask)[0][-1]
    last_mid = mid[last_idx]
    last_S = S[last_idx]
    mtm = cash + pos * last_mid + pos_vfe * last_S

    return {
        "pnl": float(mtm),
        "n_trades": n_trades,
        "n_hedges": n_hedges,
        "n_in_regime": n_in_regime,
        "closed_pnl_mean": float(np.mean(closing_pnls)) if closing_pnls else np.nan,
        "closed_pnl_std": float(np.std(closing_pnls)) if closing_pnls else np.nan,
        "n_closed": len(closing_pnls),
        "final_pos_v": pos,
        "final_pos_vfe": float(pos_vfe),
        "hedge_cost": float(hedge_cost_total),
    }


def backtest_iv_zscore(panel, K, position_cap=300, lookback=500,
                        z_open=2.0, z_close=0.5, hedge_step=100, delta_hedge=True):
    """Fade per-strike IV against its own rolling mean (no smile dependency).

    - Compute rolling-window IV mean and std over the prior `lookback` ticks
    - Z-score = (iv_t - μ) / σ
    - When |Z| > z_open: fade (sell if IV rich, buy if IV cheap)
    - Close when |Z| < z_close
    - Optional VFE delta hedge using market IV per-tick delta
    """
    bid_col, ask_col, mid_col = f"Cbid_{K}", f"Cask_{K}", f"C_{K}"
    if mid_col not in panel.columns:
        return None
    mid = panel[mid_col].values
    bid = panel[bid_col].values if bid_col in panel.columns else mid - 0.5
    ask = panel[ask_col].values if ask_col in panel.columns else mid + 0.5

    S = panel["S"].values
    T = panel["T"].values
    s_bid = panel.get("S_bid", pd.Series(S - 0.5)).values
    s_ask = panel.get("S_ask", pd.Series(S + 0.5)).values
    s_half = np.where(
        np.isfinite(s_ask) & np.isfinite(s_bid) & (s_ask - s_bid > 0),
        (s_ask - s_bid) / 2.0, 0.5,
    )

    # Per-tick IV from market
    iv = opts.implied_vol(mid, S, np.full_like(S, float(K)), T)
    g = opts.greeks(S, np.full_like(S, float(K)), T, np.where(np.isfinite(iv), iv, 0.013))
    delta_t = g["delta"]

    # Rolling Z-score
    iv_s = pd.Series(iv)
    mu = iv_s.rolling(lookback, min_periods=lookback // 2).mean().values
    sd = iv_s.rolling(lookback, min_periods=lookback // 2).std().values
    z = (iv - mu) / np.where(sd > 0, sd, np.nan)

    pos = 0
    pos_vfe = 0.0
    cash = 0.0
    n_trades = 0
    n_hedges = 0
    hedge_cost_total = 0.0
    closing_pnls = []
    open_price = None
    open_side = 0
    last_hedge_idx = -hedge_step

    for i in range(len(panel)):
        if not np.isfinite(z[i]) or not np.isfinite(mid[i]) or mid[i] <= FLOOR_PRICE + FLOOR_EPS:
            continue

        if z[i] > z_open and pos > -position_cap:
            qty = min(position_cap - abs(pos), 50)
            cash += qty * bid[i]
            pos -= qty
            if open_price is None:
                open_price = bid[i]
                open_side = -1
            n_trades += 1
        elif z[i] < -z_open and pos < position_cap:
            qty = min(position_cap - abs(pos), 50)
            cash -= qty * ask[i]
            pos += qty
            if open_price is None:
                open_price = ask[i]
                open_side = 1
            n_trades += 1
        elif pos > 0 and z[i] > -z_close:  # close long when Z back near 0
            cash += pos * bid[i]
            if open_price is not None:
                closing_pnls.append(open_side * (bid[i] - open_price))
            pos = 0
            open_price = None
            open_side = 0
            n_trades += 1
        elif pos < 0 and z[i] < z_close:
            cash -= -pos * ask[i]
            if open_price is not None:
                closing_pnls.append(open_side * (ask[i] - open_price))
            pos = 0
            open_price = None
            open_side = 0
            n_trades += 1

        if delta_hedge and (i - last_hedge_idx) >= hedge_step and np.isfinite(delta_t[i]):
            target = -pos * delta_t[i]
            adj = target - pos_vfe
            if abs(adj) > 1e-6:
                px = s_ask[i] if adj > 0 else s_bid[i]
                if not np.isfinite(px):
                    px = S[i] + (0.5 if adj > 0 else -0.5)
                cash -= adj * px
                pos_vfe = target
                hedge_cost_total += abs(adj) * s_half[i]
                n_hedges += 1
                last_hedge_idx = i

    last_idx = len(panel) - 1
    while last_idx > 0 and not np.isfinite(mid[last_idx]):
        last_idx -= 1
    mtm = cash + pos * mid[last_idx] + pos_vfe * S[last_idx]

    return {
        "pnl": float(mtm),
        "n_trades": n_trades,
        "n_hedges": n_hedges,
        "n_closed": len(closing_pnls),
        "closed_pnl_mean": float(np.mean(closing_pnls)) if closing_pnls else np.nan,
        "hedge_cost": float(hedge_cost_total),
        "final_pos_v": pos,
        "final_pos_vfe": float(pos_vfe),
        "z_max": float(np.nanmax(np.abs(z))),
    }


def analyze_day(round_name, day, coeffs, strikes_filter=None):
    bundle = historical_loader.load_round(round_name)
    tte_start = opts.tte_for_day(round_name, day)
    panel = _build_panel(bundle["activities"], day, tte_start)
    if panel.empty:
        print(f"\n{round_name} day {day}: no data, skip")
        return

    print(f"\n{'=' * 78}")
    print(f"{round_name} day {day}  ·  TTE={tte_start:.1f}d  ·  N={len(panel)}")
    print('=' * 78)

    strikes = [K for K in opts.STRIKES if (strikes_filter is None or K in strikes_filter)]

    # ---- (a) Residual stats per strike ----
    print(f"\n[a] theo_diff = market_mid − static_smile_price  ·  per-strike summary")
    print(f"    {'K':>5} {'n':>5} {'mean':>8} {'std':>7} {'p5':>7} {'p95':>7} "
          f"{'ρ(1)':>7} {'ρ(20)':>7} {'AR(1)β':>9} {'half-life':>11}")

    rev_results = {}
    for K in strikes:
        col = f"C_{K}"
        if col not in panel.columns:
            continue
        mid = panel[col].values
        S = panel["S"].values
        T = panel["T"].values
        theo = static_call(coeffs, K, S, T)
        valid = np.isfinite(mid) & (mid > FLOOR_PRICE + FLOOR_EPS) & np.isfinite(theo)
        if valid.sum() < 100:
            continue
        td = (mid - theo)[valid]
        ac = autocorr_at_lags(td, [1, 20, 100])
        beta, hl = fit_ar1_halflife(td)
        hl_str = f"{hl:>7.0f}t" if np.isfinite(hl) else "    n/a"
        print(f"    {K:>5} {len(td):>5} {td.mean():>+8.3f} {td.std():>7.3f} "
              f"{np.percentile(td, 5):>+7.3f} {np.percentile(td, 95):>+7.3f} "
              f"{ac[1]:>7.3f} {ac[20]:>7.3f} {beta:>9.4f} {hl_str:>11}")

    # ---- (b) Reversion test: forward Δ after deviation > threshold ----
    print(f"\n[b] Reversion test  ·  expected fade gain after |theo_diff − EMA20| > 0.5")
    print(f"    Positive entries = reverts (good); negative = trends against us")
    print(f"    {'K':>5} {'n_trig':>7} {'h=5':>9} {'h=20':>9} {'h=100':>9} {'h=500':>9}")
    for K in strikes:
        col = f"C_{K}"
        if col not in panel.columns:
            continue
        mid = panel[col].values
        S = panel["S"].values
        T = panel["T"].values
        theo = static_call(coeffs, K, S, T)
        valid = np.isfinite(mid) & (mid > FLOOR_PRICE + FLOOR_EPS) & np.isfinite(theo)
        if valid.sum() < 200:
            continue
        td = mid - theo
        td = np.where(valid, td, np.nan)
        drift = ema(np.where(valid, td, 0.0), THEO_NORM_WINDOW)
        rev = reversion_curve(td, drift, threshold=0.5, horizons=[5, 20, 100, 500])
        rev_results[K] = rev
        cells = [f"{rev[h]:>+9.4f}" if not np.isnan(rev[h]) else f"{'n/a':>9}"
                 for h in [5, 20, 100, 500]]
        print(f"    {K:>5} {rev['n_triggered']:>7} {' '.join(cells)}")

    # ---- (c) Fade backtest using static smile residual ----
    print(f"\n[c] Static-smile fade backtest  ·  hedge-step=100 ·  regime≥{IV_SCALPING_THR}")
    print(f"    {'K':>5} {'PnL':>9} {'n_trd':>6} {'n_clo':>6} {'avg/clip':>9} "
          f"{'n_hedge':>8} {'hcost':>8} {'in_reg':>7} {'pos_v':>6} {'pos_VFE':>9}")
    total_pnl = 0.0
    for K in strikes:
        bt = backtest_fade(panel, K, coeffs, delta_hedge=True, hedge_step=100)
        if bt is None:
            continue
        total_pnl += bt["pnl"]
        avg = bt["closed_pnl_mean"]
        avg_str = f"{avg:>+9.3f}" if np.isfinite(avg) else f"{'n/a':>9}"
        print(f"    {K:>5} {bt['pnl']:>+9.1f} {bt['n_trades']:>6} {bt['n_closed']:>6} "
              f"{avg_str} {bt['n_hedges']:>8} {bt['hedge_cost']:>8.1f} "
              f"{bt['n_in_regime']:>7} {bt['final_pos_v']:>6} {bt['final_pos_vfe']:>+9.1f}")
    print(f"    {'TOTAL':>5} {total_pnl:>+9.1f}")

    # ---- (d) IV Z-score fade — no smile dependency ----
    print(f"\n[d] Per-strike IV z-score fade  ·  lookback=500, z_open=2.0, z_close=0.5, hedge=100")
    print(f"    {'K':>5} {'PnL':>9} {'n_trd':>6} {'n_clo':>6} {'avg/clip':>9} "
          f"{'n_hedge':>8} {'hcost':>8} {'z_max':>7} {'pos_v':>6} {'pos_VFE':>9}")
    total_pnl_z = 0.0
    for K in strikes:
        bt = backtest_iv_zscore(panel, K, lookback=500, z_open=2.0, z_close=0.5,
                                  hedge_step=100, delta_hedge=True)
        if bt is None:
            continue
        total_pnl_z += bt["pnl"]
        avg = bt["closed_pnl_mean"]
        avg_str = f"{avg:>+9.3f}" if np.isfinite(avg) else f"{'n/a':>9}"
        print(f"    {K:>5} {bt['pnl']:>+9.1f} {bt['n_trades']:>6} {bt['n_closed']:>6} "
              f"{avg_str} {bt['n_hedges']:>8} {bt['hedge_cost']:>8.1f} "
              f"{bt['z_max']:>7.2f} {bt['final_pos_v']:>6} {bt['final_pos_vfe']:>+9.1f}")
    print(f"    {'TOTAL':>5} {total_pnl_z:>+9.1f}")


def main():
    global IV_SCALPING_THR, THR_OPEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-on", default="ROUND_3", help="round to fit static smile on")
    ap.add_argument("--test-on", default="ROUND_4", help="round to backtest on")
    ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--strikes", type=int, nargs="*", default=None)
    ap.add_argument("--regime-thr", type=float, default=None,
                    help=f"override IV_SCALPING_THR (default {IV_SCALPING_THR}); use 0 to disable regime gate")
    ap.add_argument("--open-thr", type=float, default=None,
                    help=f"override THR_OPEN (default {THR_OPEN})")
    args = ap.parse_args()

    if args.regime_thr is not None:
        IV_SCALPING_THR = args.regime_thr
    if args.open_thr is not None:
        THR_OPEN = args.open_thr

    print(f"\nFitting static smile on {args.fit_on} (all days, all strikes)...")
    coeffs, n = fit_static_smile(args.fit_on)
    print(f"   N={n} (m, IV) pairs · IV(m) = {coeffs[0]:+.6f}·m² {coeffs[1]:+.6f}·m {coeffs[2]:+.6f}")

    days = [args.day] if args.day is not None else sorted(
        historical_loader.get_available_days(args.test_on)
    )
    for d in days:
        analyze_day(args.test_on, d, coeffs, args.strikes)


if __name__ == "__main__":
    main()
