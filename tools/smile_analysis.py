"""Smile residuals + IV autocorrelation analysis for vouchers.

(a) Fit IV ≈ a + b·m + c·m² per timestamp where m = log(K/S). Look at how each
    strike's residual evolves over time — is the 5400 dip persistent? Are the
    wing IVs systematically rich vs the smooth fit?

(b) Compute autocorrelation of each strike's IV series at lags {1, 5, 20, 100}
    (in 100-tick units). Fit AR(1) per strike to estimate mean-reversion
    half-life.

Usage:
    python3 tools/smile_analysis.py --round ROUND_4 --day 1
    python3 tools/smile_analysis.py --round ROUND_4              # all days
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


def _build_iv_panel(activities, day, tte_start, drop_floor=True):
    """Wide DataFrame: index=timestamp, columns=strike, values=IV. Plus an 'S' col.

    drop_floor=True (default): for each (timestamp, strike), set IV=NaN when the
    voucher mid sits at the 0.5 tick floor — IVs there are quantization
    artifacts and would bias both the fit and residuals.
    """
    df = activities[activities["day"] == day]
    ve = df[df["product"] == opts.UNDERLYING][["timestamp", "mid_price"]].rename(
        columns={"mid_price": "S"}
    )

    panel = ve.copy()
    for K in opts.STRIKES:
        v = df[df["product"] == f"VEV_{K}"][["timestamp", "mid_price"]].rename(
            columns={"mid_price": f"C_{K}"}
        )
        panel = panel.merge(v, on="timestamp", how="left")

    panel = panel.dropna(subset=["S"]).reset_index(drop=True)
    T = opts.time_to_expiry(panel["timestamp"].values, tte_start)

    iv_cols = {}
    floor_counts = {}
    for K in opts.STRIKES:
        cname = f"C_{K}"
        if cname not in panel.columns:
            continue
        C = panel[cname].values
        S = panel["S"].values
        iv = opts.implied_vol(C, S, K, T)
        if drop_floor:
            mask = C <= FLOOR_PRICE + FLOOR_EPS
            floor_counts[K] = int(np.sum(mask & np.isfinite(iv)))
            iv = np.where(mask, np.nan, iv)
        iv_cols[K] = iv

    iv_df = pd.DataFrame(iv_cols, index=panel["timestamp"].values)
    iv_df.index.name = "timestamp"
    return iv_df, panel["S"].values, floor_counts


def fit_smile_residuals(iv_df, S_series, min_strikes=4):
    """For each row (timestamp), fit IV = a + b·m + c·m² across strikes.
    Return residuals DataFrame (same shape as iv_df) and fit-quality stats."""
    strikes = np.array(iv_df.columns, dtype=float)
    iv = iv_df.values
    n_t, n_k = iv.shape
    residuals = np.full_like(iv, np.nan)
    rsq = np.full(n_t, np.nan)
    coefs = np.full((n_t, 3), np.nan)

    for i in range(n_t):
        row = iv[i, :]
        S = S_series[i]
        if not np.isfinite(S) or S <= 0:
            continue
        valid = np.isfinite(row)
        if valid.sum() < min_strikes:
            continue
        m = np.log(strikes[valid] / S)
        y = row[valid]
        # Quadratic fit via normal equations (np.polyfit with deg=2)
        try:
            c2, c1, c0 = np.polyfit(m, y, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        coefs[i] = [c0, c1, c2]
        # Residual on every strike (including invalid ones we'll mark NaN)
        m_all = np.log(strikes / S)
        fitted = c0 + c1 * m_all + c2 * m_all**2
        ss_tot = np.sum((y - y.mean())**2)
        ss_res = np.sum((y - (c0 + c1 * m + c2 * m**2))**2)
        rsq[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        residuals[i, :] = row - fitted

    res_df = pd.DataFrame(residuals, index=iv_df.index, columns=iv_df.columns)
    return res_df, rsq, coefs


def autocorr_at_lags(series, lags):
    """Return dict {lag: autocorr}. NaN-aware."""
    out = {}
    s = pd.Series(series).dropna()
    if len(s) < max(lags) + 5:
        return {lag: np.nan for lag in lags}
    for lag in lags:
        out[lag] = s.autocorr(lag=lag)
    return out


def fit_ar1_halflife(series):
    """Fit y_t = α + β·y_{t-1} + ε. Return (β, half-life in steps)."""
    s = pd.Series(series).dropna()
    if len(s) < 50:
        return np.nan, np.nan
    y = s.values[1:]
    x = s.values[:-1]
    # OLS: β = cov(x,y)/var(x); α = ȳ − β·x̄
    var_x = np.var(x)
    if var_x == 0:
        return np.nan, np.nan
    beta = np.cov(x, y, ddof=0)[0, 1] / var_x
    if beta <= 0 or beta >= 1:
        return beta, np.nan
    half_life = np.log(0.5) / np.log(beta)
    return beta, half_life


def analyze_day(round_name, day, tte_override=None):
    bundle = historical_loader.load_round(round_name)
    tte_start = tte_override if tte_override is not None else opts.tte_for_day(round_name, day)
    iv_df, S_series, floor_counts = _build_iv_panel(bundle["activities"], day, tte_start)

    print(f"\n{'='*78}")
    print(f"{round_name} day {day}  ·  TTE@start={tte_start:.1f}d  ·  N={len(iv_df)} timestamps")
    if floor_counts:
        nz = {k: v for k, v in floor_counts.items() if v > 0}
        if nz:
            print(f"Floor-pinned (excluded): {nz}")
    print('='*78)

    # ---- (a) Smile residuals ----
    res_df, rsq, coefs = fit_smile_residuals(iv_df, S_series)
    print("\n[a] Quadratic smile fit residuals (IV − fitted)  ·  per-strike stats")
    print(f"    Median R² across timestamps: {np.nanmedian(rsq):.4f}")
    print(f"    Median (a, b, c) coefs: ({np.nanmedian(coefs[:,0]):.5f}, "
          f"{np.nanmedian(coefs[:,1]):.5f}, {np.nanmedian(coefs[:,2]):.5f})")
    print(f"    {'K':>6} {'n':>6} {'mean':>10} {'std':>10} {'p5':>10} "
          f"{'median':>10} {'p95':>10} {'|t|-stat':>10}")
    for K in res_df.columns:
        col = res_df[K].dropna()
        if len(col) == 0:
            continue
        m, s = col.mean(), col.std()
        # |t|-stat = mean / (std / sqrt(n-1)) — gauges if residual is significantly non-zero
        tstat = abs(m) / (s / np.sqrt(len(col))) if s > 0 else np.nan
        print(f"    {K:>6} {len(col):>6} {m:>10.5f} {s:>10.5f} "
              f"{col.quantile(0.05):>10.5f} {col.median():>10.5f} "
              f"{col.quantile(0.95):>10.5f} {tstat:>10.1f}")

    # ---- (b) Autocorrelation + AR(1) half-life ----
    print("\n[b] IV mean-reversion  ·  autocorr at lags (1=100 ticks, 5=500, 20=2k, 100=10k)")
    print(f"    {'K':>6} {'n':>6} {'std':>10} {'ρ(1)':>8} {'ρ(5)':>8} {'ρ(20)':>8} "
          f"{'ρ(100)':>8} {'AR(1)β':>10} {'half-life':>12}")
    for K in iv_df.columns:
        col = iv_df[K].dropna()
        if len(col) < 100:
            print(f"    {K:>6} {len(col):>6} (insufficient data)")
            continue
        ac = autocorr_at_lags(col.values, [1, 5, 20, 100])
        beta, hl = fit_ar1_halflife(col.values)
        hl_str = f"{hl:>9.0f} ticks" if np.isfinite(hl) else "        n/a"
        print(f"    {K:>6} {len(col):>6} {col.std():>10.5f} "
              f"{ac[1]:>8.3f} {ac[5]:>8.3f} {ac[20]:>8.3f} {ac[100]:>8.3f} "
              f"{beta:>10.4f} {hl_str:>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="ROUND_4")
    ap.add_argument("--day", type=int, default=None, help="single day; omit for all")
    ap.add_argument("--tte", type=float, default=None)
    args = ap.parse_args()

    if args.day is not None:
        analyze_day(args.round, args.day, args.tte)
    else:
        days = historical_loader.get_available_days(args.round)
        for d in sorted(days):
            analyze_day(args.round, d, args.tte)


if __name__ == "__main__":
    main()
