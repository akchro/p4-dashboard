"""Black-Scholes pricer, vectorized IV solver, and greeks for VEV vouchers.

No scipy dependency — uses Abramowitz & Stegun 26.2.17 normal CDF approximation
(accuracy ~1.5e-7, which is well below our tick precision).

Units convention throughout: T is in DAYS, sigma is per-sqrt-day. With r=0 and
these units, BS is self-consistent. A sigma of 0.015 per-sqrt-day corresponds
to ~24% annualized (multiply by sqrt(252)).
"""
import numpy as np

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_PRODUCTS = [f"VEV_{k}" for k in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"

# Timestamp unit range per day (IMC convention: 0..999_900, step 100)
TIMESTAMP_PER_DAY = 1_000_000

# Historical TTE at start of day (ROUND_3 folder convention per wiki:
# day 0 = tutorial → 8d, day 1 = R1 → 7d, day 2 = R2 → 6d).
ROUND3_TTE_AT_DAY = {0: 8.0, 1: 7.0, 2: 6.0}
LIVE_TTE_AT_START = 5.0  # round 3 starts at TTE=5d


def voucher_strike(product):
    """Return strike int if product is a voucher symbol, else None."""
    if not isinstance(product, str) or not product.startswith("VEV_"):
        return None
    try:
        return int(product[4:])
    except ValueError:
        return None


def tte_for_day(round_name, day):
    """Best-effort TTE at start of a given (round, day). Defaults to 8-day."""
    if round_name and "ROUND_3" in round_name.upper():
        return ROUND3_TTE_AT_DAY.get(int(day), max(8.0 - int(day), 0.0))
    return max(8.0 - int(day), 0.0)


def time_to_expiry(timestamps, tte_start_days):
    """Convert raw intra-day timestamps to TTE in days. Clamped at 1e-6."""
    ts = np.asarray(timestamps, dtype=float)
    return np.maximum(tte_start_days - ts / TIMESTAMP_PER_DAY, 1e-6)


def _norm_cdf(x):
    """Vectorized Φ(x) via A&S 26.2.17. Max error ~1.5e-7."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    x = np.asarray(x, dtype=float)
    sign = np.where(x >= 0, 1.0, -1.0)
    abs_x = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * abs_x)
    poly = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1
    y = 1.0 - poly * t * np.exp(-abs_x * abs_x)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def bs_call(S, K, T, sigma, r=0.0):
    """Vectorized Black-Scholes call price."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    bad = (T <= 0) | (sigma <= 0) | ~np.isfinite(S) | ~np.isfinite(K)
    safe_T = np.where(bad, 1e-9, T)
    safe_sigma = np.where(bad, 1e-9, sigma)

    sqrtT = np.sqrt(safe_T)
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * safe_sigma * safe_sigma) * safe_T) / (safe_sigma * sqrtT)
        d2 = d1 - safe_sigma * sqrtT
        out = S * _norm_cdf(d1) - K * np.exp(-r * safe_T) * _norm_cdf(d2)

    intrinsic = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)
    return np.where(bad, intrinsic, out)


def implied_vol(C, S, K, T, r=0.0, n_iter=40, tol=1e-5):
    """Vectorized Newton-Raphson IV solver.

    Returns an array of sigmas with NaN where no valid solution exists
    (outside no-arb bounds, or at price floor).
    """
    C = np.asarray(C, dtype=float)
    shape = C.shape
    S = np.broadcast_to(np.asarray(S, dtype=float), shape).astype(float)
    K = np.broadcast_to(np.asarray(K, dtype=float), shape).astype(float)
    T = np.broadcast_to(np.asarray(T, dtype=float), shape).astype(float)

    intrinsic = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)
    # Valid only if price is strictly above intrinsic and strictly below S.
    valid = (
        np.isfinite(C) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) &
        (C > intrinsic + 1e-6) & (C < S - 1e-6) & (T > 1e-9)
    )

    sigma = np.where(valid, 0.02, np.nan)  # per-sqrt-day, decent prior

    for _ in range(n_iter):
        with np.errstate(invalid="ignore", divide="ignore"):
            sqrtT = np.sqrt(T)
            d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
            price = bs_call(S, K, T, sigma, r)
            vega = S * _norm_pdf(d1) * sqrtT
            diff = C - price
            update = np.where(vega > 1e-12, diff / vega, 0.0)
            sigma = np.clip(sigma + update, 1e-6, 5.0)
        finite_diff = diff[valid]
        finite_diff = finite_diff[np.isfinite(finite_diff)]
        if finite_diff.size == 0 or np.max(np.abs(finite_diff)) < tol:
            break

    # Reject non-converged points: final residual too large, or sigma pinned
    # at the clip boundaries (Newton bounced between extremes without solving).
    final_price = bs_call(S, K, T, sigma, r)
    resid = np.abs(final_price - C)
    pinned = (sigma <= 1e-5) | (sigma >= 4.99)
    converged = valid & (resid < max(tol * 10, 0.05)) & ~pinned
    return np.where(converged, sigma, np.nan)


def greeks(S, K, T, sigma, r=0.0):
    """Returns dict of arrays: delta, gamma, theta, vega (per sqrt-day)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    bad = (T <= 0) | (sigma <= 0) | ~np.isfinite(sigma)
    safe_T = np.where(bad, 1e-9, T)
    safe_sigma = np.where(bad, 1e-9, sigma)
    sqrtT = np.sqrt(safe_T)

    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * safe_sigma * safe_sigma) * safe_T) / (safe_sigma * sqrtT)
        pdf = _norm_pdf(d1)
        delta = _norm_cdf(d1)
        gamma = pdf / (S * safe_sigma * sqrtT)
        theta = -S * pdf * safe_sigma / (2.0 * sqrtT)
        vega = S * pdf * sqrtT

    delta = np.where(bad, (S > K).astype(float), delta)
    gamma = np.where(bad, 0.0, gamma)
    theta = np.where(bad, 0.0, theta)
    vega = np.where(bad, 0.0, vega)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def log_moneyness(K, S, T):
    """Normalized log-moneyness m = ln(K/S) / sqrt(T). Standardizes smile x-axis."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    return np.log(K / S) / np.sqrt(T)


def implied_underlying(C, K, T, sigma, r=0.0, n_iter=40, tol=1e-5):
    """Invert BS_call(S, K, T, sigma) = C for S, given the other inputs.

    Newton-Raphson using delta as the derivative. Returns NaN where C is at
    or below the 0.5 price floor (no information) or the solver fails to
    converge within bounds. Useful for visualizing the underlying value that
    each voucher's price implies at a chosen baseline sigma.
    """
    C = np.asarray(C, dtype=float)
    shape = C.shape
    K_arr = np.broadcast_to(np.asarray(K, dtype=float), shape).astype(float)
    T_arr = np.broadcast_to(np.asarray(T, dtype=float), shape).astype(float)
    sig_arr = np.broadcast_to(np.asarray(sigma, dtype=float), shape).astype(float)

    valid = (
        np.isfinite(C) & np.isfinite(K_arr) & np.isfinite(T_arr) & np.isfinite(sig_arr) &
        (C > 0.5 + 1e-6) & (T_arr > 1e-9) & (sig_arr > 0)
    )

    # Initial guess: deep-ITM exact value S = K + C; converges fast elsewhere.
    S = np.where(valid, K_arr + C, K_arr)

    diff = np.full(shape, np.nan)
    for _ in range(n_iter):
        with np.errstate(invalid="ignore", divide="ignore"):
            sqrtT = np.sqrt(T_arr)
            d1 = (np.log(S / K_arr) + 0.5 * sig_arr * sig_arr * T_arr) / (sig_arr * sqrtT)
            price = bs_call(S, K_arr, T_arr, sig_arr, r)
            delta = _norm_cdf(d1)
            diff = price - C
            update = np.where(delta > 1e-9, diff / delta, 0.0)
            S = np.maximum(S - update, 1e-3)
        finite_diff = diff[valid]
        finite_diff = finite_diff[np.isfinite(finite_diff)]
        if finite_diff.size == 0 or np.max(np.abs(finite_diff)) < tol:
            break

    final_price = bs_call(S, K_arr, T_arr, sig_arr, r)
    resid = np.abs(final_price - C)
    converged = valid & (resid < max(tol * 10, 0.05))
    return np.where(converged, S, np.nan)
