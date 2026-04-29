"""Round 5 edge-identification harness.

Reads historical/ROUND_5/*.csv (3 days x 50 products) and emits a structured
diagnostic per product plus per-category cross-sectional results.

Usage (from repo root):
    python3 tools/round5_screen.py
        --> writes tools/round5_screen_output.json (machine-readable)
        --> prints a compact summary to stdout

The output JSON is consumed by tools/round5_writeup.py to render the final
markdown.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

ROUND_DIR = "historical/ROUND_5"
DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000
TICKS_PER_DAY = 10_000

# Categories — derived from the prefix of the product name (everything up to
# the last underscore, except where the suffix is itself a multi-word phrase
# like "EVENING_BREATH"). Easier to hand-list.
CATEGORIES: dict[str, list[str]] = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ],
    "PANEL": [
        "PANEL_1X2",
        "PANEL_1X4",
        "PANEL_2X2",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "PEBBLES": [
        "PEBBLES_L",
        "PEBBLES_M",
        "PEBBLES_S",
        "PEBBLES_XL",
        "PEBBLES_XS",
    ],
    "ROBOT": [
        "ROBOT_DISHES",
        "ROBOT_IRONING",
        "ROBOT_LAUNDRY",
        "ROBOT_MOPPING",
        "ROBOT_VACUUMING",
    ],
    "SLEEP_POD": [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_SUEDE",
    ],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ],
    "TRANSLATOR": [
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ],
    "UV_VISOR": [
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_YELLOW",
    ],
}


# ---------- Data loading -------------------------------------------------- #


def load_prices() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = os.path.join(ROUND_DIR, f"prices_round_5_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        # global time index (monotonic across days)
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out


def load_trades() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = os.path.join(ROUND_DIR, f"trades_round_5_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------- Diagnostic primitives ---------------------------------------- #


def autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a**2).sum() * (b**2).sum())
    if den == 0:
        return float("nan")
    return float((a * b).sum() / den)


def variance_ratio(x: np.ndarray, k: int) -> float:
    """Lo-MacKinlay style: var(k-step returns) / (k * var(1-step)).
    <1 mean-reverting, >1 trending, ~1 random walk."""
    if len(x) < k * 4:
        return float("nan")
    r1 = np.diff(x)
    rk = x[k:] - x[:-k]
    v1 = r1.var(ddof=1)
    vk = rk.var(ddof=1)
    if v1 == 0:
        return float("nan")
    return float(vk / (k * v1))


def hurst_exponent(x: np.ndarray) -> float:
    """Rescaled-range Hurst on RETURNS. H<0.5 mean-reverting, >0.5 trending,
    ~0.5 random walk. Running on the price level always gives ~1 because
    the level is integrated."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 65:
        return float("nan")
    rets = np.diff(x)
    n = len(rets)
    lags = np.unique(np.geomspace(8, min(n // 2, 512), 12).astype(int))
    rs = []
    for lag in lags:
        chunks = n // lag
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = rets[i * lag:(i + 1) * lag]
            mean = seg.mean()
            dev = seg - mean
            cs = np.cumsum(dev)
            R = cs.max() - cs.min()
            S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append((lag, np.mean(vals)))
    if len(rs) < 4:
        return float("nan")
    lags_arr = np.log([r[0] for r in rs])
    rs_arr = np.log([r[1] for r in rs])
    slope, _ = np.polyfit(lags_arr, rs_arr, 1)
    return float(slope)


def dominant_period(x: np.ndarray, min_period: int = 4, max_period: int = 5000) -> tuple[float, float]:
    """Strongest periodic component via FFT.

    Returns (period_in_ticks, power_share). Power_share = power at peak /
    total power, a rough indicator of how dominant the cycle is."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 32:
        return float("nan"), float("nan")
    # detrend (linear)
    t = np.arange(n)
    coef = np.polyfit(t, x, 1)
    detrended = x - np.polyval(coef, t)
    f = np.fft.rfft(detrended)
    power = (f.conj() * f).real
    freqs = np.fft.rfftfreq(n, d=1)
    # Skip DC + very low freqs (period > max) and very high freqs (period < min)
    valid = (freqs > 1.0 / max_period) & (freqs < 1.0 / min_period)
    if not valid.any():
        return float("nan"), float("nan")
    p_valid = power.copy()
    p_valid[~valid] = 0
    idx = int(p_valid.argmax())
    if freqs[idx] == 0:
        return float("nan"), float("nan")
    period = 1.0 / freqs[idx]
    share = float(p_valid[idx] / power[1:].sum()) if power[1:].sum() > 0 else float("nan")
    return float(period), share


def book_depth(group: pd.DataFrame) -> float:
    cols = [
        "bid_volume_1", "bid_volume_2", "bid_volume_3",
        "ask_volume_1", "ask_volume_2", "ask_volume_3",
    ]
    return float(group[cols].fillna(0).sum(axis=1).mean())


# ---------- Per-product profile ----------------------------------------- #


@dataclass
class Profile:
    product: str
    n: int
    days_present: list[int]
    mid_mean: float
    mid_std: float
    mid_min: float
    mid_max: float
    ret_std_1: float       # std of 1-step changes
    ret_kurt: float        # excess kurtosis of 1-step changes
    ret_skew: float
    ac_1: float
    ac_2: float
    ac_5: float
    ac_10: float
    ac_50: float
    vr_2: float
    vr_5: float
    vr_10: float
    vr_50: float
    hurst: float
    fft_period: float
    fft_share: float
    spread_mean: float
    spread_median: float
    spread_std: float
    depth_mean: float
    trade_volume_total: int
    n_trades: int
    # Range metrics
    range_per_day_mean: float  # avg (max-min) per day
    drift_total: float  # last - first mid
    # Edge classification
    primary_screen: str
    primary_score: float
    notes: str


def profile_product(prices_p: pd.DataFrame, trades_p: pd.DataFrame) -> Profile:
    g = prices_p.sort_values("t").reset_index(drop=True)
    mid = g["mid_price"].to_numpy(dtype=float)
    bid1 = g["bid_price_1"].to_numpy(dtype=float)
    ask1 = g["ask_price_1"].to_numpy(dtype=float)
    spread = ask1 - bid1
    valid = ~np.isnan(mid)
    mid_v = mid[valid]
    n = len(mid_v)

    # Returns / changes (price-level diffs are fine — products are mean-100/1000
    # not log-style)
    rets = np.diff(mid_v) if n > 1 else np.array([])

    def safe_kurt(x):
        if len(x) < 4:
            return float("nan")
        m = x.mean()
        s = x.std(ddof=1)
        if s == 0:
            return float("nan")
        return float(((x - m) ** 4).mean() / s**4 - 3)

    def safe_skew(x):
        if len(x) < 3:
            return float("nan")
        m = x.mean()
        s = x.std(ddof=1)
        if s == 0:
            return float("nan")
        return float(((x - m) ** 3).mean() / s**3)

    period, share = dominant_period(mid_v)

    # Per-day range/drift
    per_day = []
    for d, sub in g.groupby("day"):
        m = sub["mid_price"].dropna().to_numpy()
        if len(m) > 1:
            per_day.append((m.max() - m.min(), m[-1] - m[0]))
    range_mean = float(np.mean([p[0] for p in per_day])) if per_day else float("nan")
    drift_total = float(mid_v[-1] - mid_v[0]) if n > 1 else float("nan")

    # Trade aggregate
    n_tr = int(len(trades_p))
    qty_total = int(trades_p["quantity"].sum()) if not trades_p.empty else 0

    # Edge classification — pick the best-fitting screen.
    ac1 = autocorr(rets, 1)
    vr10 = variance_ratio(mid_v, 10)
    vr50 = variance_ratio(mid_v, 50)
    spread_mean = float(np.nanmean(spread))
    ret_std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")

    # Score each screen
    screens = {}

    # 1. deterministic: |AR1|>0.3 OR strong FFT peak (share > 0.05)
    score1 = 0.0
    if not np.isnan(ac1):
        score1 += abs(ac1) * 2
    if not np.isnan(share) and share > 0.02:
        score1 += share * 5
    screens["deterministic"] = score1

    # 2. mean-reversion: VR<<1, ac1<0
    score2 = 0.0
    if not np.isnan(vr10) and vr10 < 1:
        score2 += (1 - vr10)
    if not np.isnan(vr50) and vr50 < 1:
        score2 += (1 - vr50) * 0.5
    if not np.isnan(ac1) and ac1 < 0:
        score2 += abs(ac1)
    screens["mean_reversion"] = score2

    # 3. wide-stable-spread: spread/ret_std large
    score3 = 0.0
    if spread_mean and ret_std and ret_std > 0:
        score3 = spread_mean / max(ret_std, 1e-6)
    screens["market_making"] = score3

    # 4. trending: VR>>1, ac1>0
    score4 = 0.0
    if not np.isnan(vr10) and vr10 > 1:
        score4 += (vr10 - 1)
    if not np.isnan(vr50) and vr50 > 1:
        score4 += (vr50 - 1) * 0.5
    if not np.isnan(ac1) and ac1 > 0:
        score4 += ac1
    screens["trending"] = score4

    # Pick highest-scoring screen, with thresholds
    primary, primary_score = max(screens.items(), key=lambda kv: kv[1])
    notes = ""
    THRESH = {
        "deterministic": 0.3,
        "mean_reversion": 0.3,
        "market_making": 1.5,
        "trending": 0.3,
    }
    if primary_score < THRESH[primary]:
        primary = "skip"
        notes = "no screen above threshold"

    return Profile(
        product=str(prices_p.name) if hasattr(prices_p, "name") else str(prices_p["product"].iloc[0]),
        n=int(n),
        days_present=sorted(int(d) for d in g["day"].unique()),
        mid_mean=float(np.nanmean(mid_v)),
        mid_std=float(np.nanstd(mid_v, ddof=1)) if n > 1 else float("nan"),
        mid_min=float(np.nanmin(mid_v)) if n > 0 else float("nan"),
        mid_max=float(np.nanmax(mid_v)) if n > 0 else float("nan"),
        ret_std_1=ret_std,
        ret_kurt=safe_kurt(rets),
        ret_skew=safe_skew(rets),
        ac_1=ac1,
        ac_2=autocorr(rets, 2),
        ac_5=autocorr(rets, 5),
        ac_10=autocorr(rets, 10),
        ac_50=autocorr(rets, 50),
        vr_2=variance_ratio(mid_v, 2),
        vr_5=variance_ratio(mid_v, 5),
        vr_10=vr10,
        vr_50=vr50,
        hurst=hurst_exponent(mid_v),
        fft_period=period,
        fft_share=share,
        spread_mean=spread_mean,
        spread_median=float(np.nanmedian(spread)),
        spread_std=float(np.nanstd(spread, ddof=1)),
        depth_mean=book_depth(g),
        trade_volume_total=qty_total,
        n_trades=n_tr,
        range_per_day_mean=range_mean,
        drift_total=drift_total,
        primary_screen=primary,
        primary_score=float(primary_score),
        notes=notes,
    )


# ---------- Per-category cross-sectional ---------------------------------- #


def category_analysis(prices: pd.DataFrame, products: list[str]) -> dict:
    """For a category of 5 products, build correlation, basket residuals, lead-lag."""
    # Pivot mids onto a single time index
    pivot = (
        prices[prices["product"].isin(products)]
        .pivot_table(index="t", columns="product", values="mid_price", aggfunc="first")
        .sort_index()
    )
    pivot = pivot.dropna(how="any")
    rets = pivot.diff().dropna()
    if rets.empty:
        return {"correlation": {}, "basket": {}, "residual_mr": {}, "lead_lag": {}}

    corr = rets.corr().to_dict()
    avg_corr = float(
        np.mean([corr[a][b] for a in products for b in products if a != b])
    )

    # Equal-weight basket. Standardize each first to avoid one product dominating.
    rets_std = (rets - rets.mean()) / rets.std(ddof=1)
    basket = rets_std.mean(axis=1)

    # Residual mean-reversion test on each product
    residual_mr = {}
    for p in products:
        # regress p's mid on equal-weight basket of the others
        others = [q for q in products if q != p]
        if not all(o in pivot.columns for o in others):
            continue
        basket_mid = pivot[others].mean(axis=1)
        # OLS: pivot[p] = a + b * basket_mid + resid
        A = np.vstack([np.ones(len(basket_mid)), basket_mid.values]).T
        try:
            coef, _, _, _ = np.linalg.lstsq(A, pivot[p].values, rcond=None)
            a, b = coef
            resid = pivot[p].values - (a + b * basket_mid.values)
            ar1 = autocorr(resid, 1)
            # half-life via AR(1): h = -ln(2) / ln(phi) if 0<phi<1
            if 0 < ar1 < 1:
                half_life = -np.log(2) / np.log(ar1)
            else:
                half_life = float("inf")
            resid_z_std = float(resid.std(ddof=1))
            residual_mr[p] = {
                "beta": float(b),
                "alpha": float(a),
                "resid_std": resid_z_std,
                "ar1_resid": float(ar1),
                "half_life_ticks": float(half_life) if np.isfinite(half_life) else None,
                "vr10_resid": variance_ratio(resid, 10),
            }
        except np.linalg.LinAlgError:
            residual_mr[p] = None

    # Lead-lag: for each ordered pair, corr(rets[a].shift(1), rets[b])
    lead_lag = {}
    for a in products:
        for b in products:
            if a == b:
                continue
            if a not in rets.columns or b not in rets.columns:
                continue
            x = rets[a].shift(1).values
            y = rets[b].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 100:
                continue
            xv = x[mask]; yv = y[mask]
            if xv.std() == 0 or yv.std() == 0:
                continue
            c = float(np.corrcoef(xv, yv)[0, 1])
            lead_lag[f"{a}->{b}"] = c

    return {
        "correlation_matrix": corr,
        "avg_pairwise_correlation": avg_corr,
        "n_obs": int(len(pivot)),
        "residual_mr": residual_mr,
        "lead_lag_top": dict(
            sorted(lead_lag.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        ),
    }


# ---------- Toy backtests (Phase 4) -------------------------------------- #


def backtest_zscore_mr(mid: np.ndarray, lookback: int = 200, entry_z: float = 2.0,
                       exit_z: float = 0.3, pos_limit: int = 10,
                       cost_per_unit: float = 0.5) -> dict:
    """Simulate a simple z-score mean-reversion strategy on raw mid.
    Long when z<-entry, short when z>entry, exit when |z|<exit_z.
    Cost is in price units per unit of crossing the spread.
    Returns total PnL and Sharpe of per-tick PnL increments."""
    n = len(mid)
    if n < lookback * 2:
        return {"sharpe": float("nan"), "pnl": float("nan"), "n_trades": 0}
    pos = 0
    cash = 0.0
    pnl_series = []
    last_pos = 0
    trades = 0
    eq_prev = 0.0
    for i in range(lookback, n):
        window = mid[i - lookback:i]
        mu = window.mean()
        sd = window.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            pnl_series.append(0.0)
            continue
        z = (mid[i] - mu) / sd
        target = pos
        if pos == 0:
            if z < -entry_z:
                target = pos_limit
            elif z > entry_z:
                target = -pos_limit
        else:
            if abs(z) < exit_z:
                target = 0
        if target != pos:
            d = target - pos
            cash -= d * mid[i]
            cash -= abs(d) * cost_per_unit  # crossing the spread
            pos = target
            trades += 1
        eq = cash + pos * mid[i]
        pnl_series.append(eq - eq_prev)
        eq_prev = eq
    pnl = np.array(pnl_series)
    # Force unwind at end
    final_pnl = float(eq_prev - pos * cost_per_unit)
    if pnl.std(ddof=1) == 0 or np.isnan(pnl.std(ddof=1)):
        sharpe = float("nan")
    else:
        # Ticks per day = 10000; pseudo-daily Sharpe
        sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(10000))
    return {"sharpe": sharpe, "pnl": final_pnl, "n_trades": trades}


def backtest_passive_mm(mid: np.ndarray, spread: np.ndarray, ret_std: float,
                        pos_limit: int = 10, edge_ticks: float = 0.5,
                        fill_prob: float = 0.05) -> dict:
    """Approximate market-making PnL: capture half-spread minus adverse selection.
    Per-tick fill probability * (half_spread - 0.5*sigma)."""
    if ret_std is None or np.isnan(ret_std):
        return {"sharpe": float("nan"), "pnl": float("nan")}
    half_spread = float(np.nanmean(spread)) / 2
    edge_per_fill = half_spread - 0.5 * ret_std  # adverse-selection penalty
    n = len(mid)
    expected_fills = fill_prob * n
    pnl = expected_fills * edge_per_fill
    return {"half_spread": half_spread, "ret_std": ret_std,
            "edge_per_fill": edge_per_fill, "expected_pnl": float(pnl)}


def backtest_residual_mr(p_mid: pd.Series, basket_mid: pd.Series,
                         lookback: int = 200, entry_z: float = 2.0,
                         exit_z: float = 0.3, pos_limit: int = 10,
                         cost: float = 0.5) -> dict:
    A = np.vstack([np.ones(len(basket_mid)), basket_mid.values]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, p_mid.values, rcond=None)
    except np.linalg.LinAlgError:
        return {"sharpe": float("nan"), "pnl": float("nan")}
    a, b = coef
    resid = p_mid.values - (a + b * basket_mid.values)
    return backtest_zscore_mr(resid + p_mid.values.mean(),  # shift so cost makes sense
                              lookback=lookback, entry_z=entry_z, exit_z=exit_z,
                              pos_limit=pos_limit, cost_per_unit=cost)


# ---------- Main --------------------------------------------------------- #


def main():
    print("loading prices…", file=sys.stderr)
    prices = load_prices()
    trades = load_trades()
    print(f"  prices: {len(prices):,}  trades: {len(trades):,}", file=sys.stderr)

    profiles = {}
    print("profiling 50 products…", file=sys.stderr)
    for prod, sub in prices.groupby("product"):
        tr = trades[trades["symbol"] == prod]
        prof = profile_product(sub, tr)
        prof.product = prod
        profiles[prod] = asdict(prof)

    print("running per-category analysis…", file=sys.stderr)
    cat_results = {}
    for cat, plist in CATEGORIES.items():
        cat_results[cat] = category_analysis(prices, plist)

    # Phase 4: toy backtests on top candidates per screen
    print("running toy backtests…", file=sys.stderr)
    backtests = {}
    for prod, sub in prices.groupby("product"):
        mid = sub.sort_values("t")["mid_price"].dropna().to_numpy()
        spread = (sub.sort_values("t")["ask_price_1"] - sub.sort_values("t")["bid_price_1"]).dropna().to_numpy()
        bt_zs = backtest_zscore_mr(mid)
        bt_mm = backtest_passive_mm(mid, spread, profiles[prod]["ret_std_1"])
        backtests[prod] = {"zscore_mr": bt_zs, "passive_mm": bt_mm}

    # Residual MR backtests per category
    residual_backtests = {}
    for cat, plist in CATEGORIES.items():
        pivot = (
            prices[prices["product"].isin(plist)]
            .pivot_table(index="t", columns="product", values="mid_price", aggfunc="first")
            .sort_index()
            .dropna(how="any")
        )
        for p in plist:
            others = [q for q in plist if q != p]
            if not all(o in pivot.columns for o in others) or p not in pivot.columns:
                continue
            basket = pivot[others].mean(axis=1)
            bt = backtest_residual_mr(pivot[p], basket)
            residual_backtests[p] = bt

    # Out-of-sample split: fit on day 2-3, test on day 4
    oos_backtests = {}
    for prod, sub in prices.groupby("product"):
        sub = sub.sort_values("t")
        train = sub[sub["day"].isin([2, 3])]["mid_price"].dropna().to_numpy()
        test = sub[sub["day"] == 4]["mid_price"].dropna().to_numpy()
        if len(train) < 500 or len(test) < 500:
            continue
        bt_train = backtest_zscore_mr(train)
        bt_test = backtest_zscore_mr(test)
        oos_backtests[prod] = {"train": bt_train, "test": bt_test}

    out = {
        "profiles": profiles,
        "categories": cat_results,
        "backtests": backtests,
        "residual_backtests": residual_backtests,
        "oos_backtests": oos_backtests,
    }

    out_path = "tools/round5_screen_output.json"
    # Convert any inf/nan to None for JSON
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return o
        return o

    with open(out_path, "w") as f:
        json.dump(clean(out), f, indent=2, default=str)
    print(f"wrote {out_path}", file=sys.stderr)

    # quick stdout summary
    rows = []
    for p, prof in profiles.items():
        rows.append((
            p, prof["mid_mean"], prof["spread_mean"], prof["ret_std_1"],
            prof["ac_1"], prof["vr_10"], prof["vr_50"], prof["hurst"],
            prof["fft_period"], prof["fft_share"], prof["primary_screen"],
            prof["primary_score"]
        ))
    rows.sort(key=lambda r: -r[-1])
    print(f"\n{'product':36s} {'mid':>9s} {'spr':>5s} {'sigma':>6s} {'ac1':>7s} "
          f"{'vr10':>6s} {'vr50':>6s} {'H':>5s} {'period':>8s} {'fft':>6s} {'screen':>15s} {'score':>6s}")
    for r in rows:
        ac1 = "  -  " if r[4] is None or (isinstance(r[4], float) and np.isnan(r[4])) else f"{r[4]:+.3f}"
        vr10 = "  -  " if r[5] is None or (isinstance(r[5], float) and np.isnan(r[5])) else f"{r[5]:.2f}"
        vr50 = "  -  " if r[6] is None or (isinstance(r[6], float) and np.isnan(r[6])) else f"{r[6]:.2f}"
        H = "  -  " if r[7] is None or (isinstance(r[7], float) and np.isnan(r[7])) else f"{r[7]:.2f}"
        per = "  -  " if r[8] is None or (isinstance(r[8], float) and np.isnan(r[8])) else f"{r[8]:.0f}"
        fft = "  -  " if r[9] is None or (isinstance(r[9], float) and np.isnan(r[9])) else f"{r[9]:.3f}"
        print(f"{r[0]:36s} {r[1]:9.2f} {r[2]:5.2f} {r[3]:6.3f} {ac1:>7s} "
              f"{vr10:>6s} {vr50:>6s} {H:>5s} {per:>8s} {fft:>6s} {r[10]:>15s} {r[11]:6.2f}")


if __name__ == "__main__":
    main()
