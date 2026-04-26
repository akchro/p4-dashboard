"""Mean-reversion analysis on VELVETFRUIT_EXTRACT and the VEV_* vouchers.

Tests run per series, per day:
  - AR(1) half-life (OU-style: Δx = α + λ·x_lag + ε; λ<0 ⇒ MR)
  - Augmented Dickey-Fuller (reject H0=unit-root ⇒ stationary)
  - Hurst exponent via R/S analysis (H<0.5 ⇒ MR, =0.5 RW, >0.5 trend)
  - Variance ratio at q∈{2,5,10,50} (VR<1 ⇒ MR at that horizon)
  - Lag-1 autocorrelation on Δ-series (negative ⇒ MR / bid-ask bounce)

Series tested:
  - VELVETFRUIT_EXTRACT mid (level)
  - VELVETFRUIT_EXTRACT returns (Δlog mid)
  - Each VEV_K mid (level)
  - Each VEV_K mid net of Δ_BS · ΔS (delta-hedged residual)
  - Each VEV_K implied vol (per-tick BS-inverted)
  - Smile-relative IV residual: IV_K − IV_atm at each tick

Stitching rule: per-day analysis only (TTE jumps between days, so price
levels and IV both have benign discontinuities at day boundaries). The
report aggregates the per-day results.

Run from repo root:
    python3 analysis/mean_reversion.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import options as opts  # noqa: E402

DATA = ROOT / "historical" / "ROUND_3"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(exist_ok=True)

UND = opts.UNDERLYING
STRIKES_TRADED = [5000, 5100, 5200, 5300, 5400, 5500]  # IV-solvable per CLAUDE.md
ATM_STRIKE = 5200  # closest to spot ~5100-5200 in historical data
DAYS = (0, 1, 2)


# ---------- statistics ----------

def half_life(x: np.ndarray) -> tuple[float, float]:
    """OU half-life from AR(1) regression. Returns (half_life_ticks, lambda)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return float("nan"), float("nan")
    dx = np.diff(x)
    x_lag = x[:-1]
    X = add_constant(x_lag)
    res = OLS(dx, X).fit()
    lam = float(res.params[1])
    if lam >= 0:
        return float("inf"), lam
    return float(np.log(2) / abs(lam)), lam


def adf_pvalue(x: np.ndarray) -> float:
    """ADF p-value (regression='c'). Lower ⇒ more likely stationary."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return float("nan")
    if np.std(x) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pval, *_ = adfuller(x, autolag="AIC", regression="c")
        except Exception:
            return float("nan")
    return float(pval)


def hurst_rs(x: np.ndarray, min_chunk: int = 16) -> float:
    """R/S Hurst exponent. Returns NaN if series too short / degenerate."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 * min_chunk:
        return float("nan")
    sizes = []
    s = min_chunk
    while s <= n // 4:
        sizes.append(s)
        s *= 2
    if not sizes:
        return float("nan")
    rs_vals = []
    for size in sizes:
        nsplit = n // size
        chunks = x[: nsplit * size].reshape(nsplit, size)
        means = chunks.mean(axis=1, keepdims=True)
        dev = chunks - means
        Z = np.cumsum(dev, axis=1)
        R = Z.max(axis=1) - Z.min(axis=1)
        S = chunks.std(axis=1, ddof=0)
        valid = S > 1e-12
        if not np.any(valid):
            continue
        rs_vals.append((np.log(size), np.log((R[valid] / S[valid]).mean())))
    if len(rs_vals) < 3:
        return float("nan")
    logs_n, logs_rs = zip(*rs_vals)
    slope = np.polyfit(logs_n, logs_rs, 1)[0]
    return float(slope)


def variance_ratio(x: np.ndarray, q: int) -> float:
    """Lo-MacKinlay variance ratio: VR(q) = Var(r_q) / (q · Var(r_1))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < q * 5:
        return float("nan")
    r1 = np.diff(x)
    rq = x[q:] - x[:-q]
    v1 = r1.var(ddof=1)
    if v1 <= 0:
        return float("nan")
    return float(rq.var(ddof=1) / (q * v1))


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of Δx (returns/changes)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    dx = np.diff(x)
    if dx.size < 2 or dx.std(ddof=0) < 1e-12:
        return float("nan")
    a, b = dx[:-1], dx[1:]
    return float(np.corrcoef(a, b)[0, 1])


def summarize(name: str, x: np.ndarray) -> dict:
    """Run all stats on one series."""
    hl, lam = half_life(x)
    return {
        "series": name,
        "n": int(np.isfinite(x).sum()),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "halflife_ticks": hl,
        "ar1_lambda": lam,
        "adf_p": adf_pvalue(x),
        "hurst": hurst_rs(x),
        "vr_2": variance_ratio(x, 2),
        "vr_5": variance_ratio(x, 5),
        "vr_10": variance_ratio(x, 10),
        "vr_50": variance_ratio(x, 50),
        "ac1_dx": lag1_autocorr(x),
    }


# ---------- data ----------

def load_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    products = [UND] + [f"VEV_{k}" for k in STRIKES_TRADED]
    df = df[df["product"].isin(products)]
    pivot = df.pivot(index="timestamp", columns="product", values="mid_price").sort_index()
    pivot = pivot.dropna(subset=[UND])
    pivot["day"] = day
    return pivot


def compute_iv_panel(prices: pd.DataFrame, day: int) -> pd.DataFrame:
    """Per-tick IV solved per voucher. NaN where not solvable."""
    tte_start = opts.ROUND3_TTE_AT_DAY[day]
    T = opts.time_to_expiry(prices.index.values, tte_start)
    S = prices[UND].values
    iv = pd.DataFrame(index=prices.index)
    for k in STRIKES_TRADED:
        col = f"VEV_{k}"
        if col not in prices.columns:
            iv[col] = np.nan
            continue
        C = prices[col].values
        iv[col] = opts.implied_vol(C, S, k, T)
    return iv


def delta_hedged_residual(prices: pd.DataFrame, iv: pd.DataFrame, day: int) -> pd.DataFrame:
    """For each voucher: cumulative ΔV − Δ_BS · ΔS. If options vol-trade, this
    drifts (positive vega P&L) or mean-reverts (vol cycles)."""
    tte_start = opts.ROUND3_TTE_AT_DAY[day]
    T = opts.time_to_expiry(prices.index.values, tte_start)
    S = prices[UND].values
    out = pd.DataFrame(index=prices.index)
    for k in STRIKES_TRADED:
        col = f"VEV_{k}"
        if col not in prices.columns:
            out[col] = np.nan
            continue
        sigma = iv[col].values
        # use last-known IV when current is NaN, then a flat fallback
        sigma = pd.Series(sigma).ffill().fillna(0.013).values
        g = opts.greeks(S, k, T, sigma)
        delta = g["delta"]
        dV = np.diff(prices[col].values)
        dS = np.diff(S)
        delta_lag = delta[:-1]
        unhedged = np.concatenate([[0.0], np.cumsum(dV - delta_lag * dS)])
        out[col] = unhedged
    return out


# ---------- driver ----------

def run_day(day: int) -> dict:
    prices = load_day(day)
    iv = compute_iv_panel(prices, day)
    hedged = delta_hedged_residual(prices, iv, day)

    rows = []
    # underlying
    s = prices[UND].values
    rows.append(summarize(f"day{day}::{UND}_mid", s))
    rows.append(summarize(f"day{day}::{UND}_logret", np.log(s)))

    # voucher levels + delta-hedged residual + IV
    for k in STRIKES_TRADED:
        col = f"VEV_{k}"
        if col not in prices.columns:
            continue
        rows.append(summarize(f"day{day}::{col}_mid", prices[col].values))
        rows.append(summarize(f"day{day}::{col}_dh_resid", hedged[col].values))
        rows.append(summarize(f"day{day}::{col}_iv", iv[col].values))

    # smile-relative IV (vs ATM)
    if f"VEV_{ATM_STRIKE}" in iv.columns:
        atm = iv[f"VEV_{ATM_STRIKE}"]
        for k in STRIKES_TRADED:
            if k == ATM_STRIKE:
                continue
            col = f"VEV_{k}"
            if col not in iv.columns:
                continue
            spread = (iv[col] - atm).values
            rows.append(summarize(f"day{day}::{col}_iv_minus_atm", spread))

    return {"prices": prices, "iv": iv, "hedged": hedged, "rows": rows}


def make_plots(per_day: dict[int, dict]):
    # IV time series, stitched across days
    fig, ax = plt.subplots(figsize=(12, 5))
    offset = 0
    for d in DAYS:
        iv = per_day[d]["iv"]
        ts = iv.index.values + offset
        for k in STRIKES_TRADED:
            col = f"VEV_{k}"
            if col in iv.columns:
                ax.plot(ts, iv[col].values, lw=0.6, alpha=0.7,
                        label=col if d == 0 else None)
        offset += opts.TIMESTAMP_PER_DAY
        ax.axvline(offset, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel("stitched timestamp (day-boundaries marked)")
    ax.set_ylabel("IV per √day")
    ax.set_title("VEV implied-vol time series, all days, all solvable strikes")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "mr_iv_timeseries.png", dpi=120)
    plt.close(fig)

    # half-life sweep across strikes for IV (per day)
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in DAYS:
        rows = [r for r in per_day[d]["rows"] if r["series"].endswith("_iv")]
        rows = [r for r in rows if "VEV_" in r["series"]]
        ks, hls = [], []
        for r in rows:
            sym = r["series"].split("::")[1].replace("_iv", "")
            ks.append(int(sym.split("_")[1]))
            hls.append(r["halflife_ticks"])
        order = np.argsort(ks)
        ks = np.array(ks)[order]
        hls = np.array(hls)[order]
        ax.plot(ks, hls, "o-", label=f"day {d}")
    ax.set_xlabel("strike K")
    ax.set_ylabel("AR(1) IV half-life (ticks)")
    ax.set_yscale("log")
    ax.set_title("IV mean-reversion half-life across strikes (1 tick = 100 ts units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mr_iv_halflife.png", dpi=120)
    plt.close(fig)

    # underlying log-mid per day
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for d, ax in zip(DAYS, axes):
        s = per_day[d]["prices"][UND]
        ax.plot(s.index.values, np.log(s.values), lw=0.6)
        ax.set_title(f"day {d}: log {UND}")
        ax.set_xlabel("timestamp")
    axes[0].set_ylabel("log mid")
    fig.tight_layout()
    fig.savefig(OUT / "mr_underlying_logmid.png", dpi=120)
    plt.close(fig)


def main():
    per_day: dict[int, dict] = {}
    all_rows = []
    for d in DAYS:
        print(f"Loading day {d}…", file=sys.stderr)
        res = run_day(d)
        per_day[d] = res
        all_rows.extend(res["rows"])

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "mean_reversion_stats.csv", index=False)

    make_plots(per_day)

    # console-friendly summary
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v: .4g}")

    print("\n=== UNDERLYING ===")
    mask = df["series"].str.contains(UND)
    print(df.loc[mask, ["series", "n", "halflife_ticks", "ar1_lambda",
                        "adf_p", "hurst", "vr_2", "vr_10", "vr_50", "ac1_dx"]])

    print("\n=== VOUCHER MID (level) ===")
    mask = df["series"].str.endswith("_mid") & df["series"].str.contains("VEV_")
    print(df.loc[mask, ["series", "halflife_ticks", "adf_p", "hurst",
                        "vr_10", "ac1_dx"]])

    print("\n=== VOUCHER DELTA-HEDGED RESIDUAL ===")
    mask = df["series"].str.endswith("_dh_resid")
    print(df.loc[mask, ["series", "halflife_ticks", "adf_p", "hurst", "vr_50"]])

    print("\n=== IV ===")
    mask = df["series"].str.endswith("_iv")
    print(df.loc[mask, ["series", "n", "mean", "std", "halflife_ticks",
                        "adf_p", "hurst", "vr_10", "vr_50"]])

    print("\n=== IV minus ATM (smile residual) ===")
    mask = df["series"].str.endswith("_iv_minus_atm")
    print(df.loc[mask, ["series", "mean", "std", "halflife_ticks",
                        "adf_p", "hurst", "vr_10"]])

    print(f"\nWrote {OUT/'mean_reversion_stats.csv'}")
    print(f"Wrote {OUT/'mr_iv_timeseries.png'}")
    print(f"Wrote {OUT/'mr_iv_halflife.png'}")
    print(f"Wrote {OUT/'mr_underlying_logmid.png'}")


if __name__ == "__main__":
    main()
