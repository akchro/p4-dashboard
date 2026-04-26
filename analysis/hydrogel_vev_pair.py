"""HYDROGEL_PACK <-> VELVETFRUIT_EXTRACT pair analysis (Phase 1 + Phase 2).

Reads historical/ROUND_3/prices_round_3_day_{0,1,2}.csv and produces:
  - per-day and stitched stats for each product
  - bid-ask width correlation
  - tick-level realized vol (with multi-horizon aggregation to control bid-ask bounce)
  - contemporaneous return correlation at {1, 10, 50, 200} tick aggregations
  - lead-lag cross-correlation
  - ADF on each price series + Engle-Granger cointegration (per-day and stitched)
  - spread plots

Outputs:
  analysis/out/pair_*.png   diagnostic plots
  stdout                     all numeric results
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "historical" / "ROUND_3"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [0, 1, 2]
HYD = "HYDROGEL_PACK"
VEV = "VELVETFRUIT_EXTRACT"


# ---------- I/O ------------------------------------------------------------

def load_day(day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df[df["product"].isin([HYD, VEV])].copy()
    return df


def panel(day: int) -> pd.DataFrame:
    """Wide panel: one row per timestamp with mid, bid_1, ask_1, vol_bid_1, vol_ask_1
    for both HYD and VEV."""
    df = load_day(day)
    cols = ["timestamp", "product", "mid_price",
            "bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1"]
    df = df[cols]
    pivot = df.pivot(index="timestamp", columns="product")
    pivot.columns = [f"{p}__{c}" for c, p in pivot.columns]
    pivot = pivot.sort_index().reset_index()
    pivot["day"] = day
    return pivot


def all_days() -> pd.DataFrame:
    parts = [panel(d) for d in DAYS]
    return pd.concat(parts, ignore_index=True)


# ---------- Phase 1 --------------------------------------------------------

def phase1_levels(df: pd.DataFrame) -> dict:
    out = {}
    for p in (HYD, VEV):
        m = df[f"{p}__mid_price"].dropna()
        out[p] = dict(
            n=int(m.size),
            min=float(m.min()),
            max=float(m.max()),
            mean=float(m.mean()),
            std=float(m.std()),
            range=float(m.max() - m.min()),
        )
    return out


def phase1_spread_liquidity(df: pd.DataFrame) -> dict:
    out = {}
    for p in (HYD, VEV):
        bid = df[f"{p}__bid_price_1"]
        ask = df[f"{p}__ask_price_1"]
        mid = df[f"{p}__mid_price"]
        width = (ask - bid).dropna()
        rel = (width / mid).dropna()
        vb = df[f"{p}__bid_volume_1"].dropna()
        va = df[f"{p}__ask_volume_1"].dropna()
        out[p] = dict(
            mean_width_ticks=float(width.mean()),
            median_width_ticks=float(width.median()),
            p95_width_ticks=float(width.quantile(0.95)),
            mean_rel_width_bps=float(rel.mean() * 1e4),
            mean_top_bid_size=float(vb.mean()),
            mean_top_ask_size=float(va.mean()),
        )
    return out


def realized_vol_multi(df: pd.DataFrame, horizons=(1, 5, 10, 50, 200)) -> pd.DataFrame:
    """Realized vol of log returns at multiple aggregation horizons (in ticks).

    Higher aggregation suppresses bid-ask bounce. Returns std per sqrt(tick)
    on the base scale; we also report the day-equivalent (10000 ticks).
    """
    rows = []
    for p in (HYD, VEV):
        mid = df[f"{p}__mid_price"]
        # within-day only — split by day to avoid cross-day jumps
        per_day_rets = []
        for d, g in df.groupby("day"):
            mid_d = g[f"{p}__mid_price"].values
            for h in horizons:
                if mid_d.size <= h:
                    continue
                # log return over h ticks
                r = np.diff(np.log(mid_d[::1]))[::1]  # tick by tick
                if h == 1:
                    rh = r
                else:
                    # subsample at every h-th tick
                    sub = mid_d[::h]
                    rh = np.diff(np.log(sub))
                rows.append(dict(product=p, day=d, horizon=h, n=len(rh),
                                 std=float(np.std(rh, ddof=1)) if len(rh) > 1 else np.nan))
    return pd.DataFrame(rows)


def realized_vol_summary(rv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across days: tick-equivalent and day-equivalent vol per horizon."""
    g = rv.groupby(["product", "horizon"]).agg(
        n=("n", "sum"),
        std=("std", "mean"),  # mean across days
    ).reset_index()
    # convert per-h-tick std to per-tick equivalent: sigma_tick = sigma_h / sqrt(h)
    g["sigma_per_tick"] = g["std"] / np.sqrt(g["horizon"])
    # day-equivalent: 10000 ticks
    g["sigma_per_day_eq"] = g["sigma_per_tick"] * np.sqrt(10000)
    return g


def rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)


def phase1_width_corr(df: pd.DataFrame, window: int = 500) -> dict:
    h_w = (df[f"{HYD}__ask_price_1"] - df[f"{HYD}__bid_price_1"]) / df[f"{HYD}__mid_price"]
    v_w = (df[f"{VEV}__ask_price_1"] - df[f"{VEV}__bid_price_1"]) / df[f"{VEV}__mid_price"]
    overall = h_w.corr(v_w)
    rc = rolling_corr(h_w, v_w, window).dropna()
    return dict(overall_corr=float(overall),
                rolling_mean=float(rc.mean()),
                rolling_p10=float(rc.quantile(0.1)),
                rolling_p90=float(rc.quantile(0.9)))


def return_distribution(df: pd.DataFrame) -> dict:
    out = {}
    for p in (HYD, VEV):
        # tick log returns within day, concatenated
        rets = []
        for d, g in df.groupby("day"):
            mid = g[f"{p}__mid_price"].values
            r = np.diff(np.log(mid))
            r = r[np.isfinite(r)]
            rets.append(r)
        r = np.concatenate(rets)
        out[p] = dict(
            n=int(r.size),
            mean=float(r.mean()),
            std=float(r.std(ddof=1)),
            skew=float(stats.skew(r)),
            kurt_excess=float(stats.kurtosis(r)),
            p1=float(np.quantile(r, 0.01)),
            p99=float(np.quantile(r, 0.99)),
            jb_stat=float(stats.jarque_bera(r).statistic),
            jb_pvalue=float(stats.jarque_bera(r).pvalue),
        )
    return out


# ---------- Phase 2 --------------------------------------------------------

def aggregated_returns(df: pd.DataFrame, horizons=(1, 10, 50, 200)) -> dict:
    """Returns dict: horizon -> DataFrame with columns h, v indexed by aggregation idx,
    using within-day subsampling and concatenating across days."""
    out = {}
    for h in horizons:
        rh, rv = [], []
        for d, g in df.groupby("day"):
            hm = g[f"{HYD}__mid_price"].values[::h]
            vm = g[f"{VEV}__mid_price"].values[::h]
            n = min(len(hm), len(vm))
            if n < 2:
                continue
            rh.append(np.diff(np.log(hm[:n])))
            rv.append(np.diff(np.log(vm[:n])))
        out[h] = pd.DataFrame(dict(h=np.concatenate(rh), v=np.concatenate(rv)))
    return out


def contemporaneous_corr(returns_by_h: dict) -> pd.DataFrame:
    rows = []
    for h, df in returns_by_h.items():
        c = df["h"].corr(df["v"])
        rows.append(dict(horizon=h, n=len(df), corr=float(c)))
    return pd.DataFrame(rows)


def leadlag_corr(df: pd.DataFrame, lags=(-20, -10, -5, -1, 0, 1, 5, 10, 20)) -> pd.DataFrame:
    """corr(dVEV[t], dHYD[t+k]) on tick log-returns within-day."""
    rh_all, rv_all = [], []
    for d, g in df.groupby("day"):
        rh_all.append(np.diff(np.log(g[f"{HYD}__mid_price"].values)))
        rv_all.append(np.diff(np.log(g[f"{VEV}__mid_price"].values)))
    rh = np.concatenate(rh_all)
    rv = np.concatenate(rv_all)
    rows = []
    for k in lags:
        if k >= 0:
            a = rv[:len(rv) - k]
            b = rh[k:]
        else:
            a = rv[-k:]
            b = rh[:len(rh) + k]
        n = min(len(a), len(b))
        if n < 2:
            continue
        c = np.corrcoef(a[:n], b[:n])[0, 1]
        rows.append(dict(lag_k=k, n=n, corr=float(c)))
    return pd.DataFrame(rows)


def adf_per_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in (HYD, VEV):
        # per-day
        for d, g in df.groupby("day"):
            mid = g[f"{p}__mid_price"].dropna().values
            stat, pval, *_ = adfuller(mid, autolag="AIC")
            rows.append(dict(scope=f"day{d}", product=p, adf_stat=float(stat), p_value=float(pval),
                             stationary=pval < 0.05))
        # stitched (concat — note discontinuities at day boundaries)
        mid = df[f"{p}__mid_price"].dropna().values
        stat, pval, *_ = adfuller(mid, autolag="AIC")
        rows.append(dict(scope="all", product=p, adf_stat=float(stat), p_value=float(pval),
                         stationary=pval < 0.05))
    return pd.DataFrame(rows)


def engle_granger(df: pd.DataFrame) -> pd.DataFrame:
    """For each scope, regress HYD = a + b*VEV + e, then ADF on residuals.
    Also use statsmodels.coint as a built-in cross-check."""
    rows = []
    scopes = [(f"day{d}", df[df["day"] == d]) for d in DAYS] + [("all", df)]
    for name, g in scopes:
        h = g[f"{HYD}__mid_price"].dropna().values
        v = g[f"{VEV}__mid_price"].dropna().values
        n = min(len(h), len(v))
        h = h[:n]; v = v[:n]
        X = add_constant(v)
        model = OLS(h, X).fit()
        alpha, beta = model.params
        resid = h - (alpha + beta * v)
        adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")
        # built-in cointegration test (Engle-Granger style)
        coint_t, coint_p, _ = coint(h, v)
        rows.append(dict(scope=name, n=n,
                         alpha=float(alpha), beta=float(beta),
                         resid_adf_stat=float(adf_stat), resid_adf_p=float(adf_p),
                         coint_t=float(coint_t), coint_p=float(coint_p),
                         resid_std=float(resid.std(ddof=1)),
                         cointegrated=adf_p < 0.05))
    return pd.DataFrame(rows)


# ---------- Plots ----------------------------------------------------------

def plot_levels_dual(df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()
    # x = global tick index, with day shading
    df = df.copy().reset_index(drop=True)
    ax1.plot(df.index, df[f"{HYD}__mid_price"], color="C0", label=HYD, lw=0.6)
    ax2.plot(df.index, df[f"{VEV}__mid_price"], color="C1", label=VEV, lw=0.6)
    ax1.set_ylabel(HYD, color="C0")
    ax2.set_ylabel(VEV, color="C1")
    # day shading
    for d in DAYS:
        idx = df.index[df["day"] == d]
        if len(idx):
            ax1.axvspan(idx.min(), idx.max(), alpha=0.05, color=f"C{d}")
            ax1.text(idx.min() + (idx.max() - idx.min()) / 2, ax1.get_ylim()[1], f"day {d}",
                     ha="center", va="top", fontsize=8, color="grey")
    ax1.set_title("Mid prices, dual axis (all days stitched)")
    ax1.set_xlabel("tick index (within sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spread(df: pd.DataFrame, alpha: float, beta: float, path: Path) -> None:
    spread = df[f"{HYD}__mid_price"] - (alpha + beta * df[f"{VEV}__mid_price"])
    z = (spread - spread.mean()) / spread.std(ddof=1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(spread.values, lw=0.5)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("spread (HYD − α − β·VEV)")
    axes[0].set_title(f"Engle-Granger spread (β={beta:.4f}, α={alpha:.2f})")
    axes[1].plot(z.values, lw=0.5, color="C2")
    for k, ls in ((1.5, ":"), (2.0, "--"), (3.0, "-.")):
        axes[1].axhline(k, color="grey", ls=ls, lw=0.7)
        axes[1].axhline(-k, color="grey", ls=ls, lw=0.7)
    axes[1].set_ylabel("spread z-score")
    axes[1].set_xlabel("tick index")
    for d in DAYS:
        idx = np.where(df["day"].values == d)[0]
        if len(idx):
            for a in axes:
                a.axvline(idx.min(), color="grey", alpha=0.4, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_leadlag(ll: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(ll["lag_k"], ll["corr"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag k (corr(ΔVEV[t], ΔHYD[t+k]))")
    ax.set_ylabel("correlation")
    ax.set_title("Lead-lag of returns (positive k = HYD lags VEV)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_width_corr(df: pd.DataFrame, window: int, path: Path) -> None:
    h_w = (df[f"{HYD}__ask_price_1"] - df[f"{HYD}__bid_price_1"]) / df[f"{HYD}__mid_price"]
    v_w = (df[f"{VEV}__ask_price_1"] - df[f"{VEV}__bid_price_1"]) / df[f"{VEV}__mid_price"]
    rc = rolling_corr(h_w, v_w, window)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(rc.values, lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Rolling corr({window}) of (ask−bid)/mid: HYD vs VEV")
    ax.set_ylabel("corr")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------- Driver ---------------------------------------------------------

def fmt_dict(d: dict, indent: int = 2) -> str:
    pad = " " * indent
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(fmt_dict(v, indent + 2))
        else:
            if isinstance(v, float):
                out.append(f"{pad}{k}: {v:.6g}")
            else:
                out.append(f"{pad}{k}: {v}")
    return "\n".join(out)


def main():
    df = all_days()
    print(f"Loaded {len(df)} timestamp rows across days {DAYS}")
    print(f"  HYD non-null mids: {df[f'{HYD}__mid_price'].notna().sum()}")
    print(f"  VEV non-null mids: {df[f'{VEV}__mid_price'].notna().sum()}\n")

    print("=" * 70)
    print("PHASE 1")
    print("=" * 70)

    print("\n[1] Price level / range (per day + stitched)")
    for d in DAYS:
        sub = df[df["day"] == d]
        print(f"  day {d}:")
        print(fmt_dict(phase1_levels(sub), indent=4))
    print("  all:")
    print(fmt_dict(phase1_levels(df), indent=4))

    print("\n[2] Spread / liquidity (stitched)")
    print(fmt_dict(phase1_spread_liquidity(df)))

    print("\n[3] Tick-level realized vol (multi-horizon, mean across days)")
    rv = realized_vol_multi(df)
    rvs = realized_vol_summary(rv)
    with pd.option_context("display.float_format", "{:.6f}".format,
                           "display.max_rows", None, "display.width", 200):
        print(rvs.to_string(index=False))

    print("\n[4] Bid-ask width correlation (rolling window=500)")
    print(fmt_dict(phase1_width_corr(df)))

    print("\n[5] Tick log-return distribution")
    rd = return_distribution(df)
    print(fmt_dict(rd))

    print("\n  Dual-axis plot       -> analysis/out/pair_levels.png")
    plot_levels_dual(df, OUT / "pair_levels.png")
    print("  Width-corr plot      -> analysis/out/pair_width_corr.png")
    plot_width_corr(df, 500, OUT / "pair_width_corr.png")

    print("\n" + "=" * 70)
    print("PHASE 2")
    print("=" * 70)

    print("\n[6] Contemporaneous return correlation (within-day, stitched)")
    rets_by_h = aggregated_returns(df)
    cc = contemporaneous_corr(rets_by_h)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(cc.to_string(index=False))

    print("\n[7] Lead-lag corr(ΔVEV[t], ΔHYD[t+k])")
    ll = leadlag_corr(df)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(ll.to_string(index=False))
    plot_leadlag(ll, OUT / "pair_leadlag.png")

    print("\n[8] ADF on each price series (per-day + stitched)")
    adf = adf_per_series(df)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(adf.to_string(index=False))

    print("\n[9] Engle-Granger cointegration: HYD = α + β·VEV + ε, then ADF on ε")
    eg = engle_granger(df)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(eg.to_string(index=False))

    # use the 'all' fit for the spread visualization
    row_all = eg[eg["scope"] == "all"].iloc[0]
    print("\n[10] Spread visualization -> analysis/out/pair_spread.png")
    plot_spread(df, alpha=row_all["alpha"], beta=row_all["beta"], path=OUT / "pair_spread.png")

    # also write concise machine-readable summary
    summary = OUT / "pair_summary.txt"
    with open(summary, "w") as f:
        f.write("Phase 1 + 2 summary (machine-light, see stdout for full report)\n\n")
        f.write("Engle-Granger fits:\n")
        f.write(eg.to_string(index=False))
        f.write("\n\nADF per series:\n")
        f.write(adf.to_string(index=False))
        f.write("\n\nContemporaneous corr by horizon:\n")
        f.write(cc.to_string(index=False))
        f.write("\n\nLead-lag:\n")
        f.write(ll.to_string(index=False))
    print(f"\nMini summary saved to {summary}")


if __name__ == "__main__":
    main()
