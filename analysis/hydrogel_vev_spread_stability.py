"""Quick robustness check: half-life of mean reversion + rolling β stability.

Borrows from Phase 3 (steps 11, 12) but only enough to confirm the Phase 2
verdict. We expect (a) drifting / sign-flipping rolling β and (b) very long
half-life on the stitched residual.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "historical" / "ROUND_3"
OUT = ROOT / "analysis" / "out"

HYD = "HYDROGEL_PACK"
VEV = "VELVETFRUIT_EXTRACT"


def load_panel():
    rows = []
    for d in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{d}.csv", sep=";")
        df = df[df["product"].isin([HYD, VEV])]
        pivot = df.pivot(index="timestamp", columns="product", values="mid_price")
        pivot["day"] = d
        rows.append(pivot.reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out


def half_life(spread: np.ndarray) -> float:
    """OU-style half-life estimate from AR(1) on Δspread vs spread_lag."""
    s = spread
    ds = np.diff(s)
    s_lag = s[:-1]
    X = add_constant(s_lag)
    model = OLS(ds, X).fit()
    lam = model.params[1]  # coefficient on lag
    if lam >= 0:
        return float("inf")
    return float(np.log(2) / abs(lam))


def rolling_beta(h: np.ndarray, v: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS β of HYD on VEV. Returns array length len(h), with NaN for warmup."""
    n = len(h)
    out = np.full(n, np.nan)
    for i in range(window, n):
        H = h[i - window:i]
        V = v[i - window:i]
        X = add_constant(V)
        out[i] = OLS(H, X).fit().params[1]
    return out


def main():
    df = load_panel()
    print(f"loaded {len(df)} rows")

    # 1) Half-life per day + stitched
    print("\nHalf-life of mean reversion on EG spread:")
    for scope_name, sub in [("day0", df[df["day"] == 0]),
                            ("day1", df[df["day"] == 1]),
                            ("day2", df[df["day"] == 2]),
                            ("all", df)]:
        h = sub[HYD].values
        v = sub[VEV].values
        X = add_constant(v)
        a, b = OLS(h, X).fit().params
        spread = h - (a + b * v)
        hl = half_life(spread)
        print(f"  {scope_name:>5s}: β={b:+.4f}  spread_std={spread.std(ddof=1):7.2f}  "
              f"half-life={hl:.0f} ticks "
              f"(=> {hl/10000:.2f} days)")

    # 2) Rolling β (1000-tick window, on stitched data)
    print("\nRolling β on stitched data (window=1000):")
    h = df[HYD].values
    v = df[VEV].values
    rb = rolling_beta(h, v, window=1000)
    rb_clean = rb[~np.isnan(rb)]
    print(f"  rolling β:  mean={rb_clean.mean():+.4f}  std={rb_clean.std():.4f}")
    print(f"  fraction positive: {(rb_clean > 0).mean():.2%}")
    print(f"  range: [{rb_clean.min():+.3f}, {rb_clean.max():+.3f}]")
    print(f"  fraction with |β|>1: {(np.abs(rb_clean) > 1).mean():.2%}")

    # plot rolling β
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rb, lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(rb_clean.mean(), color="grey", ls="--", lw=0.5,
               label=f"mean={rb_clean.mean():+.3f}")
    # shade by day
    for d in (0, 1, 2):
        idx = df.index[df["day"] == d]
        ax.axvspan(idx.min(), idx.max(), alpha=0.05, color=f"C{d}")
    ax.set_title("Rolling β (HYD on VEV, window=1000 ticks)")
    ax.set_ylabel("β")
    ax.set_xlabel("tick index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pair_rolling_beta.png", dpi=140)
    plt.close(fig)
    print("  -> analysis/out/pair_rolling_beta.png")


if __name__ == "__main__":
    main()
