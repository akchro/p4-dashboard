"""Galaxy Sounds pair / basket analysis (Round 5).

Question: BLACK_HOLES vs SOLAR_FLAMES *look* inversely related on the
overlay chart. Is that a tradable mean-reverting spread, a coincidence,
or part of a wider 5-product basket structure?

Pipeline (mirrors the user's plan):
  1. Return correlation matrix at lags {1, 5, 50}, per day + stitched
  2. Pairwise hedge ratio β from OLS  Y = α + β·X
  3. Spread stationarity (ADF) on residuals
  4. Half-life of mean reversion from AR(1) on residual
  5. Out-of-sample: fit β/μ on day d, test on later days
  6. 5-product equal-weight basket — regress each product on basket;
     ADF + half-life on each residual

Inputs:  historical/ROUND_5/prices_round_5_day_{2,3,4}.csv
Outputs: stdout report
         analysis/out/galaxy_sounds_levels.png
         analysis/out/galaxy_sounds_spread_BH_SF.png
         analysis/out/galaxy_sounds_basket_residuals.png
"""
from __future__ import annotations

from itertools import combinations
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
DATA = ROOT / "historical" / "ROUND_5"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [2, 3, 4]
PRODUCTS = [
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]
SHORT = {p: p.replace("GALAXY_SOUNDS_", "") for p in PRODUCTS}


# ---------- I/O ----------------------------------------------------------

def load_day(day: int) -> pd.DataFrame:
    """Wide panel: one row per timestamp, columns = mid_price per product."""
    path = DATA / f"prices_round_5_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df[df["product"].isin(PRODUCTS)][["timestamp", "product", "mid_price"]]
    pivot = (df.pivot(index="timestamp", columns="product", values="mid_price")
               .sort_index().reset_index())
    pivot["day"] = day
    return pivot


def all_days() -> pd.DataFrame:
    return pd.concat([load_day(d) for d in DAYS], ignore_index=True)


# ---------- Step 1: return correlation ---------------------------------

def returns_at_lag(panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Within-day log returns at given lag, concatenated across days."""
    parts = []
    for _, g in panel.groupby("day"):
        sub = g[PRODUCTS].values[::lag] if lag > 1 else g[PRODUCTS].values
        if len(sub) < 2:
            continue
        r = np.diff(np.log(sub), axis=0)
        parts.append(r)
    return pd.DataFrame(np.concatenate(parts, axis=0), columns=PRODUCTS)


def corr_matrix_at_lag(panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    r = returns_at_lag(panel, lag)
    return r.corr()


# ---------- Step 2 + 3 + 4: pairwise OLS / ADF / half-life --------------

def half_life(resid: np.ndarray) -> float:
    """AR(1): ε_t = ρ·ε_{t-1} + u_t  →  HL = -ln(2)/ln(ρ)."""
    if len(resid) < 3:
        return float("nan")
    e = resid - resid.mean()
    rho_num = float(np.dot(e[:-1], e[1:]))
    rho_den = float(np.dot(e[:-1], e[:-1]))
    if rho_den == 0:
        return float("nan")
    rho = rho_num / rho_den
    if rho <= 0 or rho >= 1:
        # ρ ≥ 1 → non-mean-reverting; ρ ≤ 0 → flip-flop, no usable HL
        return float("nan")
    return float(-np.log(2.0) / np.log(rho))


def pairwise_eg(panel: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    """Engle-Granger style: for every (Y, X) pair, OLS Y = α + β·X then ADF
    on the residual. Reports β, ADF p, residual std, and half-life."""
    rows = []
    for y_name, x_name in combinations(PRODUCTS, 2):
        y = panel[y_name].dropna().values
        x = panel[x_name].dropna().values
        n = min(len(y), len(x))
        y, x = y[:n], x[:n]
        if n < 50:
            continue
        X = add_constant(x)
        m = OLS(y, X).fit()
        alpha, beta = m.params
        resid = y - (alpha + beta * x)
        try:
            adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")
        except Exception:
            adf_stat, adf_p = float("nan"), float("nan")
        # contemporaneous return correlation at lag 1 for context
        ry = np.diff(np.log(y))
        rx = np.diff(np.log(x))
        rcorr = float(np.corrcoef(ry, rx)[0, 1]) if len(ry) > 2 else float("nan")
        rows.append(dict(
            scope=scope_name,
            y=SHORT[y_name], x=SHORT[x_name],
            n=n,
            alpha=float(alpha), beta=float(beta),
            ret_corr=rcorr,
            resid_std=float(resid.std(ddof=1)),
            adf_stat=float(adf_stat), adf_p=float(adf_p),
            stationary=bool(adf_p < 0.05),
            half_life=half_life(resid),
        ))
    return pd.DataFrame(rows)


# ---------- Step 5: out-of-sample ---------------------------------------

def out_of_sample_pair(panel: pd.DataFrame, y_name: str, x_name: str) -> pd.DataFrame:
    """Fit α, β on day d_train; apply to day d_test; report
    residual mean drift, std, and ADF p on the test sample."""
    rows = []
    fits = {}
    for d in DAYS:
        g = panel[panel["day"] == d]
        y = g[y_name].dropna().values
        x = g[x_name].dropna().values
        n = min(len(y), len(x))
        if n < 50:
            continue
        X = add_constant(x[:n])
        m = OLS(y[:n], X).fit()
        a, b = m.params
        resid = y[:n] - (a + b * x[:n])
        fits[d] = (a, b, resid.mean(), resid.std(ddof=1))

    for d_train in DAYS:
        if d_train not in fits:
            continue
        a, b, mu_train, sd_train = fits[d_train]
        for d_test in DAYS:
            g = panel[panel["day"] == d_test]
            y = g[y_name].dropna().values
            x = g[x_name].dropna().values
            n = min(len(y), len(x))
            if n < 50:
                continue
            resid = y[:n] - (a + b * x[:n])
            try:
                _, adf_p, *_ = adfuller(resid, autolag="AIC")
            except Exception:
                adf_p = float("nan")
            rows.append(dict(
                train=d_train, test=d_test, n=n,
                alpha=float(a), beta=float(b),
                mu_test=float(resid.mean()),
                mu_drift=float(resid.mean() - mu_train),
                sd_test=float(resid.std(ddof=1)),
                sd_ratio=float(resid.std(ddof=1) / sd_train) if sd_train else float("nan"),
                adf_p_test=float(adf_p),
            ))
    return pd.DataFrame(rows)


# ---------- Step 6: basket residual --------------------------------------

def basket_residuals(panel: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    """Equal-weight z-score basket: for each product, regress its mid on the
    mean-of-other-4 mid. ADF + HL on the residual.

    We z-score within each product first so units don't dominate (BLACK_HOLES
    sits near 9700; PLANETARY_RINGS near 11400) — equal-weight on z-scores
    treats them symmetrically.
    """
    z = panel[PRODUCTS].copy()
    z = (z - z.mean()) / z.std(ddof=1)
    rows = []
    for p in PRODUCTS:
        others = [q for q in PRODUCTS if q != p]
        basket = z[others].mean(axis=1).values
        target = z[p].values
        mask = np.isfinite(basket) & np.isfinite(target)
        if mask.sum() < 50:
            continue
        X = add_constant(basket[mask])
        m = OLS(target[mask], X).fit()
        a, b = m.params
        resid = target[mask] - (a + b * basket[mask])
        try:
            _, adf_p, *_ = adfuller(resid, autolag="AIC")
        except Exception:
            adf_p = float("nan")
        rows.append(dict(
            scope=scope_name,
            product=SHORT[p],
            beta_basket=float(b),
            r2=float(m.rsquared),
            resid_std=float(resid.std(ddof=1)),
            adf_p=float(adf_p),
            stationary=bool(adf_p < 0.05),
            half_life=half_life(resid),
        ))
    return pd.DataFrame(rows)


# ---------- plots --------------------------------------------------------

def plot_levels(panel: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(DAYS), 1, figsize=(11, 8), sharex=False)
    if len(DAYS) == 1:
        axes = [axes]
    for ax, d in zip(axes, DAYS):
        g = panel[panel["day"] == d]
        for p in PRODUCTS:
            # z-score within day so all 5 fit on one axis
            v = g[p].values.astype(float)
            v_n = (v - np.nanmean(v)) / np.nanstd(v)
            ax.plot(g["timestamp"].values, v_n, lw=0.5, label=SHORT[p])
        ax.set_title(f"day {d}: z-scored mids (within-day)")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("z-score")
        ax.axhline(0, color="k", lw=0.4)
        ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pair_spread(panel: pd.DataFrame, y_name: str, x_name: str, path: Path) -> None:
    fig, axes = plt.subplots(len(DAYS), 1, figsize=(11, 8), sharex=False)
    if len(DAYS) == 1:
        axes = [axes]
    for ax, d in zip(axes, DAYS):
        g = panel[panel["day"] == d]
        y = g[y_name].dropna().values
        x = g[x_name].dropna().values
        n = min(len(y), len(x))
        if n < 50:
            continue
        X = add_constant(x[:n])
        m = OLS(y[:n], X).fit()
        a, b = m.params
        resid = y[:n] - (a + b * x[:n])
        z = (resid - resid.mean()) / resid.std(ddof=1)
        ax.plot(z, lw=0.5, color="C2")
        for k, ls in ((1.5, ":"), (2.0, "--"), (3.0, "-.")):
            ax.axhline(k, color="grey", ls=ls, lw=0.6)
            ax.axhline(-k, color="grey", ls=ls, lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"day {d}: spread z-score · β={b:.4f}")
        ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_basket_resids(panel: pd.DataFrame, path: Path) -> None:
    z = panel[PRODUCTS].copy()
    z = (z - z.mean()) / z.std(ddof=1)
    fig, axes = plt.subplots(len(PRODUCTS), 1, figsize=(11, 9), sharex=True)
    for ax, p in zip(axes, PRODUCTS):
        others = [q for q in PRODUCTS if q != p]
        basket = z[others].mean(axis=1).values
        target = z[p].values
        mask = np.isfinite(basket) & np.isfinite(target)
        X = add_constant(basket[mask])
        m = OLS(target[mask], X).fit()
        a, b = m.params
        resid = target[mask] - (a + b * basket[mask])
        ax.plot(resid, lw=0.45, color="C0")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{SHORT[p]}: residual vs basket-of-other-4 (β={b:.3f}, R²={m.rsquared:.3f})")
        ax.set_ylabel("resid (z units)")
    axes[-1].set_xlabel("tick index (stitched across days)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------- driver -------------------------------------------------------

def main():
    panel = all_days()
    n_total = len(panel)
    print(f"Loaded {n_total} timestamps · days {DAYS} · 5 Galaxy Sounds products")
    for p in PRODUCTS:
        s = panel[p]
        print(f"  {SHORT[p]:18}  n={s.notna().sum():6d}  "
              f"mean={s.mean():9.2f}  std={s.std():7.2f}  "
              f"min={s.min():7.0f}  max={s.max():7.0f}")

    print("\n" + "=" * 72)
    print("STEP 1 · return correlation matrix (lag=1, stitched across days)")
    print("=" * 72)
    for lag in (1, 5, 50):
        c = corr_matrix_at_lag(panel, lag)
        c.index = [SHORT[p] for p in c.index]
        c.columns = [SHORT[p] for p in c.columns]
        print(f"\n  lag={lag}  (n={len(returns_at_lag(panel, lag))})")
        with pd.option_context("display.float_format", "{:+.3f}".format,
                               "display.width", 200):
            print(c.to_string())

    print("\n  Per-day check (lag=1) — only print pairs where the sign FLIPS or |corr|<0.1 vs stitched:")
    stitched = corr_matrix_at_lag(panel, 1)
    flips = []
    for d in DAYS:
        sub = panel[panel["day"] == d]
        cd = corr_matrix_at_lag(sub, 1)
        for i, a in enumerate(PRODUCTS):
            for b in PRODUCTS[i + 1:]:
                s = stitched.loc[a, b]; v = cd.loc[a, b]
                if (np.sign(s) != np.sign(v) and abs(s) > 0.05) or abs(v) < 0.1 < abs(s):
                    flips.append(dict(day=d, pair=f"{SHORT[a]}↔{SHORT[b]}",
                                       stitched=s, this_day=v))
    if flips:
        print(pd.DataFrame(flips).to_string(index=False))
    else:
        print("    (none — sign and magnitude of return-correlation is stable across days)")

    print("\n" + "=" * 72)
    print("STEP 2-4 · pairwise OLS  Y = α + β·X  →  residual ADF + half-life")
    print("=" * 72)
    eg_all = pairwise_eg(panel, "all")
    eg_per = pd.concat([
        pairwise_eg(panel[panel["day"] == d], f"day{d}") for d in DAYS
    ], ignore_index=True)
    eg = pd.concat([eg_all, eg_per], ignore_index=True)
    show = eg[["scope", "y", "x", "beta", "ret_corr", "resid_std",
               "adf_p", "stationary", "half_life"]].copy()
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.max_rows", None, "display.width", 200):
        # sort: stationary first, then by adf_p ascending
        show = show.sort_values(by=["scope", "adf_p"]).reset_index(drop=True)
        print(show.to_string(index=False))

    print("\n  Cointegrated pairs in 'all' scope (adf_p < 0.05):")
    coint = eg_all[eg_all["adf_p"] < 0.05].sort_values("adf_p")
    if coint.empty:
        print("    (none cointegrated on stitched sample)")
    else:
        print(coint[["y", "x", "beta", "ret_corr", "adf_p", "half_life"]]
              .to_string(index=False))

    print("\n" + "=" * 72)
    print("STEP 5 · out-of-sample for the focus pair BLACK_HOLES vs SOLAR_FLAMES")
    print("=" * 72)
    oos = out_of_sample_pair(panel,
                             "GALAXY_SOUNDS_BLACK_HOLES",
                             "GALAXY_SOUNDS_SOLAR_FLAMES")
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.width", 200):
        print(oos.to_string(index=False))

    print("\n" + "=" * 72)
    print("STEP 6 · basket residual structure (each product vs equal-weight")
    print("        z-score basket of the other 4)")
    print("=" * 72)
    bk_all = basket_residuals(panel, "all")
    bk_per = pd.concat([
        basket_residuals(panel[panel["day"] == d], f"day{d}") for d in DAYS
    ], ignore_index=True)
    bk = pd.concat([bk_all, bk_per], ignore_index=True)
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.max_rows", None, "display.width", 200):
        print(bk.to_string(index=False))

    print("\n  Stationary basket-residuals in 'all' scope:")
    s = bk_all[bk_all["adf_p"] < 0.05]
    if s.empty:
        print("    (none) — no product mean-reverts cleanly to the basket")
    else:
        print(s[["product", "beta_basket", "r2", "adf_p", "half_life"]]
              .to_string(index=False))

    print("\n" + "=" * 72)
    print("Plots →")
    print("=" * 72)
    plot_levels(panel, OUT / "galaxy_sounds_levels.png")
    plot_pair_spread(panel,
                     "GALAXY_SOUNDS_BLACK_HOLES",
                     "GALAXY_SOUNDS_SOLAR_FLAMES",
                     OUT / "galaxy_sounds_spread_BH_SF.png")
    plot_basket_resids(panel, OUT / "galaxy_sounds_basket_residuals.png")
    print(f"  {OUT/'galaxy_sounds_levels.png'}")
    print(f"  {OUT/'galaxy_sounds_spread_BH_SF.png'}")
    print(f"  {OUT/'galaxy_sounds_basket_residuals.png'}")


if __name__ == "__main__":
    main()
