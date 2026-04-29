"""Stacked predictor for MICROCHIP followers.

Hypothesis (per user feedback):
  Predictors for ret_F(t) include not just ret_CIRCLE(t - LAG_F), but ALSO:
    - ret_CIRCLE at neighboring lags (LAG_F ± few ticks)
    - ret_X(t - (LAG_F - LAG_X)) for every other product X that leads F

Since:
  OVAL      ≈ CIRCLE shifted +50
  SQUARE    ≈ CIRCLE shifted +100
  RECTANGLE ≈ CIRCLE shifted +150
  TRIANGLE  ≈ CIRCLE shifted +200

The graduated chain means each downstream follower has multiple upstream
informants. For predicting ret_TRIANGLE(t):
  - ret_CIRCLE(t-200)    (original 200-tick signal)
  - ret_OVAL(t-150)      (OVAL leads TRIANGLE by 150)
  - ret_SQUARE(t-100)    (SQUARE leads TRIANGLE by 100)
  - ret_RECTANGLE(t-50)  (RECTANGLE leads TRIANGLE by 50)

Each shares the same underlying CIRCLE-N signal but with independent noise,
so combining via OLS should reduce variance.

This script fits the regressions on day 2-3, evaluates R² on day 4.
"""
from __future__ import annotations
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

CHAIN_ORDER = ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
               "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"]
LAGS = {p: i * 50 for i, p in enumerate(CHAIN_ORDER)}  # ticks ahead of CIRCLE


def load_pivot() -> pd.DataFrame:
    dfs = []
    for d in DAYS:
        f = f"historical/ROUND_5/prices_round_5_day_{d}.csv"
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    return p.pivot_table(index="t", columns="product", values="mid_price",
                         aggfunc="first").sort_index().dropna(how="any")


def predictors_for(target: str) -> List[Tuple[str, int]]:
    """Return list of (source_product, lag_in_ticks) that predict target's
    1-tick return. Lag is the number of ticks BACK to look in source."""
    target_lag = LAGS[target]
    out = []
    for src in CHAIN_ORDER:
        src_lag = LAGS[src]
        if src_lag >= target_lag:
            continue  # src is at or downstream of target — useless
        # src leads target by (target_lag - src_lag) ticks
        # so src's return at t-(target_lag - src_lag) predicts target's return at t
        gap = target_lag - src_lag
        # Add small lag-window for fuzz
        for delta in (-1, 0, +1):
            out.append((src, gap + delta))
    return out


def main():
    pivot = load_pivot()
    rets = pivot.diff()
    print(f"data: {len(pivot)} ticks, {len(rets)-1} returns", file=sys.stderr)

    # Build feature matrix per follower, fit OLS on day 2-3, eval on day 4
    print(f"\n{'follower':22s} {'n_predictors':>12s}  {'R²(IS)':>9s}  {'R²(OOS)':>9s}  "
          f"{'baseline R²(OOS)':>16s}")

    train_mask = (rets.index // TIMESTAMP_PER_DAY) < (DAYS[2] - DAYS[0])
    test_mask  = (rets.index // TIMESTAMP_PER_DAY) == (DAYS[2] - DAYS[0])

    summary = {}
    for follower in CHAIN_ORDER[1:]:  # skip CIRCLE itself
        # Build features
        preds = predictors_for(follower)
        target_lag = LAGS[follower]
        feature_names = [f"{src}_{lag}" for src, lag in preds]
        X = pd.DataFrame(index=rets.index, columns=feature_names, dtype=float)
        for (src, lag), name in zip(preds, feature_names):
            X[name] = rets[src].shift(lag)
        y = rets[follower]
        df = pd.concat([X, y.rename("y")], axis=1).dropna()
        # Train/test split via per-day boundary on df.index
        is_train = (df.index // TIMESTAMP_PER_DAY) < (DAYS[2] - DAYS[0])
        is_test  = (df.index // TIMESTAMP_PER_DAY) == (DAYS[2] - DAYS[0])
        df_tr = df[is_train]
        df_te = df[is_test]
        Xtr = df_tr[feature_names].values; ytr = df_tr["y"].values
        Xte = df_te[feature_names].values; yte = df_te["y"].values

        # OLS (closed-form)
        Xtr_b = np.column_stack([np.ones(len(Xtr)), Xtr])
        Xte_b = np.column_stack([np.ones(len(Xte)), Xte])
        beta, *_ = np.linalg.lstsq(Xtr_b, ytr, rcond=None)
        ytr_pred = Xtr_b @ beta
        yte_pred = Xte_b @ beta
        r2_is  = 1 - np.var(ytr - ytr_pred) / np.var(ytr)
        r2_oos = 1 - np.var(yte - yte_pred) / np.var(yte)

        # Baseline: single predictor (the original signal)
        single_lag = LAGS[follower] - LAGS["MICROCHIP_CIRCLE"]
        X_single = rets["MICROCHIP_CIRCLE"].shift(single_lag).rename("x")
        df_s = pd.concat([X_single, y.rename("y")], axis=1).dropna()
        is_train_s = (df_s.index // TIMESTAMP_PER_DAY) < (DAYS[2] - DAYS[0])
        is_test_s  = (df_s.index // TIMESTAMP_PER_DAY) == (DAYS[2] - DAYS[0])
        df_s_tr = df_s[is_train_s]
        df_s_te = df_s[is_test_s]
        Xtr_s = np.column_stack([np.ones(len(df_s_tr)), df_s_tr["x"].values])
        Xte_s = np.column_stack([np.ones(len(df_s_te)), df_s_te["x"].values])
        beta_s, *_ = np.linalg.lstsq(Xtr_s, df_s_tr["y"].values, rcond=None)
        yte_pred_s = Xte_s @ beta_s
        r2_oos_baseline = 1 - np.var(df_s_te["y"].values - yte_pred_s) / np.var(df_s_te["y"].values)

        summary[follower] = {
            "predictors": len(feature_names),
            "r2_is": r2_is,
            "r2_oos": r2_oos,
            "r2_oos_baseline": r2_oos_baseline,
            "beta": beta,
            "feature_names": feature_names,
        }
        print(f"{follower:22s} {len(feature_names):>12d}  {r2_is:>9.5f}  {r2_oos:>9.5f}  {r2_oos_baseline:>16.5f}")

    # Show the regression coefficients for SQUARE and TRIANGLE (most-stacked)
    print(f"\nLearned coefficients for the most-stacked followers:")
    for follower in ["MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"]:
        info = summary[follower]
        print(f"\n  {follower}:")
        print(f"    intercept = {info['beta'][0]:+.5f}")
        for name, b in zip(info["feature_names"], info["beta"][1:]):
            print(f"    {name:38s}  β = {b:+.5f}")

    # Predicted-edge magnitudes
    print(f"\nPredicted edge per σ of CIRCLE move (single vs stacked):")
    sig_C = rets["MICROCHIP_CIRCLE"].std()
    for follower in CHAIN_ORDER[1:]:
        info = summary[follower]
        sig_F = rets[follower].std()
        # baseline beta = corr * sigF/sigC
        # stacked: total prediction power
        # Use standard deviation of stacked prediction divided by σ_F
        # to get analogous "predicted move per σ of typical input"
        single_corr = np.sqrt(info["r2_oos_baseline"]) if info["r2_oos_baseline"] > 0 else 0
        stack_corr = np.sqrt(info["r2_oos"]) if info["r2_oos"] > 0 else 0
        print(f"  {follower:22s} single |ρ|={single_corr:.4f}  stacked |ρ|={stack_corr:.4f}  "
              f"uplift = {stack_corr - single_corr:+.4f}  ({stack_corr/max(single_corr,1e-6):.2f}×)")


if __name__ == "__main__":
    main()
