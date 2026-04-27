"""
Does bid-side volume on OTM voucher strikes lead price?

Adds:
- Resting-book bid_volume_1 (best bid level only — that's what seller bot hits)
- Signed trade flow (price <= bid → sell-print; price >= ask → buy-print)
- Quintile-bucketed forward returns (if relationship is non-linear)
- Cross-correlation: voucher bid-vol vs VFE return at lags
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HIST = REPO / "historical" / "ROUND_3"

OTM_STRIKES = [5300, 5400, 5500, 6000, 6500]
PRODUCTS = [f"VEV_{k}" for k in OTM_STRIKES] + ["VELVETFRUIT_EXTRACT"]


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(HIST / f"prices_round_3_day_{day}.csv", sep=";")
    df = df[df["product"].isin(PRODUCTS)].copy()
    for col in ["bid_volume_1", "bid_volume_2", "bid_volume_3",
                "ask_volume_1", "ask_volume_2", "ask_volume_3",
                "bid_price_1", "ask_price_1"]:
        df[col] = df[col].fillna(0)
    df["bid_vol_total"] = (df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"]).astype(int)
    df["ask_vol_total"] = (df["ask_volume_1"] + df["ask_volume_2"] + df["ask_volume_3"]).astype(int)
    df["bid_vol_1"] = df["bid_volume_1"].astype(int)
    df["ask_vol_1"] = df["ask_volume_1"].astype(int)
    return df


def load_trades(day: int) -> pd.DataFrame:
    return pd.read_csv(HIST / f"trades_round_3_day_{day}.csv", sep=";")


def pivot_tick(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot(index="timestamp", columns="product", values=value).sort_index()


def signed_trade_flow(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """For each (timestamp, product): qty signed +sell-side / -buy-side.
    Sell-print: trade price ≤ bid_1 (seller hit bid). Buy-print: ≥ ask_1.
    """
    bid = pivot_tick(prices, "bid_price_1")
    ask = pivot_tick(prices, "ask_price_1")
    rows = []
    for sym in [f"VEV_{k}" for k in OTM_STRIKES] + ["VELVETFRUIT_EXTRACT"]:
        if sym not in bid.columns: continue
        sub = trades[trades["symbol"] == sym].copy()
        if sub.empty: continue
        sub = sub.merge(
            pd.DataFrame({"timestamp": bid.index, "bid": bid[sym].values, "ask": ask[sym].values}),
            on="timestamp", how="left"
        )
        sub["side"] = np.where(sub["price"] <= sub["bid"], +1,
                       np.where(sub["price"] >= sub["ask"], -1, 0))
        sub["signed_qty"] = sub["side"] * sub["quantity"]
        agg = sub.groupby("timestamp").agg(
            signed=("signed_qty", "sum"),
            sell_qty=("quantity", lambda s: s[sub.loc[s.index, "side"] == 1].sum()),
            buy_qty=("quantity", lambda s: s[sub.loc[s.index, "side"] == -1].sum()),
        ).reset_index()
        agg["product"] = sym
        rows.append(agg)
    return pd.concat(rows) if rows else pd.DataFrame()


def fwd_ret(s: pd.Series, k: int) -> pd.Series:
    return s.shift(-k) - s


def lvl1_correlations(day: int, horizons=(1, 5, 10, 50)) -> None:
    df = load_prices(day)
    bid1 = pivot_tick(df, "bid_vol_1")
    ask1 = pivot_tick(df, "ask_vol_1")
    mids = pivot_tick(df, "mid_price")
    vfe = mids.get("VELVETFRUIT_EXTRACT")
    if vfe is None: return
    print(f"\n=== Day {day} — best-bid volume (lvl-1 only) vs fwd return ===")
    print(f"{'strike':>6} {'k':>4} {'corr(b1,vRet)':>15} {'corr(b1,vfeRet)':>17} "
          f"{'corr(b1-a1,vRet)':>17}")
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        if sym not in bid1.columns: continue
        b = bid1[sym]
        a = ask1[sym]
        imb1 = b - a
        m = mids[sym].ffill()
        for k in horizons:
            vret = fwd_ret(m, k); sret = fwd_ret(vfe, k)
            mask = b.notna() & vret.notna() & sret.notna()
            if mask.sum() < 50: continue
            cb_v = np.corrcoef(b[mask], vret[mask])[0, 1] if vret[mask].std() else np.nan
            cb_s = np.corrcoef(b[mask], sret[mask])[0, 1] if sret[mask].std() else np.nan
            ci_v = np.corrcoef(imb1[mask], vret[mask])[0, 1] if vret[mask].std() else np.nan
            print(f"{K:>6} {k:>4} {cb_v:>+15.4f} {cb_s:>+17.4f} {ci_v:>+17.4f}")


def trade_flow_correlations(day: int) -> None:
    prices = load_prices(day)
    trades = load_trades(day)
    flow = signed_trade_flow(prices, trades)
    if flow.empty:
        print(f"Day {day}: no trade flow"); return
    mids = pivot_tick(prices, "mid_price")
    vfe = mids.get("VELVETFRUIT_EXTRACT")
    print(f"\n=== Day {day} — printed-trade signed flow vs fwd return ===")
    print(f"  +flow = sell-print (price ≤ bid); -flow = buy-print (≥ ask)")
    print(f"{'strike':>6} {'n_print':>9} {'mean_signed':>12} {'sell_total':>11} "
          f"{'buy_total':>10} {'k':>3} {'corr(flow,vRet)':>16} {'corr(flow,vfeRet)':>18}")
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        sub = flow[flow["product"] == sym]
        if sub.empty: continue
        # Reindex flow to all timestamps (zero where no print)
        all_ts = mids.index
        f = sub.set_index("timestamp")["signed"].reindex(all_ts).fillna(0)
        m = mids[sym].ffill() if sym in mids else None
        for k in (1, 5, 10, 50):
            if m is None: continue
            vret = fwd_ret(m, k)
            sret = fwd_ret(vfe, k)
            mask = vret.notna() & sret.notna() & (f.values != 0 | vret.notna())
            if mask.sum() < 50: continue
            try:
                cf_v = np.corrcoef(f[mask], vret[mask])[0, 1]
                cf_s = np.corrcoef(f[mask], sret[mask])[0, 1]
            except Exception:
                cf_v = cf_s = np.nan
            print(f"{K:>6} {len(sub):>9} {sub['signed'].sum()/len(sub):>+12.2f} "
                  f"{int(sub['sell_qty'].sum()):>11} {int(sub['buy_qty'].sum()):>10} "
                  f"{k:>3} {cf_v:>+16.4f} {cf_s:>+18.4f}")


def quintile_buckets(day: int, k_horizon: int = 5) -> None:
    """Bucket bid_vol_1 into quintiles; report mean fwd return per bucket."""
    df = load_prices(day)
    bid1 = pivot_tick(df, "bid_vol_1")
    mids = pivot_tick(df, "mid_price")
    vfe = mids.get("VELVETFRUIT_EXTRACT")
    print(f"\n=== Day {day} — quintile of bid_vol_1, fwd return at k={k_horizon} ===")
    print(f"{'strike':>6} {'q':>3} {'n':>6} {'b1_avg':>8} "
          f"{'mean_vRet':>10} {'mean_vfeRet':>12}")
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        if sym not in bid1.columns: continue
        b = bid1[sym]
        m = mids[sym].ffill()
        vret = fwd_ret(m, k_horizon)
        sret = fwd_ret(vfe, k_horizon) if vfe is not None else pd.Series(np.nan, index=m.index)
        df_q = pd.DataFrame({"b": b, "vret": vret, "sret": sret}).dropna()
        if df_q.empty: continue
        df_q["bucket"] = pd.qcut(df_q["b"], 5, labels=False, duplicates="drop")
        for q, sub in df_q.groupby("bucket"):
            print(f"{K:>6} {int(q):>3} {len(sub):>6} {sub['b'].mean():>8.1f} "
                  f"{sub['vret'].mean():>+10.4f} {sub['sret'].mean():>+12.4f}")
        print()


def cross_lag_vfe(day: int) -> None:
    """Maybe voucher bid_vol leads VFE? Or vice versa. Scan lags ±20."""
    df = load_prices(day)
    bid1 = pivot_tick(df, "bid_vol_1")
    mids = pivot_tick(df, "mid_price")
    vfe = mids.get("VELVETFRUIT_EXTRACT")
    if vfe is None: return
    print(f"\n=== Day {day} — cross-correlation: bid_vol_1[t] vs VFE_ret[t+lag] ===")
    print(f"{'strike':>6}", end="")
    lags = [-10, -5, -1, 0, 1, 5, 10, 50]
    for L in lags: print(f" {L:>+8}", end="")
    print()
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        if sym not in bid1.columns: continue
        b = bid1[sym]
        print(f"{K:>6}", end="")
        for L in lags:
            vfe_ret = vfe.diff().shift(-L)
            mask = b.notna() & vfe_ret.notna()
            if mask.sum() < 50:
                print(f" {'  ':>8}", end="")
                continue
            c = np.corrcoef(b[mask], vfe_ret[mask])[0, 1] if vfe_ret[mask].std() else np.nan
            print(f" {c:>+8.4f}", end="")
        print()


if __name__ == "__main__":
    days = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2]
    for d in days:
        lvl1_correlations(d)
        trade_flow_correlations(d)
        quintile_buckets(d, k_horizon=5)
        cross_lag_vfe(d)
