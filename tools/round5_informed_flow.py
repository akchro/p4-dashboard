"""Round 5 — search for informed-trader footprints across all 50 products.

Round 5's historical CSVs don't tag bot-vs-bot trades with party IDs, so we
can't filter by name. Instead, look for recurring (product, qty) cells where
the AGGRESSOR direction predicts the next H ticks of mid movement.

For each market trade we infer aggressor side by comparing trade price to the
prevailing book:
  trade_price >= ask_1   ->  +1  (buyer-aggressive: someone took the offer)
  trade_price <= bid_1   ->  -1  (seller-aggressive: someone hit the bid)
  else                   ->   0  (mid/uncertain — discard)

Then for each trade compute mid-return at horizons {100, 1000, 10000} ticks.
The "informativeness" of a (product, qty) bucket is:
    edge = mean(aggressor_dir * fwd_return)
A positive edge means buys at this size precede price ups (and sells precede
downs) — the textbook informed-trader fingerprint.

Reported: top buckets by edge × sqrt(n), with H = 1000 ticks as the focal
horizon (long enough to be tradable, short enough to be statistically dense).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "historical" / "ROUND_5"
DAYS = [2, 3, 4]

HORIZONS = [100, 1_000, 10_000]
FOCAL_H = 1_000

# Min trades for a (product, qty) bucket to be worth ranking
MIN_BUCKET_N = 20


def load_day(d: int):
    a = pd.read_csv(DATA / f"prices_round_5_day_{d}.csv", sep=";")
    a = a[["timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"]].copy()
    a["day"] = d
    t = pd.read_csv(DATA / f"trades_round_5_day_{d}.csv", sep=";")
    t = t.rename(columns={"symbol": "product"})
    t["day"] = d
    return a, t


def build_panel():
    a_parts, t_parts = [], []
    for d in DAYS:
        a, t = load_day(d)
        a_parts.append(a); t_parts.append(t)
    activities = pd.concat(a_parts, ignore_index=True)
    trades = pd.concat(t_parts, ignore_index=True)

    # Join trade -> book at same (day, timestamp, product). Inner-join on
    # exact timestamp; trades hit the same tick the snapshot was taken.
    trades = trades.merge(
        activities[["day", "timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"]],
        on=["day", "timestamp", "product"], how="left",
    )
    return activities, trades


def classify_aggressor(trades: pd.DataFrame) -> pd.DataFrame:
    """+1 buyer-aggressive, -1 seller-aggressive, 0 uncertain."""
    px = trades["price"].values
    bid = trades["bid_price_1"].values
    ask = trades["ask_price_1"].values
    side = np.zeros(len(trades), dtype=np.int8)
    side[px >= ask] = +1
    side[px <= bid] = -1
    trades = trades.copy()
    trades["aggressor"] = side
    return trades


def attach_forward_returns(activities: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """For each trade, look up the mid_price at timestamp + H within the
    same day & product. Forward-fill is fine for the small fraction of ticks
    that are missing — the historical step is regular at 100ts."""
    trades = trades.copy()
    # Build per-(day,product) mid lookup as a sorted Series indexed by ts
    by_key = {}
    for (d, p), grp in activities.dropna(subset=["mid_price"]).groupby(["day", "product"]):
        by_key[(d, p)] = grp.set_index("timestamp")["mid_price"].sort_index()

    for H in HORIZONS:
        col = f"fwd_ret_{H}"
        out = np.full(len(trades), np.nan)
        for i, row in enumerate(trades.itertuples(index=False)):
            key = (row.day, row.product)
            s = by_key.get(key)
            if s is None:
                continue
            target_ts = row.timestamp + H
            # Snap to nearest existing ts at or after target
            idx = s.index.searchsorted(target_ts)
            if idx >= len(s):
                continue
            future_mid = s.iloc[idx]
            now_mid = row.mid_price
            if pd.isna(now_mid):
                continue
            out[i] = future_mid - now_mid
        trades[col] = out
    return trades


def informativeness(trades: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Bucket by (product, quantity, aggressor sign) and report:
       n, mean signed forward return, edge = sign * mean_ret.
    Aggressor is per-trade so we collapse to per-bucket avg by signing the
    forward return: signed = aggressor * fwd_ret."""
    fwd = f"fwd_ret_{horizon}"
    df = trades[trades["aggressor"] != 0].dropna(subset=[fwd]).copy()
    df["signed_fwd"] = df["aggressor"] * df[fwd]

    g = df.groupby(["product", "quantity"]).agg(
        n=("signed_fwd", "size"),
        edge=("signed_fwd", "mean"),
        std=("signed_fwd", "std"),
        n_buy=("aggressor", lambda x: int((x > 0).sum())),
        n_sell=("aggressor", lambda x: int((x < 0).sum())),
    ).reset_index()
    g = g[g["n"] >= MIN_BUCKET_N]
    g["t_stat"] = g["edge"] / (g["std"] / np.sqrt(g["n"])).replace(0, np.nan)
    g["score"] = g["edge"] * np.sqrt(g["n"])
    return g.sort_values("score", ascending=False)


def per_product_aggregate(trades: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """All sizes combined, per product."""
    fwd = f"fwd_ret_{horizon}"
    df = trades[trades["aggressor"] != 0].dropna(subset=[fwd]).copy()
    df["signed_fwd"] = df["aggressor"] * df[fwd]
    g = df.groupby("product").agg(
        n=("signed_fwd", "size"),
        edge=("signed_fwd", "mean"),
        std=("signed_fwd", "std"),
        n_buy=("aggressor", lambda x: int((x > 0).sum())),
        n_sell=("aggressor", lambda x: int((x < 0).sum())),
    ).reset_index()
    g["t_stat"] = g["edge"] / (g["std"] / np.sqrt(g["n"])).replace(0, np.nan)
    return g.sort_values("t_stat", ascending=False)


def main():
    print("Loading round 5 trades + book ...")
    activities, trades = build_panel()
    print(f"  activities: {len(activities):,}   trades: {len(trades):,}")

    trades = classify_aggressor(trades)
    n_buy = int((trades["aggressor"] > 0).sum())
    n_sell = int((trades["aggressor"] < 0).sum())
    n_amb = int((trades["aggressor"] == 0).sum())
    print(f"  aggressor classified: +1 buy {n_buy:,}  -1 sell {n_sell:,}"
          f"  0 ambiguous {n_amb:,}")

    print(f"\nAttaching forward returns at horizons {HORIZONS} ticks ...")
    trades = attach_forward_returns(activities, trades)

    # Quantity distribution — first sanity that there's any specific recurring size
    print("\n" + "=" * 72)
    print("Quantity distribution — does any size dominate? (top 25)")
    print("=" * 72)
    qd = trades["quantity"].value_counts().head(25).reset_index()
    qd.columns = ["qty", "n_trades"]
    qd["%"] = qd["n_trades"] / len(trades) * 100
    print(qd.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"Per-product aggressor edge at H={FOCAL_H} ticks (all sizes)")
    print("=" * 72)
    pp = per_product_aggregate(trades, FOCAL_H)
    print(pp.head(15).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n... (bottom 5 — anti-informed flow, also interesting)")
    print(pp.tail(5).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n" + "=" * 72)
    print(f"Top (product, qty) buckets by edge × sqrt(n) at H={FOCAL_H}")
    print(f"(min n={MIN_BUCKET_N} per bucket; positive edge = aggressor-side")
    print(" predicts the next {} ticks of mid move)".format(FOCAL_H))
    print("=" * 72)
    g = informativeness(trades, FOCAL_H)
    show_cols = ["product", "quantity", "n", "n_buy", "n_sell",
                 "edge", "std", "t_stat", "score"]
    with pd.option_context("display.float_format", "{:+.3f}".format,
                           "display.max_rows", 40, "display.width", 200):
        print(g.head(30)[show_cols].to_string(index=False))

    print("\n" + "=" * 72)
    print(f"Bottom (product, qty) buckets at H={FOCAL_H} (anti-informed; aggressor")
    print(" side LOSES — possible noise traders or stale orders)")
    print("=" * 72)
    with pd.option_context("display.float_format", "{:+.3f}".format,
                           "display.max_rows", 20, "display.width", 200):
        print(g.tail(15)[show_cols].to_string(index=False))

    print("\n" + "=" * 72)
    print(f"Same scan, broken out by horizon (top 10 each)")
    print("=" * 72)
    for H in HORIZONS:
        gh = informativeness(trades, H)
        print(f"\n--- H={H} ---")
        with pd.option_context("display.float_format", "{:+.3f}".format,
                               "display.width", 200):
            print(gh.head(10)[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
