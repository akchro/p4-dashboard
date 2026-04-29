"""Profile counterparty behavior in ROUND_4.

Builds a per-Mark behavioral profile: how often they're aggressor vs. provider,
average trade size, product mix, sided-ness (buy/sell skew), and book-vs-trade
price behavior. Run from repo root:

    python3 tools/counterparty_profile.py

Outputs a per-Mark summary and a rough classification heuristic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROUND4 = ROOT / "historical" / "ROUND_4"
DAYS = (1, 2, 3)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(ROUND4 / f"trades_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(ROUND4 / f"prices_round_4_day_{d}.csv", sep=";")
        # day column already in csv, but re-stamp to be safe
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def annotate_aggressor(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """For each trade, look up the prevailing book and infer aggressor.

    Aggressor = the side that crossed the spread.
    - price >= ask_1 → buyer was aggressor (lifted the offer)
    - price <= bid_1 → seller was aggressor (hit the bid)
    - bid_1 < price < ask_1 → ambiguous (price improvement / mid trade)
    """
    book = prices.set_index(["day", "timestamp", "product"])[
        ["bid_price_1", "ask_price_1", "mid_price"]
    ]
    # join: trades.symbol == prices.product
    j = trades.merge(
        book.reset_index().rename(columns={"product": "symbol"}),
        on=["day", "timestamp", "symbol"],
        how="left",
    )

    bid = j["bid_price_1"]
    ask = j["ask_price_1"]
    px = j["price"]
    mid = j["mid_price"]

    aggressor = np.where(
        px >= ask,
        "buyer",
        np.where(
            px <= bid,
            "seller",
            np.where(px > mid, "buyer", np.where(px < mid, "seller", "ambiguous")),
        ),
    )
    j["aggressor"] = aggressor
    j["spread_pos"] = (px - mid) / np.where((ask - bid) > 0, (ask - bid) / 2, np.nan)
    return j


def per_trader_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    traders = sorted(set(trades["buyer"]).union(trades["seller"]))
    for m in traders:
        as_buyer = trades[trades["buyer"] == m]
        as_seller = trades[trades["seller"] == m]

        n_buy = len(as_buyer)
        n_sell = len(as_seller)
        n_total = n_buy + n_sell

        qty_buy = as_buyer["quantity"].sum()
        qty_sell = as_seller["quantity"].sum()

        # aggression: when this Mark was the buyer, was buyer the aggressor?
        buy_agg = (as_buyer["aggressor"] == "buyer").sum()
        buy_pas = (as_buyer["aggressor"] == "seller").sum()  # someone hit our bid
        sell_agg = (as_seller["aggressor"] == "seller").sum()
        sell_pas = (as_seller["aggressor"] == "buyer").sum()  # someone lifted our offer

        n_aggressor = buy_agg + sell_agg
        n_passive = buy_pas + sell_pas
        n_ambig = n_total - n_aggressor - n_passive

        agg_rate = n_aggressor / n_total if n_total else float("nan")
        pas_rate = n_passive / n_total if n_total else float("nan")

        # product mix
        prods = pd.concat([as_buyer["symbol"], as_seller["symbol"]])
        opts_share = prods.str.startswith("VEV_").mean()
        underl_share = (prods == "VELVETFRUIT_EXTRACT").mean()
        hydro_share = (prods == "HYDROGEL_PACK").mean()

        # avg size
        avg_qty = pd.concat([as_buyer["quantity"], as_seller["quantity"]]).mean()
        max_qty = pd.concat([as_buyer["quantity"], as_seller["quantity"]]).max()

        # buy/sell skew
        if n_buy + n_sell:
            buy_share = n_buy / (n_buy + n_sell)
        else:
            buy_share = float("nan")

        # net signed quantity (>0 means net buyer over the period)
        net_qty = qty_buy - qty_sell

        # spread position when passive (should be near ±1, i.e. on the quote)
        all_trades = pd.concat([as_buyer, as_seller])
        passive_mask = ((all_trades["buyer"] == m) & (all_trades["aggressor"] == "seller")) | (
            (all_trades["seller"] == m) & (all_trades["aggressor"] == "buyer")
        )
        passive_trades = all_trades[passive_mask]
        # Average distance from mid (in half-spreads). MMs sit near ±1.
        passive_spread_pos = passive_trades["spread_pos"].abs().mean()

        rows.append(
            dict(
                trader=m,
                n_trades=n_total,
                n_buy=n_buy,
                n_sell=n_sell,
                buy_share=round(buy_share, 3),
                qty_total=int(qty_buy + qty_sell),
                net_qty=int(net_qty),
                avg_qty=round(avg_qty, 2),
                max_qty=int(max_qty),
                aggressor_rate=round(agg_rate, 3),
                passive_rate=round(pas_rate, 3),
                ambig_rate=round(n_ambig / n_total, 3) if n_total else float("nan"),
                opts_share=round(opts_share, 3),
                underl_share=round(underl_share, 3),
                hydro_share=round(hydro_share, 3),
                passive_dist_halfspread=round(passive_spread_pos, 3)
                if not np.isnan(passive_spread_pos)
                else float("nan"),
            )
        )
    return pd.DataFrame(rows).sort_values("n_trades", ascending=False)


def per_trader_per_product(trades: pd.DataFrame) -> pd.DataFrame:
    """Product mix per trader."""
    rows = []
    traders = sorted(set(trades["buyer"]).union(trades["seller"]))
    for m in traders:
        as_buyer = trades[trades["buyer"] == m]
        as_seller = trades[trades["seller"] == m]
        for prod in sorted(set(trades["symbol"])):
            nb = (as_buyer["symbol"] == prod).sum()
            ns = (as_seller["symbol"] == prod).sum()
            if nb + ns == 0:
                continue
            rows.append(dict(trader=m, product=prod, n_buy=int(nb), n_sell=int(ns), n_total=int(nb + ns)))
    return pd.DataFrame(rows)


def pairings(trades: pd.DataFrame) -> pd.DataFrame:
    """Who trades against whom?"""
    p = trades.groupby(["buyer", "seller"]).size().reset_index(name="n")
    return p.sort_values("n", ascending=False)


def trade_size_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Quantity distribution per trader."""
    rows = []
    traders = sorted(set(trades["buyer"]).union(trades["seller"]))
    for m in traders:
        qty = pd.concat(
            [trades.loc[trades["buyer"] == m, "quantity"], trades.loc[trades["seller"] == m, "quantity"]]
        )
        rows.append(
            dict(
                trader=m,
                n=len(qty),
                q_min=int(qty.min()),
                q_p25=int(qty.quantile(0.25)),
                q_med=int(qty.median()),
                q_mean=round(qty.mean(), 2),
                q_p75=int(qty.quantile(0.75)),
                q_p95=int(qty.quantile(0.95)),
                q_max=int(qty.max()),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    trades = load_trades()
    prices = load_prices()
    annotated = annotate_aggressor(trades, prices)

    summary = per_trader_summary(annotated)
    print("=== Per-trader summary (all 3 days) ===")
    print(summary.to_string(index=False))
    print()

    qty = trade_size_distribution(annotated)
    print("=== Trade-size distribution ===")
    print(qty.to_string(index=False))
    print()

    print("=== Top counterparty pairings ===")
    print(pairings(annotated).head(20).to_string(index=False))
    print()

    print("=== Product mix per trader ===")
    pm = per_trader_per_product(annotated)
    pivot = pm.pivot(index="trader", columns="product", values="n_total").fillna(0).astype(int)
    # reorder columns: underlying first, then vouchers by strike, then hydrogel
    cols = (
        [c for c in pivot.columns if c == "VELVETFRUIT_EXTRACT"]
        + sorted([c for c in pivot.columns if c.startswith("VEV_")], key=lambda s: int(s.split("_")[1]))
        + [c for c in pivot.columns if c == "HYDROGEL_PACK"]
    )
    print(pivot[cols].to_string())
    print()

    # per-day breakdown of aggression to test stability
    print("=== Aggression rate per day (trader → day) ===")
    rows = []
    for m in sorted(set(annotated["buyer"]).union(annotated["seller"])):
        for d in DAYS:
            sub = annotated[((annotated["buyer"] == m) | (annotated["seller"] == m)) & (annotated["day"] == d)]
            if len(sub) == 0:
                continue
            n = len(sub)
            agg = ((sub["buyer"] == m) & (sub["aggressor"] == "buyer")).sum() + (
                (sub["seller"] == m) & (sub["aggressor"] == "seller")
            ).sum()
            pas = ((sub["buyer"] == m) & (sub["aggressor"] == "seller")).sum() + (
                (sub["seller"] == m) & (sub["aggressor"] == "buyer")
            ).sum()
            rows.append(dict(trader=m, day=d, n=n, agg_rate=round(agg / n, 3), pas_rate=round(pas / n, 3)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
