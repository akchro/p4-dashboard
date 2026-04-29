"""Local backtest for microchip_trader.Trader against historical Round 5 CSVs.

Stubs out datamodel (OrderDepth/Order/TradingState) so we don't need the IMC
SDK installed. Replays each tick of historical data, lets the trader place
orders, and simulates fills two ways:

1. Taker fills: our orders that cross the existing depth match instantly.
2. Maker fills: our resting orders at prices that the bot trade flow visits
   during the same tick get filled (heuristic — assumes 100% market share at
   our price level, which over-estimates real fills).

This is NOT a perfect emulator of the IMC engine. Caveats:
- Maker fill model is generous (real engine may fill us partially or not at
  all if other resting orders are ahead of us in queue).
- Player-vs-bot only (no team-vs-team).
- Orders expire at end of tick.

Use the output as a sanity check on direction and order of magnitude, not as
a precise estimate of live PnL.
"""
from __future__ import annotations
import sys
import importlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

sys.path.insert(0, ".")

# --- Stub datamodel ---------------------------------------------------------
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = int(price)
        self.quantity = int(quantity)
    def __repr__(self):
        return f"Order({self.symbol}, {self.price}, {self.quantity})"

class TradingState:
    def __init__(self, timestamp, listings, order_depths, own_trades, market_trades, position, observations, traderData):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData

# Inject stubs so the trader file's `from datamodel import …` works
import types as _types
_dm = _types.ModuleType("datamodel")
_dm.OrderDepth = OrderDepth
_dm.Order = Order
_dm.TradingState = TradingState
sys.modules["datamodel"] = _dm

import microchip_trader as ptrader

PRODUCTS = [ptrader.CIRCLE] + list(ptrader.FOLLOWERS.keys())
POS_LIMIT = ptrader.POS_LIMIT


def build_depth(row) -> OrderDepth:
    d = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv):
            d.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av):
            d.sell_orders[int(ap)] = -int(av)  # IMC convention: ask volumes are negative
    return d


def replay_day(prices_df: pd.DataFrame, trades_df: pd.DataFrame, day: int):
    """Replay one day. Returns (pnl_per_product, total_pnl, n_trades, traderData, final_pos)"""
    df = prices_df[prices_df["day"] == day].copy()
    df = df.sort_values(["timestamp", "product"])
    timestamps = sorted(df["timestamp"].unique())

    trader = ptrader.Trader()
    position: Dict[str, int] = {p: 0 for p in PRODUCTS}
    cash: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
    n_trades = 0
    traderData = ""

    rows_by_ts = {ts: g for ts, g in df.groupby("timestamp")}

    # Index trades by (timestamp, symbol) for fast lookup
    tdf = trades_df[(trades_df["day"] == day) & (trades_df["symbol"].isin(PRODUCTS))]
    trades_by_ts: Dict[int, List] = defaultdict(list)
    for _, tr in tdf.iterrows():
        trades_by_ts[int(tr["timestamp"])].append((tr["symbol"], float(tr["price"]), int(tr["quantity"])))

    for ts in timestamps:
        rows = rows_by_ts[ts]
        depths = {p: OrderDepth() for p in PRODUCTS}
        for _, r in rows.iterrows():
            p = r["product"]
            if p in PRODUCTS:
                depths[p] = build_depth(r)

        state = TradingState(
            timestamp=ts,
            listings={},
            order_depths=depths,
            own_trades={},
            market_trades={},
            position=dict(position),
            observations=None,
            traderData=traderData,
        )
        result, _, traderData = trader.run(state)

        ts_trades = trades_by_ts.get(ts, [])

        for product, orders in result.items():
            depth = depths[product]
            for o in orders:
                # 1. Taker fill: cross existing depth
                if o.quantity > 0:
                    qty_left = o.quantity
                    for ask in sorted(depth.sell_orders):
                        if ask > o.price or qty_left <= 0:
                            break
                        avail = -depth.sell_orders[ask]
                        fill = min(qty_left, avail)
                        if fill > 0:
                            position[product] += fill
                            cash[product] -= fill * ask
                            qty_left -= fill
                            n_trades += 1
                    # 2. Maker fill: any market trade this tick at price <= our bid
                    #    (someone sold aggressively, hit our resting bid)
                    if qty_left > 0:
                        for sym, p_tr, q_tr in ts_trades:
                            if sym != product:
                                continue
                            if p_tr <= o.price and qty_left > 0:
                                fill = min(qty_left, q_tr)
                                if fill > 0:
                                    position[product] += fill
                                    cash[product] -= fill * o.price
                                    qty_left -= fill
                                    n_trades += 1
                elif o.quantity < 0:
                    qty_left = -o.quantity
                    for bid in sorted(depth.buy_orders, reverse=True):
                        if bid < o.price or qty_left <= 0:
                            break
                        avail = depth.buy_orders[bid]
                        fill = min(qty_left, avail)
                        if fill > 0:
                            position[product] -= fill
                            cash[product] += fill * bid
                            qty_left -= fill
                            n_trades += 1
                    if qty_left > 0:
                        for sym, p_tr, q_tr in ts_trades:
                            if sym != product:
                                continue
                            if p_tr >= o.price and qty_left > 0:
                                fill = min(qty_left, q_tr)
                                if fill > 0:
                                    position[product] -= fill
                                    cash[product] += fill * o.price
                                    qty_left -= fill
                                    n_trades += 1

    last_rows = rows_by_ts[timestamps[-1]]
    last_mids: Dict[str, float] = {}
    for _, r in last_rows.iterrows():
        if r["product"] in PRODUCTS:
            last_mids[r["product"]] = r["mid_price"]

    pnl_by_product = {}
    for p in PRODUCTS:
        mid = last_mids.get(p, 0)
        pnl_by_product[p] = cash[p] + position[p] * mid

    return pnl_by_product, sum(pnl_by_product.values()), n_trades, traderData, dict(position)


def main():
    print("loading historical/ROUND_5 data…")
    pdfs = []
    tdfs = []
    for d in (2, 3, 4):
        pdf = pd.read_csv(f"historical/ROUND_5/prices_round_5_day_{d}.csv", sep=";")
        pdf["day"] = d
        pdfs.append(pdf)
        tdf = pd.read_csv(f"historical/ROUND_5/trades_round_5_day_{d}.csv", sep=";")
        tdf["day"] = d
        tdfs.append(tdf)
    prices = pd.concat(pdfs, ignore_index=True)
    trades = pd.concat(tdfs, ignore_index=True)

    print(f"\n{'day':>4s}  {'pnl':>10s}  {'trades':>7s}  per-product…")
    total_pnl = 0
    total_trades = 0
    for d in (2, 3, 4):
        pnl_dict, pnl, ntrades, td, final_pos = replay_day(prices, trades, d)
        total_pnl += pnl
        total_trades += ntrades
        per = "  ".join(f"{p[8:]}={pnl_dict[p]:+8.1f}" for p in PRODUCTS)
        print(f"{d:>4d}  {pnl:>10.1f}  {ntrades:>7d}  {per}")
        print(f"        final positions: {final_pos}")
    print(f"\nTOTAL across 3 days:  pnl = {total_pnl:.1f}, trades = {total_trades}")


if __name__ == "__main__":
    main()
