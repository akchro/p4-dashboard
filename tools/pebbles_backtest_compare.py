"""Compare pebbles trader variants on the same Round 5 historical data.

Usage:
    python3 tools/pebbles_backtest_compare.py            # baseline vs aggressive
    python3 tools/pebbles_backtest_compare.py sweep      # parameter sweep on aggressive

Imports both `pebbles_trader` and `pebbles_trader_aggressive`, replays
historical/ROUND_5/{day_2, day_3, day_4} via the same fill-simulator that
tools/pebbles_backtest.py uses, and prints day-by-day PnL + total.

Replay model caveats (same as tools/pebbles_backtest.py):
  - Maker fill heuristic is generous (assumes 100% queue priority at our
    price level when a market trade prints there).
  - Player-vs-bot only.
  - Orders expire each tick.
"""
from __future__ import annotations

import importlib
import sys
import types as _types
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, ".")


# --- Stub datamodel for trader modules ---------------------------------------
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
    def __init__(self, timestamp, listings, order_depths, own_trades, market_trades,
                 position, observations, traderData):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData


_dm = _types.ModuleType("datamodel")
_dm.OrderDepth = OrderDepth
_dm.Order = Order
_dm.TradingState = TradingState
sys.modules["datamodel"] = _dm


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
            d.sell_orders[int(ap)] = -int(av)
    return d


def replay_day(trader_module, prices_df: pd.DataFrame, trades_df: pd.DataFrame, day: int):
    PRODUCTS = trader_module.PRODUCTS
    df = prices_df[prices_df["day"] == day].copy()
    df = df.sort_values(["timestamp", "product"])
    timestamps = sorted(df["timestamp"].unique())

    trader = trader_module.Trader()
    position: Dict[str, int] = {p: 0 for p in PRODUCTS}
    cash: Dict[str, float] = {p: 0.0 for p in PRODUCTS}
    n_trades = 0
    traderData = ""

    rows_by_ts = {ts: g for ts, g in df.groupby("timestamp")}

    tdf = trades_df[(trades_df["day"] == day) & (trades_df["symbol"].isin(PRODUCTS))]
    trades_by_ts: Dict[int, List] = defaultdict(list)
    for _, tr in tdf.iterrows():
        trades_by_ts[int(tr["timestamp"])].append(
            (tr["symbol"], float(tr["price"]), int(tr["quantity"])))

    max_abs_pos = 0

    for ts in timestamps:
        rows = rows_by_ts[ts]
        depths = {p: OrderDepth() for p in PRODUCTS}
        for _, r in rows.iterrows():
            p = r["product"]
            if p in PRODUCTS:
                depths[p] = build_depth(r)

        state = TradingState(
            timestamp=ts, listings={}, order_depths=depths,
            own_trades={}, market_trades={},
            position=dict(position), observations=None, traderData=traderData,
        )
        result, _, traderData = trader.run(state)
        ts_trades = trades_by_ts.get(ts, [])

        for product, orders in result.items():
            depth = depths[product]
            for o in orders:
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
        for v in position.values():
            if abs(v) > max_abs_pos:
                max_abs_pos = abs(v)

    last_rows = rows_by_ts[timestamps[-1]]
    last_mids: Dict[str, float] = {}
    for _, r in last_rows.iterrows():
        if r["product"] in PRODUCTS:
            last_mids[r["product"]] = r["mid_price"]

    pnl_by_product = {}
    for p in PRODUCTS:
        mid = last_mids.get(p, 0)
        pnl_by_product[p] = cash[p] + position[p] * mid

    return pnl_by_product, sum(pnl_by_product.values()), n_trades, dict(position), max_abs_pos


def run_module(name: str, prices, trades, label: str = ""):
    mod = importlib.import_module(name)
    importlib.reload(mod)  # ensure latest changes pick up
    label = label or name
    print(f"\n=== {label} ===")
    print(f"{'day':>4s} {'pnl':>10s} {'trades':>7s} {'maxpos':>7s}  per-product…")
    total_pnl = 0
    total_trades = 0
    overall_max = 0
    for d in (2, 3, 4):
        pnl_dict, pnl, ntr, fpos, mpos = replay_day(mod, prices, trades, d)
        total_pnl += pnl
        total_trades += ntr
        overall_max = max(overall_max, mpos)
        per = "  ".join(f"{p[8:]}={pnl_dict[p]:+8.0f}" for p in mod.PRODUCTS)
        print(f"{d:>4d} {pnl:>10.0f} {ntr:>7d} {mpos:>7d}  {per}")
        print(f"        final pos: {fpos}")
    print(f"TOTAL  pnl={total_pnl:.0f}  trades={total_trades}  max|pos|={overall_max}")
    return total_pnl, total_trades, overall_max


def sweep(prices, trades):
    """Parameter sweep over aggressive variants. Edits module globals between
    runs to test combinations without writing many files."""
    import pebbles_trader_aggressive as agg
    importlib.reload(agg)

    base_ladder = [(1, 3), (3, 3), (6, 2), (10, 1), (13, 1)]

    # Round-12: micro-tune around bt5
    bt5 = [(4, 1), (5, 3), (6, 3), (7, 1), (8, 5)]   # round-11 winner = 105794
    configs10 = [
        ("bt5_inv1.6",        bt5,                                  1.6, 0.0,  8, 5, 5),
        # finer INV grain
        ("bt5_inv1.55",       bt5,                                  1.55,0.0,  8, 5, 5),
        ("bt5_inv1.65",       bt5,                                  1.65,0.0,  8, 5, 5),
        # alt over-budget patterns
        ("alt_bt_a",          [(4,1),(5,2),(6,3),(7,2),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("alt_bt_b",          [(4,2),(5,2),(6,3),(7,1),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("alt_bt_c",          [(4,1),(5,3),(6,2),(7,2),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("alt_bt_d",          [(4,1),(5,3),(6,3),(7,2),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("alt_bt_e",          [(4,1),(5,2),(6,2),(7,2),(8,5)],      1.6, 0.0,  8, 5, 5),
        # over-budget on multiple levels
        ("ob_5_8",            [(4,1),(5,4),(6,3),(7,1),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("ob_6_8",            [(4,1),(5,3),(6,4),(7,1),(8,5)],      1.6, 0.0,  8, 5, 5),
        ("ob_4_8",            [(4,2),(5,3),(6,3),(7,1),(8,5)],      1.6, 0.0,  8, 5, 5),
        # extreme tail edge 9, 10
        ("xt_9",              [(4,1),(5,3),(6,3),(7,1),(8,3),(9,3)],1.6, 0.0,  8, 5, 5),
        ("xt_10",             [(4,1),(5,3),(6,3),(7,1),(8,3),(10,3)],1.6, 0.0,  8, 5, 5),
        # combine bt5 with dev skew
        ("bt5_dev-0.1",       bt5,                                  1.6, -0.1, 8, 5, 5),
        ("bt5_dev-0.2",       bt5,                                  1.6, -0.2, 8, 5, 5),
        # confirm taker still helping at this level
        ("bt5_no_take",       bt5,                                  1.6, 0.0, 99, 5, 5),
        ("bt5_take7",         bt5,                                  1.6, 0.0,  7, 5, 5),
        ("bt5_take_edge4",    bt5,                                  1.6, 0.0,  8, 4, 5),
        # huge tail
        ("bt10_inv1.6",       [(4,1),(5,3),(6,3),(7,1),(8,10)],     1.6, 0.0,  8, 5, 5),
        # narrower band
        ("nar_5678",          [(5,2),(6,3),(7,2),(8,5)],            1.6, 0.0,  8, 5, 5),
        ("nar_5678_2",        [(5,3),(6,2),(7,2),(8,5)],            1.6, 0.0,  8, 5, 5),
        # Add edge 3 to see if back-pulls in noise
        ("with3",             [(3,1),(4,1),(5,3),(6,3),(7,1),(8,5)],1.6, 0.0,  8, 5, 5),
    ]
    configs = [c for c in configs10 if all(s > 0 for _, s in c[1])]

    print(f"{'config':>22s} {'pnl':>9s} {'trades':>7s} {'max|p|':>6s}")
    print("-" * 50)
    rows = []
    for label, ladder, inv, dev, td_, te, tm in configs:
        # patch module globals
        agg.LADDER = ladder
        agg.INV_SKEW = inv
        agg.DEV_SKEW = dev
        agg.TAKE_DEV = td_
        agg.TAKE_EDGE = te
        agg.TAKE_MAX_QTY = tm
        total = 0
        ntr_total = 0
        max_pos = 0
        for d in (2, 3, 4):
            pnl_dict, pnl, ntr, _, mpos = replay_day(agg, prices, trades, d)
            total += pnl
            ntr_total += ntr
            max_pos = max(max_pos, mpos)
        print(f"{label:>22s} {total:>9.0f} {ntr_total:>7d} {max_pos:>6d}")
        rows.append((label, total, ntr_total, max_pos))
    print("\nbest by pnl:")
    for label, pnl, ntr, mpos in sorted(rows, key=lambda r: -r[1])[:5]:
        print(f"  {label:>22s}  pnl={pnl:.0f}  trades={ntr}  max|p|={mpos}")


def main():
    print("loading historical/ROUND_5 data…")
    pdfs, tdfs = [], []
    for d in (2, 3, 4):
        pdf = pd.read_csv(f"historical/ROUND_5/prices_round_5_day_{d}.csv", sep=";")
        pdf["day"] = d; pdfs.append(pdf)
        tdf = pd.read_csv(f"historical/ROUND_5/trades_round_5_day_{d}.csv", sep=";")
        tdf["day"] = d; tdfs.append(tdf)
    prices = pd.concat(pdfs, ignore_index=True)
    trades = pd.concat(tdfs, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep(prices, trades)
    else:
        run_module("pebbles_trader",            prices, trades, "BASELINE")
        run_module("pebbles_trader_aggressive", prices, trades, "AGGRESSIVE")


if __name__ == "__main__":
    main()
