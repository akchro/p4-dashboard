"""Test asymmetric quote-removal: when |signal| > τ, drop the wrong-side
quote entirely. Compares to baseline pure MM.

This is variant of user's strategy #2 (inventory-anticipating MM), more
aggressive than just shifting quote prices.
"""
from __future__ import annotations
import sys, importlib, json, types
from typing import Dict, List, Optional, Tuple
import pandas as pd

sys.path.insert(0, ".")

# Stub datamodel
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}
class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol; self.price = int(price); self.quantity = int(quantity)
class TradingState:
    def __init__(self, timestamp, listings, order_depths, own_trades, market_trades, position, observations, traderData):
        self.timestamp = timestamp; self.listings = listings; self.order_depths = order_depths
        self.own_trades = own_trades; self.market_trades = market_trades; self.position = position
        self.observations = observations; self.traderData = traderData
_dm = types.ModuleType("datamodel")
_dm.OrderDepth = OrderDepth; _dm.Order = Order; _dm.TradingState = TradingState
sys.modules["datamodel"] = _dm

# Inline trader class (parametrizable)
class AsymTrader:
    CIRCLE = "MICROCHIP_CIRCLE"
    FOLLOWERS = {"MICROCHIP_OVAL": 50, "MICROCHIP_SQUARE": 100,
                 "MICROCHIP_RECTANGLE": 150, "MICROCHIP_TRIANGLE": 200}
    BETAS = {"MICROCHIP_OVAL": 0.0676, "MICROCHIP_SQUARE": 0.1372,
             "MICROCHIP_RECTANGLE": 0.0866, "MICROCHIP_TRIANGLE": 0.0840}
    POS_LIMIT = 10
    TS_PER_TICK = 100

    def __init__(self, edge=4, inv=1.0, size=5, remove_thresh=99.0):
        self.QUOTE_EDGE = edge
        self.INV_SKEW = inv
        self.QUOTE_SIZE = size
        self.REMOVE_THRESHOLD = remove_thresh
        self.SIGNAL_TS_WINDOW = 30000

    @staticmethod
    def _mid(d):
        if not d.buy_orders or not d.sell_orders: return None
        return (max(d.buy_orders) + min(d.sell_orders)) / 2.0
    @staticmethod
    def _best(d):
        return ((max(d.buy_orders) if d.buy_orders else None),
                (min(d.sell_orders) if d.sell_orders else None))

    def run(self, state):
        result = {}
        td = json.loads(state.traderData) if state.traderData else {}
        history = td.get("circle_hist", [])
        cd = state.order_depths.get(self.CIRCLE)
        cm = self._mid(cd) if cd is not None else None
        if cm is not None:
            history.append([int(state.timestamp), float(cm)])
        cutoff = state.timestamp - self.SIGNAL_TS_WINDOW
        history = [h for h in history if h[0] >= cutoff]
        td["circle_hist"] = history
        ts_to_mid = {h[0]: h[1] for h in history}
        def lcr(L):
            t1 = state.timestamp - L * self.TS_PER_TICK; t0 = t1 - self.TS_PER_TICK
            m1 = ts_to_mid.get(t1); m0 = ts_to_mid.get(t0)
            return None if (m1 is None or m0 is None) else m1 - m0

        for product in [self.CIRCLE] + list(self.FOLLOWERS.keys()):
            depth = state.order_depths.get(product)
            if depth is None: continue
            position = state.position.get(product, 0)
            mid_p = self._mid(depth)
            if mid_p is None: continue
            best_bid, best_ask = self._best(depth)
            if best_bid is None or best_ask is None: continue

            signal = 0.0
            if product in self.FOLLOWERS:
                ret_C = lcr(self.FOLLOWERS[product])
                if ret_C is not None:
                    signal = self.BETAS[product] * ret_C

            inv_term = -self.INV_SKEW * (position / self.POS_LIMIT)
            anchor = mid_p + inv_term
            target_bid = anchor - self.QUOTE_EDGE
            target_ask = anchor + self.QUOTE_EDGE
            quote_bid = min(int(round(target_bid)), best_ask - 1)
            quote_ask = max(int(round(target_ask)), best_bid + 1)
            buy_budget = self.POS_LIMIT - position
            sell_budget = self.POS_LIMIT + position

            post_bid = True; post_ask = True
            if signal > self.REMOVE_THRESHOLD:
                post_ask = False
            elif signal < -self.REMOVE_THRESHOLD:
                post_bid = False

            orders = []
            if post_bid and buy_budget > 0:
                orders.append(Order(product, quote_bid, min(self.QUOTE_SIZE, buy_budget)))
            if post_ask and sell_budget > 0:
                orders.append(Order(product, quote_ask, -min(self.QUOTE_SIZE, sell_budget)))
            if orders:
                result[product] = orders

        td["last_ts"] = int(state.timestamp)
        return result, 0, json.dumps(td)


def build_depth(row):
    d = OrderDepth()
    for i in (1, 2, 3):
        bp, bv = row.get(f"bid_price_{i}"), row.get(f"bid_volume_{i}")
        ap, av = row.get(f"ask_price_{i}"), row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv): d.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av): d.sell_orders[int(ap)] = -int(av)
    return d

PRODUCTS = ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
            "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"]


def replay_day(prices_df, trades_df, day, trader):
    df = prices_df[prices_df["day"] == day].sort_values(["timestamp", "product"])
    timestamps = sorted(df["timestamp"].unique())
    position = {p: 0 for p in PRODUCTS}
    cash = {p: 0.0 for p in PRODUCTS}
    n_trades = 0
    traderData = ""
    rows_by_ts = {ts: g for ts, g in df.groupby("timestamp")}
    tdf = trades_df[(trades_df["day"] == day) & (trades_df["symbol"].isin(PRODUCTS))]
    from collections import defaultdict
    trades_by_ts = defaultdict(list)
    for _, tr in tdf.iterrows():
        trades_by_ts[int(tr["timestamp"])].append((tr["symbol"], float(tr["price"]), int(tr["quantity"])))

    for ts in timestamps:
        rows = rows_by_ts[ts]
        depths = {p: OrderDepth() for p in PRODUCTS}
        for _, r in rows.iterrows():
            if r["product"] in PRODUCTS:
                depths[r["product"]] = build_depth(r)
        state = TradingState(ts, {}, depths, {}, {}, dict(position), None, traderData)
        result, _, traderData = trader.run(state)
        ts_trades = trades_by_ts.get(ts, [])
        for product, orders in result.items():
            depth = depths[product]
            for o in orders:
                if o.quantity > 0:
                    qleft = o.quantity
                    for ask in sorted(depth.sell_orders):
                        if ask > o.price or qleft <= 0: break
                        avail = -depth.sell_orders[ask]
                        fill = min(qleft, avail)
                        if fill > 0:
                            position[product] += fill; cash[product] -= fill * ask
                            qleft -= fill; n_trades += 1
                    if qleft > 0:
                        for sym, p_tr, q_tr in ts_trades:
                            if sym != product: continue
                            if p_tr <= o.price and qleft > 0:
                                fill = min(qleft, q_tr)
                                if fill > 0:
                                    position[product] += fill; cash[product] -= fill * o.price
                                    qleft -= fill; n_trades += 1
                else:
                    qleft = -o.quantity
                    for bid in sorted(depth.buy_orders, reverse=True):
                        if bid < o.price or qleft <= 0: break
                        avail = depth.buy_orders[bid]
                        fill = min(qleft, avail)
                        if fill > 0:
                            position[product] -= fill; cash[product] += fill * bid
                            qleft -= fill; n_trades += 1
                    if qleft > 0:
                        for sym, p_tr, q_tr in ts_trades:
                            if sym != product: continue
                            if p_tr >= o.price and qleft > 0:
                                fill = min(qleft, q_tr)
                                if fill > 0:
                                    position[product] -= fill; cash[product] += fill * o.price
                                    qleft -= fill; n_trades += 1

    last_rows = rows_by_ts[timestamps[-1]]
    last_mids = {}
    for _, r in last_rows.iterrows():
        if r["product"] in PRODUCTS:
            last_mids[r["product"]] = r["mid_price"]
    pnl = sum(cash[p] + position[p] * last_mids.get(p, 0) for p in PRODUCTS)
    return pnl, n_trades


def main():
    pdfs = [pd.read_csv(f"historical/ROUND_5/prices_round_5_day_{d}.csv", sep=";").assign(day=d) for d in (2,3,4)]
    tdfs = [pd.read_csv(f"historical/ROUND_5/trades_round_5_day_{d}.csv", sep=";").assign(day=d) for d in (2,3,4)]
    prices = pd.concat(pdfs); trades = pd.concat(tdfs)

    print(f"{'config':30s} {'d2':>9s} {'d3':>9s} {'d4':>9s} {'total':>9s} {'trades':>7s}")
    for thr in [99.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.0]:
        trader = AsymTrader(edge=4, inv=1.0, size=5, remove_thresh=thr)
        pnls = []; tr = 0
        for d in (2,3,4):
            pnl, ntr = replay_day(prices, trades, d, AsymTrader(edge=4, inv=1.0, size=5, remove_thresh=thr))
            pnls.append(pnl); tr += ntr
        label = "no-remove (baseline)" if thr == 99.0 else f"thr={thr}"
        print(f"{label:30s} {pnls[0]:>9.0f} {pnls[1]:>9.0f} {pnls[2]:>9.0f} {sum(pnls):>9.0f} {tr:>7d}")


if __name__ == "__main__":
    main()
