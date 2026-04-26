"""Parse tradeHistory from a Prosperity submission log into per-product
fill diagnostics: count, signed volume, effective edge vs touch.

Run:  python3 analysis/parse_log_fills.py logs/<latest>.log
"""
import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with open(path) as f:
        obj = json.load(f)
    th = obj.get("tradeHistory")
    if isinstance(th, str):
        th = json.loads(th)
    trades = pd.DataFrame(th)
    activities = pd.read_csv(StringIO(obj["activitiesLog"]), sep=";")
    return trades, activities


def main(path):
    trades, acts = load(path)
    print(f"trades: {len(trades)} rows | columns: {list(trades.columns)}")
    if not len(trades):
        print("no trades found"); return

    # Trades from the bot's perspective: SUBMISSION as buyer or seller marks
    # us as the side. Prosperity uses "SUBMISSION" sentinel.
    BOT = "SUBMISSION"
    trades["bot_side"] = trades.apply(
        lambda r: "buy" if r.get("buyer") == BOT else ("sell" if r.get("seller") == BOT else None),
        axis=1
    )
    bot = trades[trades["bot_side"].notna()].copy()
    print(f"bot fills: {len(bot)} (of {len(trades)} total)")

    # Fill summary per product
    print("\nPer-product fill summary:")
    g = bot.groupby("symbol").agg(
        n_buys=("bot_side", lambda s: (s == "buy").sum()),
        n_sells=("bot_side", lambda s: (s == "sell").sum()),
        vol_buys=("quantity", lambda q: q[bot.loc[q.index, "bot_side"] == "buy"].sum()),
        vol_sells=("quantity", lambda q: q[bot.loc[q.index, "bot_side"] == "sell"].sum()),
        avg_px=("price", "mean"),
    )
    g["net_pos"] = g["vol_buys"] - g["vol_sells"]
    print(g.to_string())

    # Effective edge per fill: fill_px - bbo_mid_at_that_ts
    # We need to look up the touch mid at each (day, ts) for the symbol.
    # bbo_mid = (bid_price_1 + ask_price_1) / 2
    acts["bbo_mid"] = (acts["bid_price_1"] + acts["ask_price_1"]) / 2
    mids = acts.set_index(["day", "timestamp", "product"])["bbo_mid"]

    if "day" not in bot.columns:
        # tradeHistory may only have timestamp; we need to associate it with day.
        # Strategy: ts resets to 0 each day. We can derive day from ts ordering.
        bot = bot.sort_values("timestamp").reset_index(drop=True)
        day = 0
        last = -1
        days = []
        for ts in bot["timestamp"]:
            if ts < last:
                day += 1
            days.append(day)
            last = ts
        bot["day"] = days

    def get_mid(row):
        try:
            return mids.loc[(row["day"], row["timestamp"], row["symbol"])]
        except KeyError:
            return None

    bot["bbo_mid_at_fill"] = bot.apply(get_mid, axis=1)
    bot["edge"] = bot.apply(
        lambda r: (r["price"] - r["bbo_mid_at_fill"])
                  if r["bot_side"] == "sell"
                  else (r["bbo_mid_at_fill"] - r["price"]),
        axis=1
    )

    print("\nEdge stats per product (positive = captured, negative = adverse):")
    e = bot.groupby("symbol")["edge"].describe()[["count", "mean", "std", "min", "50%", "max"]]
    print(e.to_string())

    # Also: PnL per product (final cumulative)
    last_pnl = (acts.sort_values(["day", "timestamp"])
                  .groupby(["day", "product"]).tail(1)
                  .pivot(index="day", columns="product", values="profit_and_loss"))
    print("\nFinal cumulative PnL (last day):")
    print(last_pnl.iloc[-1].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "logs/2026-04-25_16-59-47.log"
    main(p)
