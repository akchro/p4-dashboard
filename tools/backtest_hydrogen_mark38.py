"""Approximate backtest of hydrogen_mark38.Trader against round 4 historical data.

Approximations (since we don't have a full Prosperity engine):
  * Each Mark 38 trade in the historical log is treated as an event we capture
    instead of Mark 14 — at OUR quoted price (best_bid+1 or best_ask-1 with
    inventory skew applied).
  * We assume our orders fill 100% (Mark 38 clip <= 6 << our quote_size 30).
  * We honor position-limit suppression and the MIN_SPREAD gate.
  * P&L = realized cash + remaining inventory marked at the next book mid.
"""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Strategy params (must match hydrogen_mark38.py)
QUOTE_SIZE = 30
MIN_SPREAD = 4
SKEW_TICKS_PER_50 = 1
HARD_LIMIT = 170
POS_LIMIT = 200


def step_quotes(best_bid, best_ask, position):
    """Mirror of hydrogen_mark38.Trader's quote logic."""
    spread = best_ask - best_bid
    if spread < MIN_SPREAD:
        return None, None

    skew = int(round((position / 50.0) * SKEW_TICKS_PER_50))
    target_bid = best_bid + 1 - skew
    target_ask = best_ask - 1 - skew

    quote_bid = best_bid < target_bid < best_ask and position < HARD_LIMIT
    quote_ask = best_bid < target_ask < best_ask and position > -HARD_LIMIT

    if quote_bid and quote_ask and target_bid >= target_ask:
        if position > 0:
            quote_bid = False
            target_ask = best_ask - 1
        else:
            quote_ask = False
            target_bid = best_bid + 1

    return (target_bid if quote_bid else None,
            target_ask if quote_ask else None)


def backtest_day(day):
    p = pd.read_csv(ROOT / f"historical/ROUND_4/prices_round_4_day_{day}.csv", sep=";")
    t = pd.read_csv(ROOT / f"historical/ROUND_4/trades_round_4_day_{day}.csv", sep=";")
    hp_p = p[p["product"] == "HYDROGEL_PACK"][["timestamp", "bid_price_1", "ask_price_1", "mid_price"]]
    hp_t = t[(t["symbol"] == "HYDROGEL_PACK") &
             (((t["buyer"] == "Mark 38") & (t["seller"] == "Mark 14")) |
              ((t["buyer"] == "Mark 14") & (t["seller"] == "Mark 38")))]

    book_by_ts = hp_p.set_index("timestamp")[["bid_price_1", "ask_price_1"]].to_dict("index")

    position = 0
    cash = 0.0
    fills = 0
    skipped = 0
    fill_log = []

    for _, row in hp_t.sort_values("timestamp").iterrows():
        ts = row["timestamp"]
        book = book_by_ts.get(ts)
        if book is None:
            continue
        bb, ba = book["bid_price_1"], book["ask_price_1"]
        my_bid, my_ask = step_quotes(bb, ba, position)

        is_38_buy = row["buyer"] == "Mark 38"  # Mark 38 buys → he hits asks → we'd sell
        qty = int(row["quantity"])

        if is_38_buy:
            if my_ask is not None and (POS_LIMIT + position) >= qty:
                cash += my_ask * qty
                position -= qty
                fills += 1
                fill_log.append((ts, "S", my_ask, qty, position))
            else:
                skipped += 1
        else:
            if my_bid is not None and (POS_LIMIT - position) >= qty:
                cash -= my_bid * qty
                position += qty
                fills += 1
                fill_log.append((ts, "B", my_bid, qty, position))
            else:
                skipped += 1

    # Mark remaining inventory at last available mid
    last_ts = hp_p["timestamp"].max()
    last_mid = float(hp_p.set_index("timestamp").loc[last_ts, "ask_price_1"] +
                     hp_p.set_index("timestamp").loc[last_ts, "bid_price_1"]) / 2
    mtm = cash + position * last_mid

    inv_series = pd.Series([f[4] for f in fill_log])
    print(f"\n=== Day {day} ===")
    print(f"  fills: {fills} / {len(hp_t)} possible (skipped: {skipped})")
    print(f"  end position:    {position:+}")
    print(f"  realized cash:   ${cash:+,.0f}")
    print(f"  mtm at last mid: ${mtm:+,.0f}")
    print(f"  inventory range during day: [{inv_series.min():+}, {inv_series.max():+}]")

    return mtm, position, cash


totals = [backtest_day(d) for d in [1, 2, 3]]
print("\n=== TOTAL ===")
print(f"  total mtm P&L: ${sum(t[0] for t in totals):+,.0f}")
print(f"  total cash:    ${sum(t[2] for t in totals):+,.0f}")
