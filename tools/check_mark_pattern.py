"""Check the Mark 38 / Mark 14 trade pattern in HYDROGEL_PACK round 4."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAYS = [1, 2, 3]

frames_p, frames_t = [], []
for d in DAYS:
    p = pd.read_csv(ROOT / f"historical/ROUND_4/prices_round_4_day_{d}.csv", sep=";")
    t = pd.read_csv(ROOT / f"historical/ROUND_4/trades_round_4_day_{d}.csv", sep=";")
    p["day"] = d
    t["day"] = d
    frames_p.append(p)
    frames_t.append(t)

prices = pd.concat(frames_p, ignore_index=True)
trades = pd.concat(frames_t, ignore_index=True)

# Filter to HYDROGEL_PACK
hp_prices = prices[prices["product"] == "HYDROGEL_PACK"][
    ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]
].copy()
hp_trades = trades[trades["symbol"] == "HYDROGEL_PACK"].copy()

print(f"Total HP trades: {len(hp_trades)}")
print("Buyer/Seller counts:")
print(hp_trades.groupby(["buyer", "seller"]).size().sort_values(ascending=False))

# Join trades with the order book at the same timestamp
m = hp_trades.merge(hp_prices, on=["day", "timestamp"], how="left", indicator=True)
print(f"\nMerge result: {m['_merge'].value_counts().to_dict()}")

# Classify: Mark 38 buying from Mark 14, vs the reverse
m38_buys = m[(m["buyer"] == "Mark 38") & (m["seller"] == "Mark 14")].copy()
m14_buys = m[(m["buyer"] == "Mark 14") & (m["seller"] == "Mark 38")].copy()

m38_buys["price_vs_ask"] = m38_buys["price"] - m38_buys["ask_price_1"]
m38_buys["price_vs_bid"] = m38_buys["price"] - m38_buys["bid_price_1"]

m14_buys["price_vs_ask"] = m14_buys["price"] - m14_buys["ask_price_1"]
m14_buys["price_vs_bid"] = m14_buys["price"] - m14_buys["bid_price_1"]

print("\n=== Mark 38 BUYS from Mark 14 ===")
print(f"  n = {len(m38_buys)}")
print(f"  trades at ask:                       {(m38_buys['price'] == m38_buys['ask_price_1']).sum()}")
print(f"  trades at bid:                       {(m38_buys['price'] == m38_buys['bid_price_1']).sum()}")
print(f"  trades inside spread:                {((m38_buys['price'] > m38_buys['bid_price_1']) & (m38_buys['price'] < m38_buys['ask_price_1'])).sum()}")
print(f"  trades above ask (printed > book):   {(m38_buys['price'] > m38_buys['ask_price_1']).sum()}")
print(f"  trades below bid:                    {(m38_buys['price'] < m38_buys['bid_price_1']).sum()}")

print("\n=== Mark 14 BUYS from Mark 38 ===")
print(f"  n = {len(m14_buys)}")
print(f"  trades at ask:                       {(m14_buys['price'] == m14_buys['ask_price_1']).sum()}")
print(f"  trades at bid:                       {(m14_buys['price'] == m14_buys['bid_price_1']).sum()}")
print(f"  trades inside spread:                {((m14_buys['price'] > m14_buys['bid_price_1']) & (m14_buys['price'] < m14_buys['ask_price_1'])).sum()}")
print(f"  trades above ask:                    {(m14_buys['price'] > m14_buys['ask_price_1']).sum()}")
print(f"  trades below bid (printed < book):   {(m14_buys['price'] < m14_buys['bid_price_1']).sum()}")

# Print some samples where the price doesn't match the ask/bid expectation
print("\n=== Sample: Mark 38 buys NOT at ask ===")
mismatch_38 = m38_buys[m38_buys["price"] != m38_buys["ask_price_1"]]
print(mismatch_38[["day", "timestamp", "price", "bid_price_1", "ask_price_1", "quantity"]].head(15))

print("\n=== Sample: Mark 14 buys NOT at bid (i.e., price != bid) ===")
mismatch_14 = m14_buys[m14_buys["price"] != m14_buys["bid_price_1"]]
print(mismatch_14[["day", "timestamp", "price", "bid_price_1", "ask_price_1", "quantity"]].head(15))
