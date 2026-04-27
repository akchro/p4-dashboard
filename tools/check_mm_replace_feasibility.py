"""Check feasibility of replacing Mark 14 as the HP market maker.

Key questions:
1. Is Mark 14 the only quoter, or are there other levels?
2. How big is each Mark 38 order vs the displayed book?
3. What's the rate / clip size — can we actually keep up given the 200 position limit?
4. Does Mark 14's quote sit at the BBO consistently, or does it walk?
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
frames_p, frames_t = [], []
for d in [1, 2, 3]:
    p = pd.read_csv(ROOT / f"historical/ROUND_4/prices_round_4_day_{d}.csv", sep=";")
    t = pd.read_csv(ROOT / f"historical/ROUND_4/trades_round_4_day_{d}.csv", sep=";")
    p["day"] = d; t["day"] = d
    frames_p.append(p); frames_t.append(t)
prices = pd.concat(frames_p, ignore_index=True)
trades = pd.concat(frames_t, ignore_index=True)

hp_p = prices[prices["product"] == "HYDROGEL_PACK"].copy()
hp_t = trades[trades["symbol"] == "HYDROGEL_PACK"].copy()

# 1. Book depth — how many levels are typically populated?
print("=== Book depth on HP ===")
for side in ["bid", "ask"]:
    for lvl in [1, 2, 3]:
        col = f"{side}_price_{lvl}"
        n_filled = hp_p[col].notna().sum()
        print(f"  {col}: {n_filled} / {len(hp_p)}  ({100*n_filled/len(hp_p):.1f}%)")

# 2. Book size at L1 — what does Mark 14 typically post?
print("\n=== L1 sizes on HP ===")
print("bid_volume_1:", hp_p["bid_volume_1"].describe().to_dict())
print("ask_volume_1:", hp_p["ask_volume_1"].describe().to_dict())

# 3. Trade clip size vs L1 size
m = hp_t.merge(hp_p[["day","timestamp","bid_price_1","bid_volume_1","ask_price_1","ask_volume_1"]],
               on=["day","timestamp"])
m38b = m[(m["buyer"]=="Mark 38")&(m["seller"]=="Mark 14")]
m14b = m[(m["buyer"]=="Mark 14")&(m["seller"]=="Mark 38")]

print("\n=== Mark 38 buy clip vs displayed L1 ask volume ===")
print(f"  trade qty:   {m38b['quantity'].describe().to_dict()}")
print(f"  ask_vol_1:   {m38b['ask_volume_1'].describe().to_dict()}")
print(f"  trades where qty == ask_vol_1: {(m38b['quantity']==m38b['ask_volume_1']).sum()} / {len(m38b)}")
print(f"  trades where qty <  ask_vol_1: {(m38b['quantity']<m38b['ask_volume_1']).sum()}")
print(f"  trades where qty >  ask_vol_1: {(m38b['quantity']>m38b['ask_volume_1']).sum()}")

print("\n=== Mark 14 buy clip vs displayed L1 bid volume ===")
print(f"  trade qty:   {m14b['quantity'].describe().to_dict()}")
print(f"  bid_vol_1:   {m14b['bid_volume_1'].describe().to_dict()}")
print(f"  trades where qty == bid_vol_1: {(m14b['quantity']==m14b['bid_volume_1']).sum()} / {len(m14b)}")

# 4. Trade rate / spacing
print("\n=== Trade arrival on HP (per day) ===")
for d in [1,2,3]:
    day_t = m[m["day"]==d]
    n = len(day_t)
    duration = 1_000_000  # ticks/day
    print(f"  day {d}: {n} trades, mean gap = {duration/max(n,1):.0f} ticks, vol = {day_t['quantity'].sum()}")

# 5. Position-limit reality check — can we cycle 4000 units/day with limit 200?
print("\n=== Inventory cycle implied if MM ===")
total_buy_vol = m38b["quantity"].sum()  # we'd be selling this much
total_sell_vol = m14b["quantity"].sum()  # we'd be buying this much
print(f"  We'd sell to Mark 38: {total_buy_vol} units over 3 days")
print(f"  We'd buy from Mark 38: {total_sell_vol} units over 3 days")
print(f"  Net imbalance:        {total_buy_vol - total_sell_vol} (Mark 38's buys - sells)")

# 6. Is the book usually wider than 1 (i.e. is there room to undercut?)
print("\n=== Spread distribution ===")
hp_p_clean = hp_p.dropna(subset=["bid_price_1","ask_price_1"]).copy()
hp_p_clean["spread"] = hp_p_clean["ask_price_1"] - hp_p_clean["bid_price_1"]
print(hp_p_clean["spread"].value_counts().sort_index())

# 7. Check time-bucket inventory: if I had to perfectly intermediate, how big does my book swing?
m = m.sort_values(["day","timestamp"])
print("\n=== Running inventory if I were the MM (signed by 'sells to Mark 38') ===")
for d in [1,2,3]:
    dd = m[m["day"]==d].copy()
    # If buyer is Mark 38, I sold => -qty; if seller is Mark 38, I bought => +qty
    dd["my_delta"] = dd.apply(
        lambda r: -r["quantity"] if r["buyer"]=="Mark 38" else (+r["quantity"] if r["seller"]=="Mark 38" else 0),
        axis=1,
    )
    dd["inv"] = dd["my_delta"].cumsum()
    print(f"  day {d}: inv min={dd['inv'].min():+}, max={dd['inv'].max():+}, end={dd['inv'].iloc[-1]:+}")
