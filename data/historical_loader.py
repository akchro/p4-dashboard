import os
import re
import glob as _glob
import pandas as pd

HISTORICAL_DIR = "historical"

NUMERIC_COLS = [
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3", "ask_price_1", "ask_volume_1",
    "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]


def get_rounds():
    """Return sorted list of subfolder names in historical/."""
    if not os.path.isdir(HISTORICAL_DIR):
        return []
    return sorted(
        d for d in os.listdir(HISTORICAL_DIR)
        if os.path.isdir(os.path.join(HISTORICAL_DIR, d))
    )


def get_available_days(round_name: str):
    """Parse day values from price filenames in a round folder."""
    folder = os.path.join(HISTORICAL_DIR, round_name)
    files = _glob.glob(os.path.join(folder, "prices_round_*_day_*.csv"))
    days = set()
    for f in files:
        m = re.search(r"day_([-\d]+)\.csv$", f)
        if m:
            days.add(int(m.group(1)))
    return sorted(days)


def load_round(round_name: str) -> dict:
    """Load all price and trade CSVs for a round into combined DataFrames."""
    folder = os.path.join(HISTORICAL_DIR, round_name)

    price_files = sorted(_glob.glob(os.path.join(folder, "prices_round_*_day_*.csv")))
    trade_files = sorted(_glob.glob(os.path.join(folder, "trades_round_*_day_*.csv")))

    # Load and concat prices
    price_dfs = []
    for f in price_files:
        df = pd.read_csv(f, sep=";")
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        price_dfs.append(df)
    activities = pd.concat(price_dfs, ignore_index=True) if price_dfs else pd.DataFrame()

    # Guard against mid_price=0 when the order book is empty — replace with NaN
    # so charts show a gap instead of spiking to zero.
    if not activities.empty and "mid_price" in activities.columns:
        activities.loc[activities["mid_price"] == 0, "mid_price"] = pd.NA

    # Compute wallmid: average of bid wall (lowest bid) and ask wall (highest ask).
    # These "walls" are the deepest visible levels where market makers quote.
    if not activities.empty:
        bid_cols = [c for c in ["bid_price_1", "bid_price_2", "bid_price_3"] if c in activities.columns]
        ask_cols = [c for c in ["ask_price_1", "ask_price_2", "ask_price_3"] if c in activities.columns]
        if bid_cols and ask_cols:
            bid_wall = activities[bid_cols].min(axis=1)
            ask_wall = activities[ask_cols].max(axis=1)
            activities["wallmid"] = (bid_wall + ask_wall) / 2

    # Load and concat trades, extracting day from filename
    trade_dfs = []
    for f in trade_files:
        df = pd.read_csv(f, sep=";")
        m = re.search(r"day_([-\d]+)\.csv$", f)
        if m:
            df["day"] = int(m.group(1))
        trade_dfs.append(df)
    trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame(
        columns=["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity", "day"]
    )

    return {"activities": activities, "trades": trades}
