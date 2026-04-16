import re
import json
from io import StringIO
import pandas as pd


def load_log(filepath: str) -> dict:
    with open(filepath) as f:
        raw = json.load(f)

    # Parse activities
    activities = pd.read_csv(StringIO(raw["activitiesLog"]), sep=";")
    numeric_cols = [
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
        "bid_price_3", "bid_volume_3", "ask_price_1", "ask_volume_1",
        "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss",
    ]
    for col in numeric_cols:
        if col in activities.columns:
            activities[col] = pd.to_numeric(activities[col], errors="coerce")

    # Guard against mid_price=0 when the order book is empty — replace with NaN
    # so charts show a gap instead of spiking to zero.
    if "mid_price" in activities.columns:
        activities.loc[activities["mid_price"] == 0, "mid_price"] = pd.NA

    # Parse trades
    trade_list = raw.get("tradeHistory", [])
    if trade_list:
        trades = pd.DataFrame(trade_list)
        trades["side"] = trades.apply(_classify_side, axis=1)
    else:
        trades = pd.DataFrame(columns=[
            "timestamp", "buyer", "seller", "symbol", "currency",
            "price", "quantity", "side",
        ])

    # Parse logs
    log_list = raw.get("logs", [])
    if log_list:
        logs = pd.DataFrame(log_list)
        logs["timestamp"] = pd.to_numeric(logs["timestamp"], errors="coerce")
    else:
        logs = pd.DataFrame(columns=["timestamp", "lambdaLog", "sandboxLog"])

    # Compute dashboard_wallmid from order book levels: avg of bid wall and ask wall
    bid_cols = [c for c in ["bid_price_1", "bid_price_2", "bid_price_3"] if c in activities.columns]
    ask_cols = [c for c in ["ask_price_1", "ask_price_2", "ask_price_3"] if c in activities.columns]
    if bid_cols and ask_cols:
        bid_wall = activities[bid_cols].min(axis=1)
        ask_wall = activities[ask_cols].max(axis=1)
        activities["dashboard_wallmid"] = (bid_wall + ask_wall) / 2

    # Parse wallmid from lambdaLog (optional — not all logs have it)
    activities = _merge_wallmid(activities, logs)

    return {
        "activities": activities,
        "trades": trades,
        "logs": logs,
        "submission_id": raw.get("submissionId", ""),
    }


def _classify_side(row):
    if row.get("seller") == "SUBMISSION":
        return "sell"
    if row.get("buyer") == "SUBMISSION":
        return "buy"
    return "market"


_WALLMID_RE = re.compile(r"wallmid:\s*([\d.eE+-]+)")


def _merge_wallmid(activities, logs):
    """Extract wallmid values from lambdaLog and add as a column to activities.

    Each lambdaLog entry may contain one 'wallmid: <value>' line per product,
    ordered alphabetically by product name. If no wallmid data is found the
    activities DataFrame is returned unchanged.
    """
    if logs.empty or "lambdaLog" not in logs.columns:
        return activities

    products_sorted = sorted(activities["product"].unique())
    if not products_sorted:
        return activities

    rows = []
    for _, log_row in logs.iterrows():
        ts = log_row.get("timestamp")
        text = log_row.get("lambdaLog")
        if pd.isna(ts) or not isinstance(text, str):
            continue
        values = _WALLMID_RE.findall(text)
        if len(values) != len(products_sorted):
            continue
        for product, val in zip(products_sorted, values):
            rows.append({"timestamp": int(ts), "product": product, "wallmid": float(val)})

    if not rows:
        return activities

    wm = pd.DataFrame(rows)
    activities = activities.merge(wm, on=["timestamp", "product"], how="left")
    return activities
