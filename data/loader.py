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
