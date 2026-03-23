import os
import glob as _glob
from data.loader import load_log

_data = None
_filepath = None
LOGS_DIR = "logs"


def load(filepath: str):
    global _data, _filepath
    _data = load_log(filepath)
    _filepath = filepath


def is_loaded() -> bool:
    return _data is not None


def get_activities(product: str, day=None):
    df = _data["activities"]
    df = df[df["product"] == product]
    if day is not None:
        df = df[df["day"] == day]
    return df


def get_trades(product: str, day=None):
    df = _data["trades"]
    df = df[df["symbol"] == product]
    if day is not None and "day" in df.columns:
        df = df[df["day"] == day]
    return df


def get_log_at(timestamp: int):
    logs = _data["logs"]
    match = logs[logs["timestamp"] == timestamp]
    if match.empty:
        return {"lambdaLog": "", "sandboxLog": ""}
    row = match.iloc[0]
    return {
        "lambdaLog": row.get("lambdaLog", ""),
        "sandboxLog": row.get("sandboxLog", ""),
    }


def get_products():
    return sorted(_data["activities"]["product"].unique().tolist())


def get_days():
    return sorted(_data["activities"]["day"].unique().tolist())


def get_submission_id():
    return _data["submission_id"]


def get_log_files():
    pattern = os.path.join(LOGS_DIR, "*.log")
    return sorted(_glob.glob(pattern))
