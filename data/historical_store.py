from data.historical_loader import load_round, get_rounds, get_available_days

_data = None
_round_name = None


def load(round_name: str):
    global _data, _round_name
    _data = load_round(round_name)
    _round_name = round_name


def is_loaded() -> bool:
    return _data is not None


def get_activities(product: str, day=None):
    df = _data["activities"]
    if df.empty:
        return df
    df = df[df["product"] == product]
    if day is not None:
        df = df[df["day"] == day]
    return df


def get_trades(product: str, day=None):
    df = _data["trades"]
    if df.empty:
        return df
    df = df[df["symbol"] == product]
    if day is not None and "day" in df.columns:
        df = df[df["day"] == day]
    return df


def get_all_activities(day=None):
    df = _data["activities"]
    if df.empty:
        return df
    if day is not None:
        df = df[df["day"] == day]
    return df


def get_all_trades(day=None):
    df = _data["trades"]
    if df.empty:
        return df
    if day is not None and "day" in df.columns:
        df = df[df["day"] == day]
    return df


def get_products():
    df = _data["activities"]
    if df.empty:
        return []
    return sorted(df["product"].unique().tolist())


def get_days():
    df = _data["activities"]
    if df.empty:
        return []
    return sorted(df["day"].unique().tolist())


def get_traders():
    """Return sorted list of unique counterparty names found in trades."""
    df = _data["trades"]
    if df.empty:
        return []
    names = set()
    for col in ("buyer", "seller"):
        if col in df.columns:
            for v in df[col].dropna().unique():
                s = str(v).strip()
                if s:
                    names.add(s)
    return sorted(names)
