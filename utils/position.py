import pandas as pd


def cumulative_position(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "position"])
    df = trades.sort_values("timestamp").copy()
    df["signed_qty"] = df.apply(
        lambda r: r["quantity"] if r["side"] == "buy"
        else -r["quantity"] if r["side"] == "sell"
        else 0,
        axis=1,
    )
    df["position"] = df["signed_qty"].cumsum()
    return df[["timestamp", "position"]]
