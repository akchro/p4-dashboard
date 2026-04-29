"""Extra counterparty checks: inventory drift, spread positioning, per-product role."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROUND4 = ROOT / "historical" / "ROUND_4"
DAYS = (1, 2, 3)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.concat(
        [pd.read_csv(ROUND4 / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS]
    )
    pr = pd.concat([pd.read_csv(ROUND4 / f"prices_round_4_day_{d}.csv", sep=";") for d in DAYS])
    return tr, pr


def annotate(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    book = prices.set_index(["day", "timestamp", "product"])[
        ["bid_price_1", "ask_price_1", "mid_price"]
    ].reset_index().rename(columns={"product": "symbol"})
    j = trades.merge(book, on=["day", "timestamp", "symbol"], how="left")
    bid, ask, px, mid = j["bid_price_1"], j["ask_price_1"], j["price"], j["mid_price"]
    j["aggressor"] = np.where(
        px >= ask, "buyer",
        np.where(px <= bid, "seller",
                 np.where(px > mid, "buyer", np.where(px < mid, "seller", "ambiguous")))
    )
    j["spread_pos"] = (px - mid) / np.where((ask - bid) > 0, (ask - bid) / 2, np.nan)
    return j


def inventory_drift(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-trader, per-product running inventory: max/min position, end position."""
    rows = []
    traders = sorted(set(trades["buyer"]).union(trades["seller"]))
    products = sorted(set(trades["symbol"]))
    for m in traders:
        for prod in products:
            sub = trades[(trades["symbol"] == prod) & ((trades["buyer"] == m) | (trades["seller"] == m))]
            if len(sub) == 0:
                continue
            sub = sub.sort_values(["day", "timestamp"])
            signed = np.where(sub["buyer"] == m, sub["quantity"], -sub["quantity"])
            pos = signed.cumsum()
            rows.append(dict(
                trader=m, product=prod, n=len(sub),
                pos_min=int(pos.min()), pos_max=int(pos.max()), pos_end=int(pos[-1]),
                qty_total=int(sub["quantity"].sum()),
            ))
    return pd.DataFrame(rows)


def role_per_product(trades: pd.DataFrame) -> pd.DataFrame:
    """For each (trader, product), summarize aggression/buy-share."""
    rows = []
    traders = sorted(set(trades["buyer"]).union(trades["seller"]))
    products = sorted(set(trades["symbol"]))
    for m in traders:
        for prod in products:
            sub = trades[(trades["symbol"] == prod) & ((trades["buyer"] == m) | (trades["seller"] == m))]
            if len(sub) == 0:
                continue
            n = len(sub)
            n_buy = (sub["buyer"] == m).sum()
            agg = ((sub["buyer"] == m) & (sub["aggressor"] == "buyer")).sum() + (
                (sub["seller"] == m) & (sub["aggressor"] == "seller")
            ).sum()
            rows.append(dict(
                trader=m, product=prod, n=n,
                buy_share=round(n_buy / n, 2),
                agg_rate=round(agg / n, 2),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    tr, pr = load()
    j = annotate(tr, pr)

    print("=== Inventory drift per (trader, product) ===")
    inv = inventory_drift(j)
    # Only show rows where the trader actually traded enough
    inv = inv[inv["n"] >= 5].sort_values(["trader", "qty_total"], ascending=[True, False])
    print(inv.to_string(index=False))
    print()

    print("=== Role per (trader, product): buy_share & agg_rate ===")
    rp = role_per_product(j)
    rp = rp[rp["n"] >= 10].sort_values(["trader", "n"], ascending=[True, False])
    print(rp.to_string(index=False))
    print()

    # Position at end of each day for the most active products
    print("=== End-of-day position drift ===")
    for m in sorted(set(j["buyer"]).union(j["seller"])):
        for prod in ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK", "VEV_5500", "VEV_6000", "VEV_4000"]:
            for d in DAYS:
                sub = j[(j["symbol"] == prod) & (j["day"] == d) & ((j["buyer"] == m) | (j["seller"] == m))]
                if len(sub) == 0:
                    continue
                signed = np.where(sub["buyer"] == m, sub["quantity"], -sub["quantity"])
                start_pos = int(signed.cumsum()[0])
                end_pos = int(signed.sum())
                if abs(end_pos) >= 1:
                    print(f"{m:>8s} | day {d} | {prod:>22s} | end_pos={end_pos:+5d} | n_trades={len(sub):3d}")


if __name__ == "__main__":
    main()
