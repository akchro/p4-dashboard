"""Does Mark 67's aggressive volume on VELVETFRUIT_EXTRACT predict the market?

Mark 67 only lifts offers on the underlying. Their trade sizes range 2-15 with
median 9. The thesis: maybe big clips (12-15) signal informed buying and the
market reacts, while small clips (2-5) are noise.

For each Mark 67 trade we measure forward returns at multiple horizons,
bucket by quantity, and compare bucketed mean/median forward returns. We
also compare against an unconditional baseline (the avg forward return of
*any* random tick) and against other aggressive-buyer trades on the same
product (Mark 55 buys, Mark 22 buys, etc) to see if Mark 67 specifically
carries information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROUND4 = ROOT / "historical" / "ROUND_4"
DAYS = (1, 2, 3)
PRODUCT = "VELVETFRUIT_EXTRACT"
HORIZONS = (100, 500, 1_000, 5_000, 10_000, 50_000)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.concat(
        [pd.read_csv(ROUND4 / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS]
    )
    pr = pd.concat([pd.read_csv(ROUND4 / f"prices_round_4_day_{d}.csv", sep=";") for d in DAYS])
    return tr, pr


def mid_lookup(prices: pd.DataFrame) -> pd.DataFrame:
    """Return mid_price indexed by (day, timestamp) for one product."""
    p = prices[prices["product"] == PRODUCT][["day", "timestamp", "mid_price"]]
    return p.sort_values(["day", "timestamp"]).reset_index(drop=True)


def forward_returns(events: pd.DataFrame, mids: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """For each event row, compute mid(t+h) - mid(t) for h in horizons.

    `events` must have 'day' and 'timestamp' columns.
    """
    out = events.copy()
    for h in horizons:
        out[f"fwd_{h}"] = np.nan

    # Build per-day arrays for fast lookup
    for d in DAYS:
        m = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = m.index.values
        px_arr = m.values
        ev = out[out["day"] == d]
        if ev.empty:
            continue

        # for each event, find the mid at t (snap to nearest <=)
        # and at t+h (nearest <=)
        for i, row in ev.iterrows():
            t = row["timestamp"]
            # snap to nearest existing timestamp at or before t
            idx = np.searchsorted(ts_arr, t, side="right") - 1
            if idx < 0:
                continue
            mid_t = px_arr[idx]
            for h in horizons:
                target = t + h
                idx_h = np.searchsorted(ts_arr, target, side="right") - 1
                if idx_h < 0 or idx_h >= len(px_arr):
                    continue
                out.at[i, f"fwd_{h}"] = px_arr[idx_h] - mid_t
    return out


def baseline_returns(mids: pd.DataFrame, n_samples: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Unconditional baseline: pick random (day, timestamp) and compute fwd returns."""
    rng = np.random.default_rng(seed)
    samples = []
    for d in DAYS:
        ts_arr = mids[mids["day"] == d]["timestamp"].values
        if len(ts_arr) == 0:
            continue
        chosen = rng.choice(ts_arr, size=min(n_samples, len(ts_arr)), replace=False)
        samples.append(pd.DataFrame({"day": d, "timestamp": chosen}))
    samples = pd.concat(samples, ignore_index=True)
    return forward_returns(samples, mids)


def summarize_buckets(events: pd.DataFrame, label_col: str, label: str) -> pd.DataFrame:
    rows = []
    for bucket, sub in events.groupby(label_col):
        n = len(sub)
        if n < 3:
            continue
        row = {"group": label, "bucket": bucket, "n": n}
        for h in HORIZONS:
            col = f"fwd_{h}"
            row[f"mean_{h}"] = round(sub[col].mean(), 3)
            row[f"med_{h}"] = round(sub[col].median(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    tr, pr = load()
    mids = mid_lookup(pr)

    # All trades on VELVETFRUIT_EXTRACT
    velvet = tr[tr["symbol"] == PRODUCT].copy()

    # Mark 67 buys (their only role)
    m67 = velvet[velvet["buyer"] == "Mark 67"].copy()
    print(f"Mark 67 buys on {PRODUCT}: n={len(m67)}, qty range {int(m67['quantity'].min())}-{int(m67['quantity'].max())}")
    print(f"Quantity distribution:\n{m67['quantity'].value_counts().sort_index().to_string()}")
    print()

    # Forward returns for every Mark 67 trade
    m67_fwd = forward_returns(m67[["day", "timestamp", "quantity", "price"]], mids)

    # Bucket by quantity
    def bucket(q):
        if q <= 5: return "1_small_2-5"
        if q <= 9: return "2_med_6-9"
        if q <= 12: return "3_big_10-12"
        return "4_xl_13-15"

    m67_fwd["qbucket"] = m67_fwd["quantity"].apply(bucket)

    print("=== Mark 67 forward mid-returns by quantity bucket ===")
    summ = summarize_buckets(m67_fwd, "qbucket", "Mark 67")
    print(summ.to_string(index=False))
    print()

    # Also: per exact quantity
    print("=== Mark 67 forward mid-returns by exact quantity ===")
    summ2 = summarize_buckets(m67_fwd, "quantity", "Mark 67")
    print(summ2.to_string(index=False))
    print()

    # Baseline
    base = baseline_returns(mids, n_samples=3000)
    base_means = {f"mean_{h}": round(base[f"fwd_{h}"].mean(), 3) for h in HORIZONS}
    base_meds = {f"med_{h}": round(base[f"fwd_{h}"].median(), 3) for h in HORIZONS}
    print(f"=== Unconditional baseline (random ticks, n={len(base)}) ===")
    print(f"means: {base_means}")
    print(f"medians: {base_meds}")
    print()

    # Compare to other aggressive buyers (Mark 55 buys, Mark 22 buys)
    others_buy = velvet[velvet["buyer"].isin(["Mark 55", "Mark 22", "Mark 14", "Mark 01"]) &
                        (~velvet["buyer"].eq("Mark 67"))].copy()
    others_buy["qbucket"] = others_buy["quantity"].apply(bucket)
    print(f"=== Other aggressive/passive buyers on {PRODUCT}, n={len(others_buy)} ===")
    others_fwd = forward_returns(others_buy[["day","timestamp","quantity","buyer"]].rename(columns={"buyer":"who"}), mids)
    summ3 = summarize_buckets(others_fwd.assign(qbucket=others_fwd["quantity"].apply(bucket)),
                              "qbucket", "Other buyers")
    print(summ3.to_string(index=False))
    print()

    # Per-trader (only aggressors) at large size
    print("=== Forward returns per buyer, large clips only (qty>=10) ===")
    rows=[]
    for who in ["Mark 67","Mark 55","Mark 22","Mark 14","Mark 01"]:
        sub = velvet[(velvet["buyer"]==who) & (velvet["quantity"]>=10)]
        if len(sub) < 5: continue
        f = forward_returns(sub[["day","timestamp","quantity"]].copy(), mids)
        row = {"buyer": who, "n": len(f)}
        for h in HORIZONS:
            row[f"mean_{h}"] = round(f[f"fwd_{h}"].mean(), 3)
            row[f"med_{h}"] = round(f[f"fwd_{h}"].median(), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # Same idea for SELLERS at qty>=10
    print("=== Forward returns per seller, large clips only (qty>=10) ===")
    rows=[]
    for who in ["Mark 49","Mark 55","Mark 22","Mark 14","Mark 01"]:
        sub = velvet[(velvet["seller"]==who) & (velvet["quantity"]>=10)]
        if len(sub) < 5: continue
        f = forward_returns(sub[["day","timestamp","quantity"]].copy(), mids)
        row = {"seller": who, "n": len(f)}
        for h in HORIZONS:
            row[f"mean_{h}"] = round(f[f"fwd_{h}"].mean(), 3)
            row[f"med_{h}"] = round(f[f"fwd_{h}"].median(), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
