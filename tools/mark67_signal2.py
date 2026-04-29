"""Deeper Mark 67 signal analysis.

Questions:
1. Is the "mid drifts up after Mark 67 trades" effect mechanical (one offer
   gets lifted, mid auto-bumps) or does it persist beyond the immediate book
   refresh?
2. Does the effect actually depend on quantity?
3. Do Mark 67 trades cluster in time (multiple trades within a few ticks of
   each other, which would conflate "single signal" with "multiple signals")?
4. Compare directly against Mark 55, the other 100%-aggressive buyer of
   underlying, on equal terms.
5. Same exercise but for sells (Mark 67 doesn't sell, so use Mark 22 sell or
   Mark 55 sell as a sell-side analog).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROUND4 = ROOT / "historical" / "ROUND_4"
DAYS = (1, 2, 3)
PRODUCT = "VELVETFRUIT_EXTRACT"
HORIZONS = (100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.concat(
        [pd.read_csv(ROUND4 / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS]
    )
    pr = pd.concat([pd.read_csv(ROUND4 / f"prices_round_4_day_{d}.csv", sep=";") for d in DAYS])
    return tr, pr


def fwd(events: pd.DataFrame, mids: pd.DataFrame) -> pd.DataFrame:
    """Vectorized forward returns at multiple horizons."""
    out = events.copy().reset_index(drop=True)
    for h in HORIZONS:
        out[f"fwd_{h}"] = np.nan
    for d in DAYS:
        m = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = m.index.to_numpy()
        px_arr = m.to_numpy()
        ev_idx = out.index[out["day"] == d]
        for i in ev_idx:
            t = out.at[i, "timestamp"]
            j = np.searchsorted(ts_arr, t, side="right") - 1
            if j < 0:
                continue
            mid_t = px_arr[j]
            for h in HORIZONS:
                k = np.searchsorted(ts_arr, t + h, side="right") - 1
                if 0 <= k < len(px_arr):
                    out.at[i, f"fwd_{h}"] = px_arr[k] - mid_t
    return out


def trader_buy_summary(velvet, mids, who):
    sub = velvet[velvet["buyer"] == who][["day", "timestamp", "quantity"]].copy()
    f = fwd(sub, mids)
    row = {"who": who, "side": "buy", "n": len(f), "avg_qty": round(f["quantity"].mean(), 2)}
    for h in HORIZONS:
        row[f"mean_{h}"] = round(f[f"fwd_{h}"].mean(), 3)
    return row


def trader_sell_summary(velvet, mids, who):
    sub = velvet[velvet["seller"] == who][["day", "timestamp", "quantity"]].copy()
    f = fwd(sub, mids)
    row = {"who": who, "side": "sell", "n": len(f), "avg_qty": round(f["quantity"].mean(), 2)}
    for h in HORIZONS:
        row[f"mean_{h}"] = round(f[f"fwd_{h}"].mean(), 3)
    return row


def main():
    tr, pr = load()
    velvet = tr[tr["symbol"] == PRODUCT].copy()
    mids = pr[pr["product"] == PRODUCT][["day", "timestamp", "mid_price"]].copy()

    # 1) Per-trader buy/sell signal table
    print("=== Forward mid-returns per trader (all sizes) ===")
    rows = []
    for who in ["Mark 67", "Mark 55", "Mark 22", "Mark 14", "Mark 01"]:
        if (velvet["buyer"] == who).sum() >= 5:
            rows.append(trader_buy_summary(velvet, mids, who))
    for who in ["Mark 49", "Mark 55", "Mark 22", "Mark 14", "Mark 01"]:
        if (velvet["seller"] == who).sum() >= 5:
            rows.append(trader_sell_summary(velvet, mids, who))
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # 2) Mechanical vs persistent: compare mid_after_book vs mid_now
    # The "fwd_100" still has the trade in it (mid resets on next tick because
    # the trade itself didn't change the book — the *bid_1 / ask_1 from the
    # next snapshot* may already reflect quote refreshes from MMs).
    # So look at fwd starting one tick *before* the trade vs after the trade
    # for an estimate of "trade-induced impact".
    m67 = velvet[velvet["buyer"] == "Mark 67"][["day", "timestamp", "quantity"]].copy()
    # mid AT trade tick (snap to most recent <=)
    # mid 1 tick AFTER (snap to first >=)
    rows = []
    for d in DAYS:
        msub = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = msub.index.to_numpy()
        px_arr = msub.to_numpy()
        for _, ev in m67[m67["day"] == d].iterrows():
            t = ev["timestamp"]
            i = np.searchsorted(ts_arr, t, side="right") - 1
            i_next = np.searchsorted(ts_arr, t, side="right")
            if i < 0 or i_next >= len(px_arr):
                continue
            rows.append(dict(qty=ev["quantity"],
                             mid_at=px_arr[i],
                             mid_next=px_arr[i_next],
                             jump=px_arr[i_next] - px_arr[i]))
    j = pd.DataFrame(rows)
    print("=== Mark 67: instantaneous mid jump on the trade tick ===")
    print(f"n={len(j)}, mean jump = {j['jump'].mean():.3f}, median = {j['jump'].median():.1f}")
    print(f"Distribution of jump:\n{j['jump'].value_counts().sort_index().to_string()}")
    print()
    # Same for Mark 55 buys
    rows = []
    m55b = velvet[velvet["buyer"] == "Mark 55"][["day", "timestamp", "quantity"]].copy()
    for d in DAYS:
        msub = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = msub.index.to_numpy()
        px_arr = msub.to_numpy()
        for _, ev in m55b[m55b["day"] == d].iterrows():
            t = ev["timestamp"]
            i = np.searchsorted(ts_arr, t, side="right") - 1
            i_next = np.searchsorted(ts_arr, t, side="right")
            if i < 0 or i_next >= len(px_arr):
                continue
            rows.append(dict(qty=ev["quantity"], jump=px_arr[i_next] - px_arr[i]))
    j55 = pd.DataFrame(rows)
    print(f"Mark 55 buys instantaneous jump: n={len(j55)}, mean={j55['jump'].mean():.3f}, median={j55['jump'].median():.1f}")
    rows = []
    m55s = velvet[velvet["seller"] == "Mark 55"][["day", "timestamp", "quantity"]].copy()
    for d in DAYS:
        msub = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = msub.index.to_numpy()
        px_arr = msub.to_numpy()
        for _, ev in m55s[m55s["day"] == d].iterrows():
            t = ev["timestamp"]
            i = np.searchsorted(ts_arr, t, side="right") - 1
            i_next = np.searchsorted(ts_arr, t, side="right")
            if i < 0 or i_next >= len(px_arr):
                continue
            rows.append(dict(qty=ev["quantity"], jump=px_arr[i_next] - px_arr[i]))
    j55s = pd.DataFrame(rows)
    print(f"Mark 55 sells instantaneous jump: n={len(j55s)}, mean={j55s['jump'].mean():.3f}, median={j55s['jump'].median():.1f}")
    print()

    # 3) Persistence beyond mechanical bump:
    # subtract the immediate jump from each forward return.
    # If the residual is still positive at fwd_5000, there's real signal.
    print("=== Mark 67: forward return AFTER subtracting instantaneous jump ===")
    sub = velvet[velvet["buyer"] == "Mark 67"][["day", "timestamp", "quantity"]].copy()
    f = fwd(sub, mids)
    # need to also compute jump per row
    jumps = []
    for d in DAYS:
        msub = mids[mids["day"] == d].set_index("timestamp")["mid_price"].sort_index()
        ts_arr = msub.index.to_numpy()
        px_arr = msub.to_numpy()
        rs = f[f["day"] == d]
        for _, ev in rs.iterrows():
            t = ev["timestamp"]
            i = np.searchsorted(ts_arr, t, side="right") - 1
            i_next = np.searchsorted(ts_arr, t, side="right")
            if i < 0 or i_next >= len(px_arr):
                jumps.append(np.nan)
            else:
                jumps.append(px_arr[i_next] - px_arr[i])
    f["jump"] = jumps
    for h in HORIZONS:
        f[f"residual_{h}"] = f[f"fwd_{h}"] - f["jump"]
    print(f"means by qty bucket:")
    f["qbucket"] = pd.cut(f["quantity"], [0, 5, 9, 12, 15], labels=["small_2-5", "med_6-9", "big_10-12", "xl_13-15"])
    summ = f.groupby("qbucket", observed=True).agg(
        n=("quantity", "size"),
        avg_q=("quantity", "mean"),
        **{f"mean_{h}": (f"fwd_{h}", "mean") for h in HORIZONS},
        **{f"resid_{h}": (f"residual_{h}", "mean") for h in HORIZONS},
        jump=("jump", "mean"),
    ).round(3)
    print(summ.to_string())
    print()

    # 4) Time-clustering of Mark 67 trades. Are they bursts?
    print("=== Mark 67 trade timing: gap to next Mark 67 trade ===")
    for d in DAYS:
        sub = velvet[(velvet["buyer"] == "Mark 67") & (velvet["day"] == d)].sort_values("timestamp")
        if len(sub) < 2:
            continue
        gaps = np.diff(sub["timestamp"].to_numpy())
        print(f"day {d}: n={len(sub)}, gap p25={np.percentile(gaps,25):.0f}, "
              f"median={np.median(gaps):.0f}, p75={np.percentile(gaps,75):.0f}, mean={gaps.mean():.0f}")
    print()

    # 5) For "isolated" Mark 67 trades (no other Mark 67 trade within 5000 ticks),
    # is the forward return still up?
    print("=== Mark 67: 'isolated' trades only (no follow-up within 5000 ticks) ===")
    rows = []
    for d in DAYS:
        sub = velvet[(velvet["buyer"] == "Mark 67") & (velvet["day"] == d)].sort_values("timestamp")
        ts = sub["timestamp"].to_numpy()
        for i, t in enumerate(ts):
            next_gap = ts[i + 1] - t if i + 1 < len(ts) else 1_000_000
            if next_gap >= 5000:
                rows.append(dict(day=d, timestamp=t, quantity=sub.iloc[i]["quantity"]))
    iso = pd.DataFrame(rows)
    f_iso = fwd(iso, mids)
    iso_means = {f"mean_{h}": round(f_iso[f"fwd_{h}"].mean(), 3) for h in HORIZONS}
    print(f"n={len(f_iso)}, avg_qty={f_iso['quantity'].mean():.2f}")
    print(f"forward means: {iso_means}")


if __name__ == "__main__":
    main()
