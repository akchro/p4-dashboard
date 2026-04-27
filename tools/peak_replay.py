"""
Replay 2026-04-26_01-25-29.log through the new detect_peak / detect_dip
imported directly from voucher_long_short.py. Reports fire rate and fwd
return per voucher / per side.
"""
from __future__ import annotations
import importlib.util, json, sys
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "2026-04-26_01-25-29.log"
STRAT = REPO / "voucher_long_short.py"

class _Stub: ...
sys.modules.setdefault("datamodel", type(sys)("datamodel"))
sys.modules["datamodel"].OrderDepth = _Stub
sys.modules["datamodel"].TradingState = _Stub
sys.modules["datamodel"].Order = _Stub
spec = importlib.util.spec_from_file_location("voucher_long_short", STRAT)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

SIGNAL_STRIKES = mod.SIGNAL_STRIKES
TARGETS = ["VEV_5000", "VEV_5100", "VEV_5200"]


def main():
    with open(LOG) as f:
        df = pd.read_csv(StringIO(json.load(f)["activitiesLog"]), sep=";")
    keep = TARGETS + [f"VEV_{k}" for k in SIGNAL_STRIKES]
    df = df[df["product"].isin(keep)].copy()
    for c in ["bid_volume_1", "ask_volume_1"]:
        df[c] = df[c].fillna(0).astype(int)
    df["t"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    bids = df.pivot(index="t", columns="product", values="bid_volume_1").sort_index()
    asks = df.pivot(index="t", columns="product", values="ask_volume_1").sort_index()
    mids = df.pivot(index="t", columns="product", values="mid_price").sort_index()
    sig_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
    bull_arr = sum(bids[s].fillna(0) - asks[s].fillna(0) for s in sig_syms).tolist()

    print(f"=== {LOG.name} === ticks={len(bull_arr):,}")
    print(f"DIV_LOOKBACK={mod.DIV_LOOKBACK} DIV_SMOOTH={mod.DIV_SMOOTH} "
          f"DIV_MIN_BULL={mod.DIV_MIN_BULL} DIV_DECAY={mod.DIV_DECAY}")

    for tgt in TARGETS:
        m = mids[tgt].ffill().tolist()
        n = len(bull_arr)
        peak_fires = np.zeros(n, dtype=bool)
        dip_fires = np.zeros(n, dtype=bool)
        for i in range(n):
            bh = bull_arr[max(0, i - mod.BULL_HIST_LEN + 1): i + 1]
            mh = m[max(0, i - mod.MID_HIST_LEN + 1): i + 1]
            if len(mh) < mod.DIV_LOOKBACK:
                continue
            peak_fires[i] = mod.Trader.detect_peak(bh, mh)
            dip_fires[i]  = mod.Trader.detect_dip(bh, mh)

        m_arr = np.array(m, dtype=float)
        print(f"\n[{tgt}] peak fires {int(peak_fires.sum())}, dip fires {int(dip_fires.sum())}")
        for label, fires, sign in [("peak (short)", peak_fires, +1),
                                   ("dip  (long)",  dip_fires,  -1)]:
            row = []
            for k in (5, 20, 50, 100):
                fwd = np.full_like(m_arr, np.nan)
                fwd[:-k] = m_arr[k:] - m_arr[:-k]
                valid = fires & ~np.isnan(fwd)
                # PnL contribution (per-fire ret × sign), positive = profit
                mu = fwd[valid].mean() if valid.sum() else float("nan")
                pnl = -sign * mu  # short profits when fwd < 0
                row.append((k, mu, pnl))
            f = "  ".join(f"k={k}:{mu:+.3f} (pnl/fire={pnl:+.3f})" for k, mu, pnl in row)
            print(f"  {label}  {f}")


if __name__ == "__main__":
    main()
