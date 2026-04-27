"""
Test the user's hypothesis on the live log:
  Bullish OTM signal that EXHAUSTS (drops to ~0 or flips) marks a peak;
  Bearish OTM signal that exhausts marks a trough.

Plus a few alternative peak/dip predictors:
  A) signal-flip                     — sign change of bull
  B) signal-exhaustion               — |bull| drops below threshold after run
  C) signal momentum (1st diff)      — d(bull)/dt crosses zero
  D) underlying VFE deceleration     — d(S)/dt slope flip near current
  E) cumulative-signal divergence    — price still rising while bull peaked

For each, compute: average forward return at horizons k ∈ {5, 20, 50, 100}.
A useful peak signal should give NEGATIVE forward returns; useful dip signal
gives POSITIVE forward returns.
"""
from __future__ import annotations
import json
import sys
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

LOG = Path(__file__).resolve().parent.parent / "logs" / "2026-04-26_01-00-02.log"

SIGNAL_STRIKES = [5300, 5400, 5500]
TARGETS = ["VEV_5000", "VEV_5100", "VEV_5200", "VELVETFRUIT_EXTRACT"]
ALL = TARGETS + [f"VEV_{k}" for k in SIGNAL_STRIKES]


def load_activities(path: Path) -> pd.DataFrame:
    with open(path) as f:
        obj = json.load(f)
    df = pd.read_csv(StringIO(obj["activitiesLog"]), sep=";")
    return df


def build_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    df = df[df["product"].isin(ALL)].copy()
    for c in ["bid_volume_1", "ask_volume_1"]:
        df[c] = df[c].fillna(0).astype(int)
    # build a continuous timeline (timestamp ascending). Days concatenate by adding 1e6.
    df["t"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    bids = df.pivot(index="t", columns="product", values="bid_volume_1").sort_index()
    asks = df.pivot(index="t", columns="product", values="ask_volume_1").sort_index()
    mids = df.pivot(index="t", columns="product", values="mid_price").sort_index()
    sig_syms = [f"VEV_{k}" for k in SIGNAL_STRIKES if f"VEV_{k}" in bids.columns]
    bull = sum(bids[s].fillna(0) - asks[s].fillna(0) for s in sig_syms)
    return bids, asks, mids, bull


def fwd_ret(price: pd.Series, k: int) -> pd.Series:
    return price.shift(-k) - price


def run_trigger_analysis(name: str, trigger: pd.Series, mids: pd.DataFrame,
                         horizons=(5, 20, 50, 100)) -> None:
    """Trigger is a boolean mask (events of interest). Print fwd return stats
    on each target."""
    n = int(trigger.sum())
    if n == 0:
        print(f"\n[{name}]  no events found")
        return
    print(f"\n[{name}]  events: {n}  ({n/len(trigger):.1%} of ticks)")
    print(f"  {'target':>22}  " + "  ".join(f"k={k:<3}" for k in horizons))
    for tgt in TARGETS:
        if tgt not in mids.columns:
            continue
        m = mids[tgt].ffill()
        row = []
        for k in horizons:
            fr = fwd_ret(m, k)
            mask = trigger & fr.notna()
            if mask.sum() < 20:
                row.append("    n/a")
                continue
            mu = fr[mask].mean()
            row.append(f"{mu:+7.3f}")
        print(f"  {tgt:>22}  " + "  ".join(row))


def main():
    print(f"Loading {LOG.name} …")
    df = load_activities(LOG)
    bids, asks, mids, bull = build_panel(df)
    print(f"  ticks={len(bull):,}  bull stats: mean={bull.mean():.1f}  "
          f"std={bull.std():.1f}  min={bull.min():.0f}  max={bull.max():.0f}")
    pos = (bull > 0).mean()
    neg = (bull < 0).mean()
    flat = (bull == 0).mean()
    print(f"  sign distribution: bull>0 {pos:.1%}  bull<0 {neg:.1%}  bull==0 {flat:.1%}")

    # =====================================================================
    # Baseline: any-tick fwd return when bull > 0 / bull < 0 / bull == 0
    # =====================================================================
    print("\n=== Baseline — fwd return conditional on signal sign ===")
    print(f"  {'target':>22}  k    {'sig>0':>9} {'sig<0':>9} {'sig=0':>9}  spread(>0 - <0)")
    for tgt in TARGETS:
        if tgt not in mids.columns:
            continue
        m = mids[tgt].ffill()
        for k in (5, 20, 50):
            fr = fwd_ret(m, k)
            mask = fr.notna()
            p = fr[mask & (bull > 0)].mean()
            ng = fr[mask & (bull < 0)].mean()
            z = fr[mask & (bull == 0)].mean()
            print(f"  {tgt:>22} {k:>3}  {p:+9.3f} {ng:+9.3f} {z:+9.3f}  {(p - ng):+8.3f}")

    # =====================================================================
    # A) Signal flip — sign change. Bull→Bear flip = candidate peak;
    #    Bear→Bull flip = candidate trough.
    # =====================================================================
    bull_prev = bull.shift(1)
    flip_peak = (bull_prev > 0) & (bull <= 0)   # was bullish, now isn't
    flip_dip  = (bull_prev < 0) & (bull >= 0)   # was bearish, now isn't
    run_trigger_analysis("A1: bull→non-bull (peak hypothesis)", flip_peak, mids)
    run_trigger_analysis("A2: bear→non-bear (trough hypothesis)", flip_dip, mids)

    # =====================================================================
    # B) Signal exhaustion — was strongly bull/bear, now near zero.
    # =====================================================================
    HIGH = 20
    LOW = 5
    bull_max5 = bull.shift(1).rolling(5, min_periods=1).max()
    bull_min5 = bull.shift(1).rolling(5, min_periods=1).min()
    exh_peak = (bull_max5 >= HIGH) & (bull.abs() <= LOW)
    exh_dip  = (bull_min5 <= -HIGH) & (bull.abs() <= LOW)
    run_trigger_analysis(f"B1: bull≥{HIGH}-then-|bull|≤{LOW} (peak)", exh_peak, mids)
    run_trigger_analysis(f"B2: bear≤-{HIGH}-then-|bull|≤{LOW} (trough)", exh_dip, mids)

    # =====================================================================
    # C) Signal momentum (1st diff) — bull peaked then started falling.
    # =====================================================================
    db = bull.diff()
    db_prev = db.shift(1)
    mom_peak = (bull > 0) & (db_prev >= 0) & (db < 0)   # bull was rising or flat, now falling
    mom_dip  = (bull < 0) & (db_prev <= 0) & (db > 0)   # bear was falling or flat, now rising
    run_trigger_analysis("C1: bull rising→falling (peak)", mom_peak, mids)
    run_trigger_analysis("C2: bear falling→rising (trough)", mom_dip, mids)

    # =====================================================================
    # D) Smoothed signal slope flip — robust version of C.
    # =====================================================================
    bull_smooth = bull.rolling(10, min_periods=1).mean()
    db_s = bull_smooth.diff()
    db_s_prev = db_s.shift(1)
    sm_peak = (bull_smooth > 5) & (db_s_prev > 0) & (db_s <= 0)
    sm_dip  = (bull_smooth < -5) & (db_s_prev < 0) & (db_s >= 0)
    run_trigger_analysis("D1: smoothed bull peaks (peak)", sm_peak, mids)
    run_trigger_analysis("D2: smoothed bear bottoms (trough)", sm_dip, mids)

    # =====================================================================
    # E) Cumulative-signal divergence — price still up, signal already
    #    rolled over. Price's 20-tick move > 0 AND smoothed signal already
    #    fell from its 20-tick high.
    # =====================================================================
    K_DIV = 20
    for tgt in TARGETS[:3]:  # vouchers only
        if tgt not in mids.columns:
            continue
        m = mids[tgt].ffill()
        m_chg = m - m.shift(K_DIV)
        bs_peak = bull_smooth.rolling(K_DIV, min_periods=1).max()
        bs_min  = bull_smooth.rolling(K_DIV, min_periods=1).min()
        div_peak = (m_chg > 0) & (bull_smooth < bs_peak * 0.5) & (bs_peak > 10)
        div_dip  = (m_chg < 0) & (bull_smooth > bs_min * 0.5) & (bs_min < -10)
        n_peak = int(div_peak.sum())
        n_dip = int(div_dip.sum())
        if n_peak >= 20:
            mu_p = (m.shift(-20) - m)[div_peak].mean()
            print(f"\n[E1: {tgt} divergence-peak]  n={n_peak}  fwd20 mean={mu_p:+.3f}")
        if n_dip >= 20:
            mu_d = (m.shift(-20) - m)[div_dip].mean()
            print(f"[E2: {tgt} divergence-dip]   n={n_dip}  fwd20 mean={mu_d:+.3f}")


if __name__ == "__main__":
    main()
