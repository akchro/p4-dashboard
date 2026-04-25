"""Realized vol diagnostic for VELVETFRUIT_EXTRACT — multi-day, multi-frequency.

Key questions:
  1. What is the true realized σ once bid-ask bounce is removed?
  2. Does it match the IV the surface is quoting?
  3. Does that gap persist day-over-day or jump around?

Background math:
  · Bid-ask bounce: per-tick mid moves between bid and ask each time the
    side touched flips. Inflates short-frequency variance by 2δ²/4 = δ²/2.
  · Roll (1984): true variance per tick = obs variance - 2|cov(r_t, r_{t-1})|
    when bounce is the dominant noise.
  · Reasonable signal threshold: per-interval price std should exceed the
    bid-ask spread (~6 ticks). At σ_per_day ≈ 0.013 and S ≈ 5250, this
    needs intervals of order √(δ/(σ·S))² × 10000 ticks → ~50+ tick spacing
    starts to dominate bounce.

Subcommands:
  freq     realized σ at multiple sampling intervals (full round, optionally per day)
  cone     vol cone: distribution of realized σ over rolling windows
  iv_vs_S  scatter: ΔIV vs ΔS at fixed K (does IV move when S moves?)
  pnl      simulate gamma P&L for a delta-hedged short ATM call (sell vol if rich)
           and a delta-hedged long ATM call (buy vol if cheap) at given hedge cadence.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import historical_loader
from utils import options as opts


def _underlying(acts, day=None):
    df = acts[acts["product"] == opts.UNDERLYING].copy()
    if day is not None:
        df = df[df["day"] == day]
    return df.sort_values(["day", "timestamp"])[["day", "timestamp", "mid_price",
                                                  "bid_price_1", "ask_price_1"]]


def _vouchers(acts, day=None):
    df = acts[acts["product"].str.startswith("VEV_", na=False)].copy()
    if day is not None:
        df = df[df["day"] == day]
    return df.sort_values(["day", "timestamp"])


def _rv_at_interval(log_S, k):
    """Realized σ per √day from log returns at every k-th tick.

    Returns (sigma_perday, n_returns, roll_corrected_sigma)."""
    if k >= len(log_S):
        return float("nan"), 0, float("nan")
    s = log_S[::k]
    if len(s) < 5:
        return float("nan"), 0, float("nan")
    r = np.diff(s)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return float("nan"), 0, float("nan")
    var_per = r.var(ddof=1)
    n_per_day = 10000 / k
    sigma_perday = np.sqrt(var_per * n_per_day)
    # Roll-bounce correction: subtract 2*|cov(r_t, r_{t-1})| if negative.
    if len(r) > 5:
        cov1 = np.cov(r[:-1], r[1:], ddof=1)[0, 1]
        adj_var = var_per - 2 * abs(min(cov1, 0))
        adj_var = max(adj_var, 1e-12)
    else:
        adj_var = var_per
    sigma_roll = np.sqrt(adj_var * n_per_day)
    return sigma_perday, len(r), sigma_roll


def cmd_freq(args):
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]

    intervals = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
    print(f"Realized σ of {opts.UNDERLYING} mid (per √day, ×1e-2) "
          f"+ Roll-corrected · {args.round}")
    print(f"  k = sampling interval (in 100-ts ticks; 10000 ticks per day)")

    if args.combined:
        # All 3 days back-to-back. Treat as one continuous series.
        ue = _underlying(acts).dropna(subset=["mid_price"])
        S = ue["mid_price"].to_numpy()
        log_S = np.log(S)
        print(f"  Combined ALL days · n_obs={len(S)}")
        print(f"  {'k':>5} {'n_ret':>6} {'σ_obs':>8} {'σ_roll':>8} {'%bounce':>8}")
        for k in intervals:
            s_obs, n, s_roll = _rv_at_interval(log_S, k)
            pct = 100 * (1 - (s_roll / s_obs)) if np.isfinite(s_obs) and s_obs > 0 else float("nan")
            print(f"  {k:>5} {n:>6} {s_obs*1e2:>7.4f} {s_roll*1e2:>7.4f} "
                  f"{pct:>7.1f}%")

    print()
    print(f"  Per-day breakdown (σ_obs · σ_roll · ×1e-2):")
    print(f"  {'day':>4} " + "  ".join(f"k={k:<4}" for k in intervals))

    for day in sorted(acts["day"].unique()):
        ue = _underlying(acts, day).dropna(subset=["mid_price"])
        S = ue["mid_price"].to_numpy()
        if len(S) < 100:
            continue
        log_S = np.log(S)
        cells = []
        for k in intervals:
            s_obs, _, s_roll = _rv_at_interval(log_S, k)
            cells.append(f"{s_obs*1e2:>4.2f}/{s_roll*1e2:<4.2f}")
        print(f"  {day:>4}  " + "  ".join(cells))

    # Same calc for IV mean per day
    print()
    print(f"  Mean IV across strikes 5000-5500 (×1e-2):")
    for day in sorted(acts["day"].unique()):
        tte = opts.tte_for_day(args.round, day)
        ue = acts[(acts["day"] == day) & (acts["product"] == opts.UNDERLYING)] \
            .sort_values("timestamp")
        merged = ue[["timestamp", "mid_price"]].rename(columns={"mid_price": "S"})
        ivs = []
        for K in [5000, 5100, 5200, 5300, 5400, 5500]:
            v = acts[(acts["day"] == day) & (acts["product"] == f"VEV_{K}")] \
                .sort_values("timestamp")[["timestamp", "mid_price"]] \
                .rename(columns={"mid_price": "C"})
            df = pd.merge(merged, v, on="timestamp")
            T = opts.time_to_expiry(df["timestamp"].values, tte)
            iv = opts.implied_vol(df["C"].values, df["S"].values,
                                   np.full(len(df), float(K)), T)
            iv = iv[np.isfinite(iv)]
            ivs.append(iv.mean() if len(iv) else np.nan)
        print(f"  day {day} · IV mean = {np.nanmean(ivs)*1e2:.4f}  "
              f"(per strike: {[f'{v*1e2:.3f}' for v in ivs]})")


def cmd_cone(args):
    """Vol-cone: realized σ over rolling windows of various sizes."""
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    ue = _underlying(acts, args.day).dropna(subset=["mid_price"])
    S = ue["mid_price"].to_numpy()
    log_S = np.log(S)

    # Window sizes in 100-ts ticks
    windows = [50, 100, 200, 500, 1000, 2000]
    interval = args.interval

    print(f"Vol cone · day {args.day if args.day is not None else 'all'} "
          f"· interval={interval}")
    print(f"  Each window slides through the day; we compute realized σ_perday")
    print(f"  using returns spaced {interval} ticks apart inside each window.")
    print(f"  {'window':>8} {'n_obs':>6} {'p10':>8} {'p25':>8} "
          f"{'median':>8} {'p75':>8} {'p90':>8}")
    r = np.diff(log_S[::interval])
    if len(r) < max(windows):
        print("  (not enough returns for largest window)")
        return
    n_per_day = 10000 / interval
    for W in windows:
        # Rolling std × sqrt(n_per_day)
        rolling = pd.Series(r).rolling(W).std(ddof=1) * np.sqrt(n_per_day)
        rolling = rolling.dropna()
        if len(rolling) < 2:
            continue
        print(f"  {W:>8} {len(rolling):>6} "
              f"{np.percentile(rolling,10)*1e2:>7.4f} "
              f"{np.percentile(rolling,25)*1e2:>7.4f} "
              f"{np.percentile(rolling,50)*1e2:>7.4f} "
              f"{np.percentile(rolling,75)*1e2:>7.4f} "
              f"{np.percentile(rolling,90)*1e2:>7.4f}")


def cmd_iv_vs_s(args):
    """Does IV move when S moves? Regression of ΔIV on ΔS at given lag."""
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    days = [args.day] if args.day is not None else sorted(acts["day"].unique())
    interval = args.interval

    print(f"ΔIV vs ΔS regression · interval={interval} ticks")
    print(f"  Slope=∂IV/∂S, ×1e6 (e.g. 50 means IV moves +50e-6 per 1-point S move)")
    print(f"  {'day':>4} {'K':>5} {'n':>5} {'ΔIV mean':>10} {'ΔS mean':>10} "
          f"{'slope·1e6':>10} {'corr':>6} {'R²':>6}")

    for day in days:
        tte = opts.tte_for_day(args.round, day)
        for K in [5000, 5100, 5200, 5300, 5400, 5500]:
            ue = acts[(acts["day"] == day) & (acts["product"] == opts.UNDERLYING)] \
                .sort_values("timestamp")
            v = acts[(acts["day"] == day) & (acts["product"] == f"VEV_{K}")] \
                .sort_values("timestamp")
            m = pd.merge(ue[["timestamp", "mid_price"]],
                         v[["timestamp", "mid_price"]],
                         on="timestamp", suffixes=("_S", "_C"))
            if len(m) < interval * 5:
                continue
            T = opts.time_to_expiry(m["timestamp"].values, tte)
            iv = opts.implied_vol(m["mid_price_C"].values,
                                   m["mid_price_S"].values,
                                   np.full(len(m), float(K)), T)
            S = m["mid_price_S"].to_numpy()
            mask = np.isfinite(iv) & np.isfinite(S)
            iv = iv[mask]
            S = S[mask]
            if len(iv) < interval * 5:
                continue
            d_iv = iv[interval:] - iv[:-interval]
            d_S = S[interval:] - S[:-interval]
            ok = np.isfinite(d_iv) & np.isfinite(d_S)
            d_iv, d_S = d_iv[ok], d_S[ok]
            if len(d_iv) < 50:
                continue
            slope, _ = np.polyfit(d_S, d_iv, 1)
            corr = np.corrcoef(d_S, d_iv)[0, 1]
            print(f"  {day:>4} {K:>5} {len(d_iv):>5} "
                  f"{d_iv.mean()*1e6:>+9.2f} {d_S.mean():>+9.4f} "
                  f"{slope*1e6:>+9.3f} {corr:>+6.3f} {corr**2:>5.3f}")


def cmd_pnl(args):
    """Long-vol P&L sim. Two modes:
       --realistic: pay ask to buy call, receive bid when shorting S, etc.
       (default): use mid prices throughout (idealized).

    Long 1 call at t=0, initialize delta hedge, rebalance every hedge_every ticks,
    liquidate at end of day. Reports both gross (mid) and net (with spreads) P&L.
    """
    rounds = historical_loader.load_round(args.round)
    acts = rounds["activities"]
    days = [args.day] if args.day is not None else sorted(acts["day"].unique())
    K = args.strike
    hedge_every = args.hedge_every

    print(f"Delta-hedged long-call P&L · K={K} · hedge_every={hedge_every} ticks "
          f"· {'with bid-ask spreads' if args.realistic else 'mid only'}")
    print(f"  {'day':>4} {'C0':>8} {'σ_iv':>7} {'rv':>7} "
          f"{'pnl_call':>9} {'pnl_hedge':>10} {'pnl_total':>10} "
          f"{'n_rebal':>7} {'spr_cost':>9}")

    for day in days:
        tte = opts.tte_for_day(args.round, day)
        ue = acts[(acts["day"] == day) & (acts["product"] == opts.UNDERLYING)] \
            .sort_values("timestamp")[["timestamp", "mid_price",
                                        "bid_price_1", "ask_price_1"]] \
            .rename(columns={"mid_price": "S_mid", "bid_price_1": "S_bid",
                             "ask_price_1": "S_ask"})
        v = acts[(acts["day"] == day) & (acts["product"] == f"VEV_{K}")] \
            .sort_values("timestamp")[["timestamp", "mid_price",
                                        "bid_price_1", "ask_price_1"]] \
            .rename(columns={"mid_price": "C_mid", "bid_price_1": "C_bid",
                             "ask_price_1": "C_ask"})
        m = pd.merge(ue, v, on="timestamp").dropna()
        if len(m) < 100:
            continue
        S_mid = m["S_mid"].to_numpy()
        S_bid = m["S_bid"].to_numpy()
        S_ask = m["S_ask"].to_numpy()
        C_mid = m["C_mid"].to_numpy()
        C_bid = m["C_bid"].to_numpy()
        C_ask = m["C_ask"].to_numpy()
        T_arr = opts.time_to_expiry(m["timestamp"].to_numpy(), tte)

        iv0 = float(opts.implied_vol(np.array([C_mid[0]]), S_mid[0],
                                       float(K), T_arr[0])[0])
        sigma = iv0 if np.isfinite(iv0) else 0.013
        d_init = float(opts.greeks(np.array([S_mid[0]]), np.array([float(K)]),
                                    np.array([T_arr[0]]), np.array([sigma]))["delta"][0])

        # Entry: buy call, short delta of S
        if args.realistic:
            entry_call = C_ask[0]      # we pay ask
            entry_S_short = S_bid[0]   # we receive bid for shorting
        else:
            entry_call = C_mid[0]
            entry_S_short = S_mid[0]
        cash = d_init * entry_S_short  # received from short
        cash -= entry_call             # paid for call
        hedge_pos = -d_init
        spread_cost = (C_ask[0] - C_mid[0]) + d_init * (S_mid[0] - S_bid[0])

        n_rebal = 0
        last_hedge_idx = 0
        for i in range(1, len(m)):
            if (i - last_hedge_idx) >= hedge_every:
                iv_now = float(opts.implied_vol(np.array([C_mid[i]]),
                                                  S_mid[i], float(K),
                                                  T_arr[i])[0])
                sigma_now = iv_now if np.isfinite(iv_now) else sigma
                new_delta = float(opts.greeks(np.array([S_mid[i]]),
                                                np.array([float(K)]),
                                                np.array([T_arr[i]]),
                                                np.array([sigma_now]))["delta"][0])
                hedge_change = -new_delta - hedge_pos
                if args.realistic:
                    px = S_ask[i] if hedge_change > 0 else S_bid[i]
                    spread_cost += abs(hedge_change) * abs(S_mid[i] - px)
                else:
                    px = S_mid[i]
                cash -= hedge_change * px
                hedge_pos = -new_delta
                last_hedge_idx = i
                n_rebal += 1

        # Liquidate at end of day
        if args.realistic:
            exit_call = C_bid[-1]   # sell call at bid
            exit_S_close = S_ask[-1]  # cover short at ask
            spread_cost += (C_mid[-1] - C_bid[-1])
            spread_cost += abs(hedge_pos) * (S_ask[-1] - S_mid[-1])
        else:
            exit_call = C_mid[-1]
            exit_S_close = S_mid[-1]
        cash += exit_call
        cash += hedge_pos * exit_S_close

        call_pnl = (exit_call - entry_call)
        hedge_pnl_only = cash - call_pnl
        total = cash

        log_S = np.log(S_mid)
        r = np.diff(log_S[::5])
        rv = float(r.std(ddof=1) * np.sqrt(2000)) if len(r) > 5 else float("nan")

        print(f"  {day:>4} {C_mid[0]:>8.2f} {iv0*1e2:>6.3f} {rv*1e2:>6.3f} "
              f"{call_pnl:>+9.2f} {hedge_pnl_only:>+10.2f} {total:>+10.2f} "
              f"{n_rebal:>7} {spread_cost:>9.2f}")


def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("freq", help="realized σ at multiple frequencies (multi-day)")
    sp.add_argument("--round", default="ROUND_3")
    sp.add_argument("--combined", action="store_true",
                    help="treat all days as one continuous series (best statistics)")
    sp.set_defaults(fn=cmd_freq)

    sp = sub.add_parser("cone", help="rolling vol cone")
    sp.add_argument("--round", default="ROUND_3")
    sp.add_argument("--day", type=int)
    sp.add_argument("--interval", type=int, default=10,
                    help="return spacing in 100-ts ticks (default 10 = 1000 ts)")
    sp.set_defaults(fn=cmd_cone)

    sp = sub.add_parser("iv_vs_s", help="ΔIV vs ΔS regression at fixed K")
    sp.add_argument("--round", default="ROUND_3")
    sp.add_argument("--day", type=int)
    sp.add_argument("--interval", type=int, default=10)
    sp.set_defaults(fn=cmd_iv_vs_s)

    sp = sub.add_parser("pnl", help="delta-hedged long call P&L (gamma-scalp)")
    sp.add_argument("--round", default="ROUND_3")
    sp.add_argument("--day", type=int)
    sp.add_argument("--strike", type=int, default=5200)
    sp.add_argument("--hedge-every", type=int, default=10)
    sp.add_argument("--realistic", action="store_true",
                    help="cross bid-ask on every fill (entry, hedge rebalance, exit)")
    sp.set_defaults(fn=cmd_pnl)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
