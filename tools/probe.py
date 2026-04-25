"""CLI for inspecting the same data the dashboard visualizes.

Usage (from repo root):
    python3 tools/probe.py <subcommand> [args...]

Subcommands:
    summary    per-product mid-price statistics
    price      mid/book price over a time range or window
    book       order book snapshot at a timestamp
    trades     filtered trade log
    smile      IV across strikes at a timestamp
    iv         IV distribution per strike over a day
    arb        find no-arb violations (voucher priced outside bounds)
    greeks     BS greeks at a timestamp (uses market IV)

All subcommands default to --round ROUND_3. Use --day to pick a day (0/1/2).
Use --json for machine-readable output.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import historical_loader
from utils import options as opts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CACHE = {}


def _load(round_name):
    if round_name not in _CACHE:
        _CACHE[round_name] = historical_loader.load_round(round_name)
    return _CACHE[round_name]


def _acts(round_name, product=None, day=None):
    df = _load(round_name)["activities"]
    if product is not None:
        df = df[df["product"] == product]
    if day is not None:
        df = df[df["day"] == day]
    return df


def _trades(round_name, product=None, day=None):
    df = _load(round_name)["trades"]
    if df.empty:
        return df
    if product is not None:
        df = df[df["symbol"] == product]
    if day is not None and "day" in df.columns:
        df = df[df["day"] == day]
    return df


def _parse_range(s):
    """'100:500' -> (100, 500). ':500' -> (0, 500). '100:' -> (100, 10**9)."""
    if s is None:
        return (0, 10**9)
    if ":" not in s:
        v = int(s)
        return (v, v)
    lo, hi = s.split(":", 1)
    return (int(lo) if lo else 0, int(hi) if hi else 10**9)


def _tte(round_name, day, override):
    if override is not None:
        return float(override)
    return opts.tte_for_day(round_name, day)


def _nearest(df, ts, col="timestamp"):
    if df.empty:
        return None
    i = (df[col] - ts).abs().idxmin()
    return df.loc[i]


def _fmt_float(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_summary(args):
    df = _load(args.round)["activities"]
    if args.day is not None:
        df = df[df["day"] == args.day]
    rows = []
    for product in sorted(df["product"].unique()):
        mp = df[df["product"] == product]["mid_price"].dropna()
        if mp.empty:
            continue
        rows.append({
            "product": product, "n": int(len(mp)),
            "min": float(mp.min()), "max": float(mp.max()),
            "mean": float(mp.mean()), "std": float(mp.std()),
        })
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    day_label = f"day {args.day}" if args.day is not None else "all days"
    print(f"Summary · {args.round} · {day_label}")
    print(f"  {'product':<24} {'n':>6} {'min':>10} {'max':>10} {'mean':>10} {'std':>8}")
    for r in rows:
        print(f"  {r['product']:<24} {r['n']:>6} "
              f"{r['min']:>10.2f} {r['max']:>10.2f} {r['mean']:>10.2f} {r['std']:>8.3f}")


def cmd_price(args):
    df = _acts(args.round, args.product, args.day)
    lo, hi = _parse_range(args.range)
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    if df.empty:
        print(f"(no {args.product} data in [{lo}, {hi}])")
        return
    cols = ["timestamp", "mid_price", "bid_price_1", "ask_price_1",
            "bid_volume_1", "ask_volume_1"]
    cols = [c for c in cols if c in df.columns]

    if args.stats:
        mp = df["mid_price"].dropna()
        out = {
            "product": args.product, "day": args.day,
            "range": [int(df["timestamp"].min()), int(df["timestamp"].max())],
            "n": int(len(mp)), "min": float(mp.min()), "max": float(mp.max()),
            "mean": float(mp.mean()), "std": float(mp.std()),
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"{args.product} day {args.day} · t in [{out['range'][0]:,}, {out['range'][1]:,}]")
            print(f"  n={out['n']} min={out['min']:.2f} max={out['max']:.2f} "
                  f"mean={out['mean']:.2f} std={out['std']:.3f}")
        return

    if args.sample:
        step = max(len(df) // args.sample, 1)
        df = df.iloc[::step]

    if args.json:
        print(df[cols].to_json(orient="records"))
    else:
        print(df[cols].to_string(index=False))


def cmd_book(args):
    df = _acts(args.round, args.product, args.day)
    row = _nearest(df, args.ts)
    if row is None:
        print(f"(no {args.product} on day {args.day})")
        return
    out = {"product": args.product, "day": args.day,
           "ts_requested": args.ts, "ts_actual": int(row["timestamp"]),
           "mid": float(row["mid_price"]) if pd.notna(row["mid_price"]) else None,
           "bids": [], "asks": []}
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp):
            out["bids"].append({"price": float(bp), "volume": int(bv) if pd.notna(bv) else None})
        if pd.notna(ap):
            out["asks"].append({"price": float(ap), "volume": int(av) if pd.notna(av) else None})

    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"{args.product} day {args.day} · t={out['ts_actual']:,} (nearest to {args.ts:,})")
    print(f"  mid = {out['mid']}")
    for a in reversed(out["asks"]):
        print(f"  ASK  {a['price']:>10.2f}  × {a['volume']}")
    print(f"  " + "-" * 30)
    for b in out["bids"]:
        print(f"  BID  {b['price']:>10.2f}  × {b['volume']}")


def cmd_trades(args):
    df = _trades(args.round, args.product, args.day)
    if df.empty:
        print(f"(no trades for {args.product} on day {args.day})")
        return
    lo, hi = _parse_range(args.range)
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    if df.empty:
        print(f"(no trades in [{lo}, {hi}])")
        return
    if args.min_qty:
        df = df[df["quantity"] >= args.min_qty]
    if args.json:
        print(df.to_json(orient="records"))
        return
    cols = [c for c in ["timestamp", "symbol", "price", "quantity", "buyer", "seller"] if c in df.columns]
    print(df[cols].to_string(index=False))


def cmd_smile(args):
    ve = _acts(args.round, opts.UNDERLYING, args.day)
    row = _nearest(ve.dropna(subset=["mid_price"]), args.ts)
    if row is None:
        print(f"(no {opts.UNDERLYING} data)")
        return
    S = float(row["mid_price"])
    actual_ts = int(row["timestamp"])
    tte_start = _tte(args.round, args.day, args.tte)
    T = max(tte_start - actual_ts / opts.TIMESTAMP_PER_DAY, 1e-6)

    rows = []
    for K in opts.STRIKES:
        v = _acts(args.round, f"VEV_{K}", args.day)
        v_row = _nearest(v.dropna(subset=["mid_price"]), args.ts)
        if v_row is None:
            continue
        C = float(v_row["mid_price"])
        if not np.isfinite(C):
            continue
        intrinsic = max(S - K, 0.0)
        m = float(np.log(K / S) / np.sqrt(T))
        iv = float(opts.implied_vol(np.array([C]), S, float(K), T)[0])
        rows.append({
            "K": K, "C": C, "intr": intrinsic, "extr": C - intrinsic,
            "m": m, "iv": iv,
        })

    if args.json:
        print(json.dumps({
            "S": S, "T": T, "ts_actual": actual_ts, "tte_start": tte_start,
            "smile": rows,
        }, indent=2, default=lambda x: None if pd.isna(x) else x))
        return

    print(f"Smile · {args.round} day {args.day} · t={actual_ts:,} "
          f"· S={S:.2f} · T={T:.3f}d (TTE@start={tte_start})")
    print(f"  {'K':>6} {'C':>8} {'intr':>8} {'extr':>8} {'m':>8} {'IV':>10}")
    for r in rows:
        iv_s = _fmt_float(r["iv"], 5)
        print(f"  {r['K']:>6} {r['C']:>8.2f} {r['intr']:>8.2f} {r['extr']:>8.2f} "
              f"{r['m']:>8.3f} {iv_s:>10}")


def cmd_iv(args):
    ve = _acts(args.round, opts.UNDERLYING, args.day)[["timestamp", "mid_price"]].rename(
        columns={"mid_price": "S"})
    strikes = args.strikes or opts.STRIKES
    tte_start = _tte(args.round, args.day, args.tte)
    out = []
    for K in strikes:
        v = _acts(args.round, f"VEV_{K}", args.day)[["timestamp", "mid_price"]].rename(
            columns={"mid_price": "C"})
        merged = pd.merge(v, ve, on="timestamp", how="inner").dropna()
        if merged.empty:
            continue
        T = opts.time_to_expiry(merged["timestamp"].values, tte_start)
        iv = opts.implied_vol(merged["C"].values, merged["S"].values,
                              np.full(len(merged), K, float), T)
        iv = iv[np.isfinite(iv)]
        if len(iv) == 0:
            continue
        out.append({
            "strike": int(K), "n": int(len(iv)),
            "min": float(iv.min()), "p5": float(np.percentile(iv, 5)),
            "median": float(np.median(iv)), "mean": float(iv.mean()),
            "p95": float(np.percentile(iv, 95)), "max": float(iv.max()),
            "std": float(iv.std()),
        })

    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"IV distribution · {args.round} day {args.day} · TTE@start={tte_start}d")
    print(f"  {'K':>6} {'n':>6} {'min':>8} {'p5':>8} {'median':>8} "
          f"{'mean':>8} {'p95':>8} {'max':>8} {'std':>8}")
    for r in out:
        print(f"  {r['strike']:>6} {r['n']:>6} "
              f"{r['min']:>8.5f} {r['p5']:>8.5f} {r['median']:>8.5f} "
              f"{r['mean']:>8.5f} {r['p95']:>8.5f} {r['max']:>8.5f} {r['std']:>8.5f}")


def cmd_arb(args):
    """Find timestamps where voucher bid/ask violates no-arb bounds.

    Violations checked:
      ask < max(S − K, 0)   → lift the voucher, short the stock, risk-free
      bid > S               → sell the voucher, buy stock, risk-free
    """
    ve = _acts(args.round, opts.UNDERLYING, args.day)[
        ["timestamp", "mid_price", "bid_price_1", "ask_price_1"]
    ].rename(columns={"mid_price": "S", "bid_price_1": "S_bid", "ask_price_1": "S_ask"})
    if ve.empty:
        print("(no underlying)")
        return
    strikes = args.strikes or opts.STRIKES

    report = []
    for K in strikes:
        v = _acts(args.round, f"VEV_{K}", args.day)[
            ["timestamp", "mid_price", "bid_price_1", "ask_price_1"]
        ].rename(columns={"mid_price": "C", "bid_price_1": "C_bid", "ask_price_1": "C_ask"})
        if v.empty:
            continue
        m = pd.merge(v, ve, on="timestamp", how="inner")

        # Use S_ask (the price we'd pay when hedging) for intrinsic upper bound check.
        intrinsic_against_ask = np.maximum(m["S_bid"] - K, 0.0)  # min achievable S-K from shorting
        ask_violations = m[m["C_ask"].notna() & (m["C_ask"] < intrinsic_against_ask - args.tol)]
        bid_violations = m[m["C_bid"].notna() & (m["C_bid"] > m["S_ask"] + args.tol)]

        if not ask_violations.empty or not bid_violations.empty:
            report.append({
                "strike": K,
                "ask_below_intrinsic": len(ask_violations),
                "bid_above_stock": len(bid_violations),
                "first_ask_violation_ts": int(ask_violations["timestamp"].iloc[0]) if not ask_violations.empty else None,
                "first_bid_violation_ts": int(bid_violations["timestamp"].iloc[0]) if not bid_violations.empty else None,
            })

    if args.json:
        print(json.dumps(report, indent=2))
        return
    if not report:
        print(f"No no-arb violations found (tol={args.tol})")
        return
    print(f"No-arb violations · {args.round} day {args.day} · tol={args.tol}")
    for r in report:
        parts = []
        if r["ask_below_intrinsic"]:
            parts.append(f"ask<intr at {r['ask_below_intrinsic']} ts (first={r['first_ask_violation_ts']:,})")
        if r["bid_above_stock"]:
            parts.append(f"bid>S at {r['bid_above_stock']} ts (first={r['first_bid_violation_ts']:,})")
        print(f"  VEV_{r['strike']}: {'; '.join(parts)}")


def cmd_greeks(args):
    ve = _acts(args.round, opts.UNDERLYING, args.day)
    row = _nearest(ve.dropna(subset=["mid_price"]), args.ts)
    if row is None:
        print("(no underlying)")
        return
    S = float(row["mid_price"])
    actual_ts = int(row["timestamp"])
    tte_start = _tte(args.round, args.day, args.tte)
    T = max(tte_start - actual_ts / opts.TIMESTAMP_PER_DAY, 1e-6)

    strikes = args.strikes or opts.STRIKES
    out = []
    for K in strikes:
        v = _acts(args.round, f"VEV_{K}", args.day)
        v_row = _nearest(v.dropna(subset=["mid_price"]), args.ts)
        if v_row is None:
            continue
        C = float(v_row["mid_price"])
        iv = float(opts.implied_vol(np.array([C]), S, float(K), T)[0])
        if not np.isfinite(iv):
            # Use a default σ when IV is unsolvable so greeks still print
            iv_used = 0.015
            iv_label = "0.015 (fallback)"
        else:
            iv_used = iv
            iv_label = f"{iv:.5f}"
        g = opts.greeks(np.array([S]), np.array([float(K)]),
                        np.array([T]), np.array([iv_used]))
        out.append({
            "K": K, "C": C, "iv": iv_label,
            "delta": float(g["delta"][0]), "gamma": float(g["gamma"][0]),
            "theta": float(g["theta"][0]), "vega": float(g["vega"][0]),
        })

    if args.json:
        print(json.dumps({
            "S": S, "T": T, "ts_actual": actual_ts, "greeks": out,
        }, indent=2))
        return
    print(f"Greeks · {args.round} day {args.day} · t={actual_ts:,} · S={S:.2f} · T={T:.3f}d")
    print(f"  {'K':>6} {'C':>8} {'IV':>18} {'delta':>8} {'gamma':>10} "
          f"{'theta':>10} {'vega':>10}")
    for r in out:
        print(f"  {r['K']:>6} {r['C']:>8.2f} {r['iv']:>18} "
              f"{r['delta']:>8.4f} {r['gamma']:>10.6f} "
              f"{r['theta']:>10.3f} {r['vega']:>10.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        prog="tools/probe.py",
        description="Inspect dashboard data from the command line.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp, require_day=True):
        sp.add_argument("--round", default="ROUND_3", help="round folder (default: ROUND_3)")
        if require_day:
            sp.add_argument("--day", type=int, required=True, help="day (0/1/2)")
        else:
            sp.add_argument("--day", type=int, help="day (0/1/2); omit for all days")
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    sp = sub.add_parser("summary", help="per-product mid statistics")
    _common(sp, require_day=False)
    sp.set_defaults(fn=cmd_summary)

    sp = sub.add_parser("price", help="price / mid over a time range")
    _common(sp)
    sp.add_argument("--product", required=True)
    sp.add_argument("--range", help="t0:t1 timestamp range (e.g. 500000:600000)")
    sp.add_argument("--stats", action="store_true", help="print summary stats only")
    sp.add_argument("--sample", type=int, help="print N evenly-spaced rows")
    sp.set_defaults(fn=cmd_price)

    sp = sub.add_parser("book", help="order book snapshot at a timestamp")
    _common(sp)
    sp.add_argument("--product", required=True)
    sp.add_argument("--ts", type=int, required=True)
    sp.set_defaults(fn=cmd_book)

    sp = sub.add_parser("trades", help="filtered trade log")
    _common(sp)
    sp.add_argument("--product", required=True)
    sp.add_argument("--range", help="t0:t1 timestamp range")
    sp.add_argument("--min-qty", type=int, help="filter trades with qty >= min-qty")
    sp.set_defaults(fn=cmd_trades)

    sp = sub.add_parser("smile", help="IV by strike at a timestamp")
    _common(sp)
    sp.add_argument("--ts", type=int, required=True)
    sp.add_argument("--tte", type=float, help="override TTE@start (days)")
    sp.set_defaults(fn=cmd_smile)

    sp = sub.add_parser("iv", help="IV distribution per strike over a day")
    _common(sp)
    sp.add_argument("--strikes", type=int, nargs="+",
                    help="subset of strikes (default: all)")
    sp.add_argument("--tte", type=float, help="override TTE@start (days)")
    sp.set_defaults(fn=cmd_iv)

    sp = sub.add_parser("arb", help="find no-arb bound violations")
    _common(sp)
    sp.add_argument("--strikes", type=int, nargs="+")
    sp.add_argument("--tol", type=float, default=0.5,
                    help="ignore violations smaller than this (default: 0.5 tick)")
    sp.set_defaults(fn=cmd_arb)

    sp = sub.add_parser("greeks", help="greeks at a timestamp (uses market IV)")
    _common(sp)
    sp.add_argument("--ts", type=int, required=True)
    sp.add_argument("--strikes", type=int, nargs="+")
    sp.add_argument("--tte", type=float)
    sp.set_defaults(fn=cmd_greeks)

    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
