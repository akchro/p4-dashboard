"""Analyze per-strike IV residuals against a fitted smile.

For each timestamp we:
  1. Compute IV at each strike using mid / bid / ask voucher prices.
  2. Fit a quadratic smile  iv = a + b*m + c*m^2  where m = ln(K/S)/sqrt(T).
  3. Compute residuals (iv - smile) per strike.

Per strike we then summarize:
  - mean / std of residual
  - fraction of time above zero (above the smile fit)
  - cross-strike correlation
  - lag-1 autocorrelation (persistence)
  - half-spread cost in IV units (vega-implied)

Designed to surface signals in the implied-vol surface without committing
to a trading rule.
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


LIQUID_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
DEFAULT_FIT_STRIKES = LIQUID_STRIKES  # used to fit smile each tick


def _load_logfile_activities(path: str) -> pd.DataFrame:
    """Parse the embedded activitiesLog CSV from a JSON submission log."""
    import io
    raw = Path(path).read_text()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        first = raw.split("\n", 1)[0]
        obj = json.loads(first)
    csv_text = obj["activitiesLog"]
    df = pd.read_csv(io.StringIO(csv_text), sep=";")
    for col in historical_loader.NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_panel(activities: pd.DataFrame, day: int, tte_at_start: float) -> pd.DataFrame:
    """Return wide DataFrame: index=timestamp, columns=(prod, field) for
    underlying mid/bid/ask + each voucher mid/bid/ask."""
    df = activities[activities["day"] == day].copy()
    products = [opts.UNDERLYING] + [f"VEV_{K}" for K in opts.STRIKES]
    df = df[df["product"].isin(products)]
    keep = ["timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"]
    df = df[keep].rename(columns={
        "mid_price": "mid", "bid_price_1": "bid", "ask_price_1": "ask",
    })
    pivot = df.pivot(index="timestamp", columns="product",
                     values=["mid", "bid", "ask"])
    pivot.columns = [f"{prod}__{field}" for field, prod in pivot.columns]
    pivot = pivot.sort_index()
    pivot["T"] = np.maximum(tte_at_start - pivot.index.values / opts.TIMESTAMP_PER_DAY,
                            1e-6)
    return pivot


def _iv_grid(panel: pd.DataFrame, side: str = "mid") -> pd.DataFrame:
    """Compute IV for each strike at each timestamp using `side` voucher prices.
    Underlying always uses mid (cleaner; otherwise we'd be conflating sides)."""
    S = panel[f"{opts.UNDERLYING}__mid"].to_numpy()
    T = panel["T"].to_numpy()
    out = {}
    for K in opts.STRIKES:
        col = f"VEV_{K}__{side}"
        if col not in panel.columns:
            continue
        C = panel[col].to_numpy()
        iv = opts.implied_vol(C, S, float(K), T)
        out[K] = iv
    iv_df = pd.DataFrame(out, index=panel.index)
    return iv_df


def _moneyness_grid(panel: pd.DataFrame, strikes) -> pd.DataFrame:
    """Compute m = ln(K/S)/sqrt(T) for each strike at each timestamp."""
    S = panel[f"{opts.UNDERLYING}__mid"].to_numpy()
    T = panel["T"].to_numpy()
    out = {}
    for K in strikes:
        out[K] = np.log(float(K) / S) / np.sqrt(T)
    return pd.DataFrame(out, index=panel.index)


def _fit_quadratic(m_row: np.ndarray, iv_row: np.ndarray):
    """Fit iv = a + b*m + c*m^2 by OLS. Returns (a,b,c) or NaNs if <3 finite pts."""
    mask = np.isfinite(m_row) & np.isfinite(iv_row)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(mask.sum()), m_row[mask], m_row[mask] ** 2])
    y = iv_row[mask]
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    return beta[0], beta[1], beta[2]


def _compute_residuals(panel: pd.DataFrame, side: str, fit_strikes):
    """Per-timestamp quadratic smile residuals.

    Smile is fit each tick using `side` IVs at `fit_strikes`. Residuals are
    returned for ALL liquid strikes so we can see which ones the fit puts
    above/below the smile."""
    iv_df = _iv_grid(panel, side=side)
    m_df = _moneyness_grid(panel, opts.STRIKES)

    fit_iv = iv_df[fit_strikes].to_numpy()
    fit_m = m_df[fit_strikes].to_numpy()

    n_rows = fit_iv.shape[0]
    fits = np.zeros((n_rows, 3))
    for i in range(n_rows):
        fits[i] = _fit_quadratic(fit_m[i], fit_iv[i])

    a, b, c = fits[:, 0], fits[:, 1], fits[:, 2]

    resid = {}
    for K in iv_df.columns:
        m = m_df[K].to_numpy()
        smile = a + b * m + c * m * m
        resid[K] = iv_df[K].to_numpy() - smile
    return pd.DataFrame(resid, index=panel.index), fits, iv_df, m_df


def _vega_iv_unit(panel: pd.DataFrame, K: int, sigma_ref: float = 0.013):
    """How many IV units does a 1-tick price change correspond to at strike K?
    Returns Series indexed by timestamp."""
    S = panel[f"{opts.UNDERLYING}__mid"].to_numpy()
    T = panel["T"].to_numpy()
    g = opts.greeks(S, np.full_like(S, float(K)), T, np.full_like(S, sigma_ref))
    vega = g["vega"]
    # 1 tick price ↔ 1/vega IV change
    return pd.Series(np.where(vega > 1e-9, 1.0 / vega, np.nan), index=panel.index)


def cmd_summarize(args):
    """Per-strike residual summary for one day, mid/bid/ask."""
    rounds = historical_loader.load_round(args.round)
    activities = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _build_panel(activities, args.day, tte)

    rows = []
    for side in ("mid", "bid", "ask"):
        resid, _, _, _ = _compute_residuals(panel, side, DEFAULT_FIT_STRIKES)
        for K in opts.STRIKES:
            r = resid[K].dropna()
            if r.empty:
                continue
            rows.append({
                "side": side, "K": K, "n": int(len(r)),
                "mean": float(r.mean()), "std": float(r.std()),
                "p5": float(np.percentile(r, 5)),
                "median": float(np.median(r)),
                "p95": float(np.percentile(r, 95)),
                "frac_above_zero": float((r > 0).mean()),
                "abs_mean": float(np.mean(np.abs(r))),
            })

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"Smile residuals (iv - quad_fit) · day {args.day} · TTE@start={tte}d "
          f"· fit strikes={DEFAULT_FIT_STRIKES}")
    print(f"  {'side':<5} {'K':>5} {'n':>5} "
          f"{'mean(×1e-4)':>12} {'std(×1e-4)':>11} "
          f"{'p5':>8} {'med':>8} {'p95':>8} {'%>0':>6}")
    for r in rows:
        print(f"  {r['side']:<5} {r['K']:>5} {r['n']:>5} "
              f"{r['mean']*1e4:>12.3f} {r['std']*1e4:>11.3f} "
              f"{r['p5']*1e4:>8.2f} {r['median']*1e4:>8.2f} {r['p95']*1e4:>8.2f} "
              f"{r['frac_above_zero']*100:>5.1f}%")


def cmd_compare_sides(args):
    """For each strike, ask: 'is mid above smile?' vs 'is bid above mid-fit smile?'

    A real signal needs the BID-side IV (worst case for buying the option)
    to still sit above the MID-side smile fit elsewhere, or vice versa.
    """
    rounds = historical_loader.load_round(args.round)
    activities = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _build_panel(activities, args.day, tte)

    # Mid-side smile: best estimate of fair IV across strikes
    resid_mid, _, iv_mid, m_df = _compute_residuals(panel, "mid", DEFAULT_FIT_STRIKES)
    # Bid IV per strike
    iv_bid = _iv_grid(panel, "bid")
    iv_ask = _iv_grid(panel, "ask")

    # We want to know: at each timestamp, how much would the residual change if
    # we replaced the strike's mid-IV with its bid-IV (i.e. 'I sold here') or
    # ask-IV ('I bought here')? We use the mid smile as the fair-value reference.
    fits = []
    for col in DEFAULT_FIT_STRIKES:
        pass  # already inside resid_mid via fit
    # rebuild smile from fit by reusing residuals: smile_iv = mid_iv - resid_mid
    smile_iv = iv_mid - resid_mid

    rows = []
    for K in opts.STRIKES:
        if K not in iv_mid.columns:
            continue
        r_mid = (iv_mid[K] - smile_iv[K]).dropna()
        r_bid = (iv_bid[K] - smile_iv[K]).dropna()
        r_ask = (iv_ask[K] - smile_iv[K]).dropna()
        rows.append({
            "K": K,
            "n": int(len(r_mid)),
            "mid_resid_mean": float(r_mid.mean()) if len(r_mid) else float("nan"),
            "bid_resid_mean": float(r_bid.mean()) if len(r_bid) else float("nan"),
            "ask_resid_mean": float(r_ask.mean()) if len(r_ask) else float("nan"),
            "frac_bid_above_smile": float((r_bid > 0).mean()) if len(r_bid) else 0.0,
            "frac_ask_below_smile": float((r_ask < 0).mean()) if len(r_ask) else 0.0,
            # how much half-spread eats into mid-residual
            "half_spread_iv": float(((r_ask - r_bid) / 2).mean()) if len(r_ask) and len(r_bid) else float("nan"),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"Bid/mid/ask residual vs MID smile · day {args.day}")
    print(f"  K     n    mid_r    bid_r    ask_r   %bid>smile  %ask<smile  half_spr_iv")
    for r in rows:
        print(f"  {r['K']:<5} {r['n']:>5} "
              f"{r['mid_resid_mean']*1e4:>+8.3f} "
              f"{r['bid_resid_mean']*1e4:>+8.3f} "
              f"{r['ask_resid_mean']*1e4:>+8.3f}  "
              f"{r['frac_bid_above_smile']*100:>8.1f}%   "
              f"{r['frac_ask_below_smile']*100:>8.1f}%   "
              f"{r['half_spread_iv']*1e4:>+10.3f}")
    print("  (residuals × 1e-4, i.e. IV points)")


def cmd_corr(args):
    """Cross-strike correlation and lag-1 autocorrelation of mid-side residuals."""
    rounds = historical_loader.load_round(args.round)
    activities = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _build_panel(activities, args.day, tte)
    resid, _, _, _ = _compute_residuals(panel, "mid", DEFAULT_FIT_STRIKES)

    # restrict to liquid strikes that solve for IV most of the time
    cols = [K for K in LIQUID_STRIKES if K in resid.columns]
    R = resid[cols].dropna(how="any")

    corr = R.corr()
    ac1 = {K: float(R[K].autocorr(lag=1)) for K in cols}
    # Half-life for AR(1): -ln(2)/ln(rho)  (when rho positive)
    halflife = {}
    for K, rho in ac1.items():
        if 0 < rho < 1:
            halflife[K] = float(-np.log(2) / np.log(rho))
        else:
            halflife[K] = None

    if args.json:
        print(json.dumps({
            "corr": corr.to_dict(),
            "autocorr_lag1": ac1,
            "halflife_ticks": halflife,
        }, indent=2))
        return
    print(f"Cross-strike residual correlation · day {args.day} · "
          f"n={len(R)} clean rows · fit strikes={DEFAULT_FIT_STRIKES}")
    print("       " + "  ".join(f"{K:>6}" for K in cols))
    for K in cols:
        row = "  ".join(f"{corr.loc[K, K2]:>+6.3f}" for K2 in cols)
        print(f"  {K:>5}  {row}")
    print()
    print("Lag-1 autocorrelation and half-life (in 100-tick units):")
    print(f"  {'K':>5}  {'rho1':>8}  {'half-life':>10}")
    for K in cols:
        hl = halflife[K]
        hl_s = f"{hl:.1f}" if hl is not None else "∞ (no decay)"
        print(f"  {K:>5}  {ac1[K]:>+8.4f}  {hl_s:>10}")


def cmd_timeseries(args):
    """Dump residual time-series for given strikes to CSV (useful for plotting)."""
    rounds = historical_loader.load_round(args.round)
    activities = rounds["activities"]
    tte = opts.tte_for_day(args.round, args.day)
    panel = _build_panel(activities, args.day, tte)
    resid, _, iv_df, m_df = _compute_residuals(panel, args.side, DEFAULT_FIT_STRIKES)

    strikes = args.strikes or LIQUID_STRIKES
    cols = [K for K in strikes if K in resid.columns]
    out = resid[cols].copy()
    out["S"] = panel[f"{opts.UNDERLYING}__mid"].values
    out["T"] = panel["T"].values
    if args.csv:
        out.to_csv(args.csv)
        print(f"wrote {len(out)} rows to {args.csv}")
    else:
        print(out.tail(args.tail).to_string())


def cmd_log(args):
    """Run summarize on a backtest/practice log file's day."""
    activities = _load_logfile_activities(args.path)
    days = sorted(activities["day"].unique())
    print(f"Days in log: {days}")
    for day in days:
        # Round 3 live started at TTE=5, but logs may include earlier days.
        # Best heuristic: assume each day in a multi-day log is consecutive
        # working back from 5d. We let the user override.
        if args.tte_at_start_day0 is not None:
            tte = args.tte_at_start_day0 - day
        elif len(days) > 1:
            # If multiple days, treat day 0 in the log = TTE 5 (live start)
            # and decrement.  Adjust if this is wrong for your log.
            tte = 5.0 - (day - days[0])
        else:
            tte = 5.0  # single live day → TTE=5
        if tte <= 0:
            continue
        panel = _build_panel(activities, day, tte)
        if panel.empty:
            continue
        for side in ("mid", "bid", "ask"):
            resid, _, _, _ = _compute_residuals(panel, side, DEFAULT_FIT_STRIKES)
            print(f"\nLog day={day} TTE={tte:.1f} side={side}")
            print(f"  K     n    mean(×1e-4)  std(×1e-4)   %>0")
            for K in opts.STRIKES:
                if K not in resid.columns:
                    continue
                r = resid[K].dropna()
                if r.empty:
                    continue
                print(f"  {K:<5} {len(r):>5} "
                      f"{r.mean()*1e4:>+11.3f}  {r.std()*1e4:>10.3f}  "
                      f"{(r>0).mean()*100:>5.1f}%")


def _build_parser():
    p = argparse.ArgumentParser(description="Smile-residual diagnostics.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp, require_day=True):
        sp.add_argument("--round", default="ROUND_3")
        if require_day:
            sp.add_argument("--day", type=int, required=True)
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("summarize", help="per-strike residual summary")
    _common(sp)
    sp.set_defaults(fn=cmd_summarize)

    sp = sub.add_parser("sides", help="bid vs mid vs ask residuals vs mid smile")
    _common(sp)
    sp.set_defaults(fn=cmd_compare_sides)

    sp = sub.add_parser("corr", help="cross-strike + autocorrelation")
    _common(sp)
    sp.set_defaults(fn=cmd_corr)

    sp = sub.add_parser("timeseries", help="dump residual time-series (CSV optional)")
    _common(sp)
    sp.add_argument("--side", default="mid", choices=("mid", "bid", "ask"))
    sp.add_argument("--strikes", type=int, nargs="+")
    sp.add_argument("--csv", help="write to CSV instead of printing")
    sp.add_argument("--tail", type=int, default=20)
    sp.set_defaults(fn=cmd_timeseries)

    sp = sub.add_parser("log", help="analyze a JSON submission log file")
    sp.add_argument("path")
    sp.add_argument("--tte-at-start-day0", type=float,
                    help="override TTE at start of day 0 in the log")
    sp.set_defaults(fn=cmd_log)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
