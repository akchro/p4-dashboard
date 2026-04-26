"""Analyze the probing-trader log 403865.log.

The probing trader buys 5 of every voucher at ts=99000 and lets them sit until
the run ends at ts=99900. We reverse-engineer the effective liquidation price
from PnL, then compare it against:
  - the visible market mid at the last tick,
  - the Black-Scholes price using per-strike IV at the last tick,
  - the Black-Scholes price using realized vol of the underlying.

Outputs a table to stdout and a PNG chart to docs/probe_liquidation.png.
"""
import json
import csv
import io
import sys
from pathlib import Path
import collections

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils import options as opts


LOG_PATH = REPO / "logs" / "403865.log"
DAY = 2  # log header reports day=2
QTY = 5  # qty bought per voucher
BUY_TS = 99000
LAST_TS = 99900


def load_activities(log_path):
    with open(log_path) as f:
        data = json.load(f)
    rows = list(csv.DictReader(io.StringIO(data["activitiesLog"]), delimiter=";"))
    by_prod = collections.defaultdict(list)
    for r in rows:
        by_prod[r["product"]].append({
            "ts": int(r["timestamp"]),
            "mid": float(r["mid_price"]) if r["mid_price"] else float("nan"),
            "bid": float(r["bid_price_1"]) if r["bid_price_1"] else float("nan"),
            "ask": float(r["ask_price_1"]) if r["ask_price_1"] else float("nan"),
            "pnl": float(r["profit_and_loss"]) if r["profit_and_loss"] else 0.0,
        })
    for p in by_prod:
        by_prod[p].sort(key=lambda x: x["ts"])
    return data, by_prod


def submission_buys(trade_history, ts):
    out = {}
    for t in trade_history:
        if t.get("buyer") == "SUBMISSION" and t["timestamp"] == ts:
            out[t["symbol"]] = {"price": float(t["price"]), "qty": int(t["quantity"])}
    return out


def realized_vol_per_sqrt_day(mid_series, interval_ticks=10):
    """Realized σ per √day from log returns sampled every k * 100-ts ticks.

    Returns (sigma_obs, sigma_roll) — raw and Roll-bounce-corrected.
    """
    s = np.asarray(mid_series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < interval_ticks * 5:
        return float("nan"), float("nan")
    log_s = np.log(s[::interval_ticks])
    r = np.diff(log_s)
    if len(r) < 5:
        return float("nan"), float("nan")
    var_per = r.var(ddof=1)
    n_per_day = 10000 / interval_ticks
    sigma_obs = np.sqrt(var_per * n_per_day)
    cov1 = np.cov(r[:-1], r[1:], ddof=1)[0, 1]
    adj_var = max(var_per - 2 * abs(min(cov1, 0)), 1e-12)
    sigma_roll = np.sqrt(adj_var * n_per_day)
    return sigma_obs, sigma_roll


def main():
    data, by_prod = load_activities(LOG_PATH)

    buys = submission_buys(data["tradeHistory"], BUY_TS)
    if not buys:
        raise SystemExit(f"No SUBMISSION buys at ts={BUY_TS}")

    underlying_rows = by_prod[opts.UNDERLYING]
    last_under = next(r for r in underlying_rows if r["ts"] == LAST_TS)
    S_last = last_under["mid"]

    tte_start = opts.tte_for_day("ROUND_3", DAY)
    T_last = float(opts.time_to_expiry([LAST_TS], tte_start)[0])

    mid_series = [r["mid"] for r in underlying_rows]
    sigma_obs, sigma_roll = realized_vol_per_sqrt_day(mid_series, interval_ticks=10)
    sigma_obs_k1, sigma_roll_k1 = realized_vol_per_sqrt_day(mid_series, interval_ticks=1)
    # Use the k=1 Roll-corrected estimate as our "best" realized σ. It strips
    # bid-ask bounce inflation that contaminates raw tick-level σ.
    sigma_rv = sigma_roll_k1

    # First pass: per-strike IV at the last tick.
    pre = []
    for K in opts.STRIKES:
        prod = f"VEV_{K}"
        last = next(r for r in by_prod[prod] if r["ts"] == LAST_TS)
        iv = float(opts.implied_vol(
            np.array([last["mid"]]), S_last, float(K), T_last)[0])
        pre.append((K, last, iv))

    # Pick a flat-IV reference: median of the well-defined ATM IVs.
    atm_ivs = [iv for K, _, iv in pre if 5000 <= K <= 5500 and np.isfinite(iv)]
    sigma_iv_flat = float(np.median(atm_ivs)) if atm_ivs else float("nan")

    rows = []
    for K, last, iv in pre:
        prod = f"VEV_{K}"
        buy = buys.get(prod, {})
        cost = buy.get("price", float("nan"))
        mid_last = last["mid"]
        pnl_last = last["pnl"]
        liq_implied = cost + pnl_last / QTY

        bs_iv_perK = float(opts.bs_call(S_last, float(K), T_last, iv)) if np.isfinite(iv) else float("nan")
        bs_iv_flat = float(opts.bs_call(S_last, float(K), T_last, sigma_iv_flat))
        bs_rv = float(opts.bs_call(S_last, float(K), T_last, sigma_rv))
        bs_rv_raw = float(opts.bs_call(S_last, float(K), T_last, sigma_obs_k1))

        rows.append({
            "K": K,
            "cost": cost,
            "liq": liq_implied,
            "mid": mid_last,
            "bid": last["bid"],
            "ask": last["ask"],
            "pnl": pnl_last,
            "iv": iv,
            "bs_iv_perK": bs_iv_perK,
            "bs_iv_flat": bs_iv_flat,
            "bs_rv": bs_rv,
            "bs_rv_raw": bs_rv_raw,
        })

    # Print table
    print(f"\n=== Probe analysis · log 403865.log · day {DAY} ===")
    print(f"  Underlying S at ts={LAST_TS}: {S_last:.2f}")
    print(f"  TTE at liquidation: {T_last:.4f} days")
    print(f"  Realized σ (k= 1·100ts, raw)  : {sigma_obs_k1*100:.4f} %/√day")
    print(f"  Realized σ (k= 1·100ts, Roll) : {sigma_roll_k1*100:.4f} %/√day  ← used for BS-RV")
    print(f"  Realized σ (k=10·100ts, raw)  : {sigma_obs*100:.4f} %/√day")
    print(f"  Realized σ (k=10·100ts, Roll) : {sigma_roll*100:.4f} %/√day")
    print(f"  Flat ATM IV (median K∈[5000,5500]): {sigma_iv_flat*100:.4f} %/√day")
    print()

    print(f"  {'K':>5} {'buy':>5} {'liq':>8} {'mid':>7} {'bid':>4} {'ask':>4} "
          f"{'pnl/5':>9} {'IV%':>6} {'BS-IV(K)':>9} {'BS-IV(flat)':>11} "
          f"{'BS-RV':>8} {'BS-RV(raw)':>11} {'liq-mid':>8} {'mid-BS-RV':>10}")
    for r in rows:
        iv_str = f"{r['iv']*100:.3f}" if np.isfinite(r["iv"]) else "n/a"
        bs_iv_str = f"{r['bs_iv_perK']:.3f}" if np.isfinite(r["bs_iv_perK"]) else "n/a"
        print(f"  {r['K']:>5} {r['cost']:>5.0f} {r['liq']:>8.3f} {r['mid']:>7.2f} "
              f"{r['bid']:>4.0f} {r['ask']:>4.0f} {r['pnl']:>9.3f} "
              f"{iv_str:>6} {bs_iv_str:>9} {r['bs_iv_flat']:>11.3f} "
              f"{r['bs_rv']:>8.3f} {r['bs_rv_raw']:>11.3f} "
              f"{r['liq']-r['mid']:>+8.3f} {r['mid']-r['bs_rv']:>+10.3f}")

    # Markdown table
    md_path = REPO / "docs" / "probe_403865_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Probe trader · log 403865 · liquidation prices\n\n")
        f.write(f"- Day {DAY}; trader bought 5 of each voucher at ts={BUY_TS}; run ended at ts={LAST_TS}.\n")
        f.write(f"- Underlying S = {S_last:.2f} at liquidation; TTE = {T_last:.4f} days.\n")
        f.write(f"- **Realized σ** on VELVETFRUIT_EXTRACT, Roll-bounce-corrected at k=1: "
                f"{sigma_roll_k1*100:.4f} %/√day (raw {sigma_obs_k1*100:.4f}).\n")
        f.write(f"- **Flat ATM IV** (median IV across K∈[5000,5500]): "
                f"{sigma_iv_flat*100:.4f} %/√day.\n\n")
        f.write("**liq** = effective liquidation price = `cost + PnL/qty`. "
                "**BS-IV(K)** prices each strike at its own IV (so it equals mid by "
                "construction — sanity check). **BS-IV(flat)** prices every strike at the "
                "ATM IV — shows what the *no-smile* model says. **BS-RV** uses the "
                "Roll-corrected realized σ (one number, applied flat).\n\n")
        f.write("| K | buy | liq | mid | bid | ask | PnL/5 | IV % | BS-IV(K) | BS-IV(flat) | BS-RV | BS-RV(raw) | liq − mid | mid − BS-RV |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            iv_str = f"{r['iv']*100:.3f}" if np.isfinite(r["iv"]) else "—"
            bs_iv_str = f"{r['bs_iv_perK']:.3f}" if np.isfinite(r["bs_iv_perK"]) else "—"
            f.write(f"| {r['K']} | {r['cost']:.0f} | {r['liq']:.3f} | {r['mid']:.2f} | "
                    f"{r['bid']:.0f} | {r['ask']:.0f} | {r['pnl']:.3f} | "
                    f"{iv_str} | {bs_iv_str} | {r['bs_iv_flat']:.3f} | "
                    f"{r['bs_rv']:.3f} | {r['bs_rv_raw']:.3f} | "
                    f"{r['liq']-r['mid']:+.3f} | {r['mid']-r['bs_rv']:+.3f} |\n")
        f.write("\n## What to notice\n\n")
        f.write(f"- For all ITM/ATM strikes, **liq ≈ mid** (within ±0.2 ticks), so the "
                f"engine settles at mid.\n")
        f.write(f"- For **VEV_6000 and VEV_6500** (price floor: bid=0, ask=1), liq = 0.0, "
                f"NOT the mid of 0.5. The engine effectively closes floor-pinned positions "
                f"at the bid. Buying these as 'lottery tickets' loses the full premium.\n")
        f.write(f"- **BS-IV(K) = mid** by construction; included as numerical sanity check.\n")
        f.write(f"- **BS-IV(flat) ≠ mid** at OTM wings — that's the smile. ATM IV understates "
                f"the wing IVs the market is actually quoting.\n")
        f.write(f"- **BS-RV >> mid** for ATM strikes because the per-tick realized σ "
                f"(~{sigma_roll_k1*100:.2f}%/√day) is still inflated by bid-ask bounce — even "
                f"after Roll correction. CLAUDE.md notes the 'true' σ from longer windows is "
                f"closer to ~1.3%/√day. The market IV (~{sigma_iv_flat*100:.2f}%) is the more "
                f"trustworthy vol estimate here.\n")
    print(f"\n  Wrote {md_path.relative_to(REPO)}")

    # ---- chart ----
    Ks = np.array([r["K"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: full strike range, log scale
    ax = axes[0, 0]
    ax.plot(Ks, [r["cost"] for r in rows], marker="^", color="tab:gray",
            linestyle=":", label="buy (cost)")
    ax.plot(Ks, [r["liq"] for r in rows], marker="o", color="tab:red",
            label="liquidation")
    ax.plot(Ks, [r["mid"] for r in rows], marker="s", color="tab:blue",
            alpha=0.7, label=f"mid (ts={LAST_TS})")
    ax.plot(Ks, [r["bs_iv_flat"] for r in rows], marker="P", color="tab:green",
            label=f"BS · flat ATM IV ({sigma_iv_flat*100:.2f}%)")
    ax.plot(Ks, [r["bs_rv"] for r in rows], marker="d", color="tab:orange",
            label=f"BS · realized σ ({sigma_roll_k1*100:.2f}% Roll)")
    ax.set_yscale("log")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Price (log)")
    ax.set_title("All strikes (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

    # Top-right: ATM/OTM zoom on linear scale (K=5000..5500)
    ax = axes[0, 1]
    mask = (Ks >= 5000) & (Ks <= 5500)
    Ksz = Ks[mask]; idx = np.where(mask)[0]
    ax.plot(Ksz, [rows[i]["cost"] for i in idx], marker="^", color="tab:gray",
            linestyle=":", label="buy (cost)")
    ax.plot(Ksz, [rows[i]["liq"] for i in idx], marker="o", color="tab:red",
            label="liquidation")
    ax.plot(Ksz, [rows[i]["mid"] for i in idx], marker="s", color="tab:blue",
            alpha=0.7, label="mid")
    ax.plot(Ksz, [rows[i]["bs_iv_flat"] for i in idx], marker="P",
            color="tab:green", label="BS · flat ATM IV")
    ax.plot(Ksz, [rows[i]["bs_rv"] for i in idx], marker="d",
            color="tab:orange", label="BS · realized σ")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Price")
    ax.set_title("ATM zoom (linear, K=5000–5500)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Bottom-left: liq − mid (the "execution slippage")
    ax = axes[1, 0]
    width = 80
    ax.bar(Ks - width/2, [r["liq"] - r["mid"] for r in rows], width=width,
           color="tab:red", alpha=0.8, label="liq − mid")
    ax.bar(Ks + width/2, [r["liq"] - r["cost"] for r in rows], width=width,
           color="tab:gray", alpha=0.8, label="liq − cost (=PnL/5)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Strike K"); ax.set_ylabel("Δ price")
    ax.set_title("Execution: where the system marked you out")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Bottom-right: model gap (mid − BS prices)
    ax = axes[1, 1]
    ax.bar(Ks - width/2, [r["mid"] - r["bs_iv_flat"] for r in rows], width=width,
           color="tab:green", alpha=0.8, label="mid − BS-IV(flat)")
    ax.bar(Ks + width/2, [r["mid"] - r["bs_rv"] for r in rows], width=width,
           color="tab:orange", alpha=0.8, label="mid − BS-RV")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Strike K"); ax.set_ylabel("Δ price")
    ax.set_title("Model gap: market vs flat-vol BS")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Probe-trader log 403865 · day {DAY} · S={S_last:.2f} · T={T_last:.3f}d "
                 f"· bought 5 of each at ts={BUY_TS}, exited at ts={LAST_TS}",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    img_path = REPO / "docs" / "probe_403865_liquidation.png"
    plt.savefig(img_path, dpi=130)
    print(f"  Wrote {img_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
