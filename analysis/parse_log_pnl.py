"""Parse a Prosperity submission log into a per-product PnL frame and plot.

PnL column is the last field of the activitiesLog CSV. We just want the
final-row PnL per (day, product), and the cumulative PnL trace per product.
"""
import json
import sys
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path) as f:
        obj = json.load(f)
    df = pd.read_csv(StringIO(obj["activitiesLog"]), sep=";")
    return df


def main(path):
    df = load(path)
    print(f"rows={len(df)} | days={sorted(df['day'].unique())} | "
          f"products={sorted(df['product'].unique())}")

    # Per-day final PnL per product
    last = (df.sort_values(["day", "timestamp"])
              .groupby(["day", "product"]).tail(1)
              .pivot(index="day", columns="product", values="profit_and_loss"))
    print("\nFinal PnL per day per product:")
    with pd.option_context("display.float_format", "{:>10.0f}".format,
                           "display.max_columns", None, "display.width", 200):
        print(last.to_string())

    print("\nTotals across days (sum):")
    totals = last.sum().sort_values(ascending=False)
    with pd.option_context("display.float_format", "{:>10.0f}".format):
        print(totals.to_string())
    print(f"\nGrand total: {totals.sum():.0f}")

    # Plot cumulative PnL per product, with day boundaries.
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    # global tick = day*1e6 + ts so ordering is correct across stitched days
    df["g"] = df["day"] * 1_000_000 + df["timestamp"]

    products = sorted(df["product"].unique())
    voucher = [p for p in products if p.startswith("VEV_")]
    underlying = [p for p in products if p in ("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for p in underlying:
        sub = df[df["product"] == p]
        axes[0].plot(sub["g"].values, sub["profit_and_loss"].values, label=p, lw=0.8)
    axes[0].set_title("PnL — underlying & HYD")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("PnL")

    cmap = plt.cm.viridis
    for i, p in enumerate(voucher):
        sub = df[df["product"] == p]
        axes[1].plot(sub["g"].values, sub["profit_and_loss"].values,
                     label=p, lw=0.8, color=cmap(i / max(1, len(voucher) - 1)))
    axes[1].set_title("PnL — vouchers")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    axes[1].set_ylabel("PnL")
    axes[1].set_xlabel("global tick (day*1e6 + ts)")

    # day separators
    for d in sorted(df["day"].unique())[1:]:
        for ax in axes:
            ax.axvline(d * 1_000_000, color="grey", lw=0.5, ls="--")

    fig.tight_layout()
    out_path = OUT / "log_pnl_by_product.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n-> {out_path}")

    # Also a side-by-side: each voucher vs underlying on its own axis pair
    n = len(voucher)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    vfe = df[df["product"] == "VELVETFRUIT_EXTRACT"]
    for ax, p in zip(axes, voucher):
        sub = df[df["product"] == p]
        ax.plot(sub["g"].values, sub["profit_and_loss"].values, color="C0", lw=0.7, label=p)
        ax2 = ax.twinx()
        ax2.plot(vfe["g"].values, vfe["profit_and_loss"].values, color="C3", lw=0.4, alpha=0.5, label="VFE")
        ax.set_ylabel(p, fontsize=8)
        ax.axhline(0, color="grey", lw=0.3)
        for d in sorted(df["day"].unique())[1:]:
            ax.axvline(d * 1_000_000, color="grey", lw=0.3, ls="--")
    fig.suptitle("Each voucher (blue) vs VELVETFRUIT_EXTRACT (red, twin axis)")
    fig.tight_layout()
    out2 = OUT / "log_voucher_vs_vfe.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"-> {out2}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/2026-04-25_16-59-47.log")
