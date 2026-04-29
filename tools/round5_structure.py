"""Round 5 structural-relationship probes.

Seven analyses for finding hidden linear / rank / lagged constraints among the
50 round-5 products. Each analysis writes a figure to analysis/out/ and prints
top findings to stdout. Run all with:

    python3 tools/round5_structure.py

Or limit to a subset:

    python3 tools/round5_structure.py --tasks t1 t2
    python3 tools/round5_structure.py --tasks t7 --top 100
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROUND_DIR = "historical/ROUND_5"
OUT_DIR = "analysis/out"
DAYS = (2, 3, 4)
TIMESTAMP_PER_DAY = 1_000_000

CATEGORIES: dict[str, list[str]] = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ],
    "PANEL": [
        "PANEL_1X2",
        "PANEL_1X4",
        "PANEL_2X2",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "PEBBLES": [
        "PEBBLES_L",
        "PEBBLES_M",
        "PEBBLES_S",
        "PEBBLES_XL",
        "PEBBLES_XS",
    ],
    "ROBOT": [
        "ROBOT_DISHES",
        "ROBOT_IRONING",
        "ROBOT_LAUNDRY",
        "ROBOT_MOPPING",
        "ROBOT_VACUUMING",
    ],
    "SLEEP_POD": [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_SUEDE",
    ],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ],
    "TRANSLATOR": [
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ],
    "UV_VISOR": [
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_YELLOW",
    ],
}

ALL_PRODUCTS = [p for plist in CATEGORIES.values() for p in plist]
P2CAT = {p: c for c, plist in CATEGORIES.items() for p in plist}
SHORT = {p: p.replace(c + "_", "") for c, plist in CATEGORIES.items() for p in plist}

BETAS = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)
LAGS = (0, 10, 50, 100, 200)


# ---------- Loading ------------------------------------------------------- #


def load_wide() -> pd.DataFrame:
    """Wide DataFrame: index=t, columns=product, values=mid_price."""
    dfs = []
    for d in DAYS:
        f = os.path.join(ROUND_DIR, f"prices_round_5_day_{d}.csv")
        df = pd.read_csv(f, sep=";", usecols=["day", "timestamp", "product", "mid_price"])
        df["t"] = (df["day"] - DAYS[0]) * TIMESTAMP_PER_DAY + df["timestamp"]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    wide = df.pivot_table(index="t", columns="product", values="mid_price").sort_index()
    # Ensure column order matches ALL_PRODUCTS
    wide = wide[ALL_PRODUCTS]
    return wide


def day_index(t: pd.Index) -> np.ndarray:
    """Return day-id per timestamp (0,1,2 for the 3 input days)."""
    return (t.values // TIMESTAMP_PER_DAY).astype(int)


# ---------- T1: β-grid stationarity heatmap ------------------------------- #


def t1_beta_heatmap(wide: pd.DataFrame, top: int = 30):
    """For each ordered pair (i,j) and β ∈ BETAS, compute std(P_i + β P_j)
    after removing per-day means (so day-to-day level shifts don't dominate).
    Cell = min over β of std(spread) / std(P_i). Low = cointegrated.
    """
    n = len(ALL_PRODUCTS)
    X = wide.values.astype(float)  # (T, n)
    days = day_index(wide.index)

    # Per-day demean each product so a single residual sigma per product is comparable
    Xd = X.copy()
    sigma = np.zeros(n)
    for p_idx in range(n):
        col = X[:, p_idx]
        for d_id in np.unique(days):
            mask = days == d_id
            col_d = col[mask]
            col[mask] = col_d - np.nanmean(col_d)
        Xd[:, p_idx] = col
        sigma[p_idx] = np.nanstd(col)

    R = np.full((n, n), np.nan)
    Bbest = np.full((n, n), np.nan)
    for i in range(n):
        xi = Xd[:, i]
        for j in range(n):
            if i == j:
                continue
            xj = Xd[:, j]
            best_r = np.inf
            best_b = np.nan
            for b in BETAS:
                s = np.nanstd(xi + b * xj)
                if sigma[i] == 0:
                    continue
                r = s / sigma[i]
                if r < best_r:
                    best_r = r
                    best_b = b
            R[i, j] = best_r
            Bbest[i, j] = best_b

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    im0 = axes[0].imshow(R, cmap="viridis", vmin=0, vmax=1.2)
    axes[0].set_title("T1 — min_β std(P_i + β·P_j) / std(P_i)\n(low = cointegrated)")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(ALL_PRODUCTS, fontsize=5)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Bbest, cmap="coolwarm", vmin=-2, vmax=2)
    axes[1].set_title("T1 — argmin β")
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(ALL_PRODUCTS, fontsize=5)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t1_beta_heatmap.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[T1] saved {out}")

    # Top hits
    flat = []
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(R[i, j]):
                continue
            flat.append((R[i, j], Bbest[i, j], ALL_PRODUCTS[i], ALL_PRODUCTS[j]))
    flat.sort()
    print(f"\n[T1] Top {top} cointegration candidates (P_i + β·P_j, demeaned per-day):")
    print(f"{'ratio':>6}  {'β':>5}  same_cat  {'P_i':<28}  {'P_j':<28}")
    for ratio, b, pi, pj in flat[:top]:
        same = "Y" if P2CAT[pi] == P2CAT[pj] else "."
        print(f"{ratio:6.3f}  {b:+5.1f}     {same}      {pi:<28}  {pj:<28}")
    return {"R": R, "B": Bbest, "top": flat[:top]}


# ---------- T2: SVD per category ----------------------------------------- #


def t2_svd_per_category(wide: pd.DataFrame):
    """Run SVD on each category's centered price matrix. Plot all 5 singular
    values + the eigenvector at each rank. The smallest singular value's
    eigenvector reveals the most stable linear combination.
    """
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(n_cats, 6, figsize=(22, 2.4 * n_cats))
    summary = {}
    for k, (cat, plist) in enumerate(CATEGORIES.items()):
        sub = wide[plist].dropna()
        # Per-day demean to remove level shifts between days
        sub_d = sub.copy()
        for d_id in np.unique(day_index(sub.index)):
            mask = day_index(sub.index) == d_id
            sub_d.loc[sub.index[mask]] = (
                sub.loc[sub.index[mask]] - sub.loc[sub.index[mask]].mean()
            )
        Xc = sub_d.values
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Singular values
        ax = axes[k, 0]
        ax.bar(range(5), s)
        ax.set_yscale("log")
        ax.set_title(f"{cat}\nσ values", fontsize=9)
        ax.set_xticks(range(5))

        # Plot all 5 eigenvectors (ordered: largest σ → smallest σ)
        labels = [SHORT[p] for p in plist]
        for r in range(5):
            ax = axes[k, r + 1]
            v = Vt[r]
            colors = ["steelblue" if x >= 0 else "indianred" for x in v]
            ax.bar(range(5), v, color=colors)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(5))
            ax.set_xticklabels(labels, rotation=60, fontsize=6)
            ax.set_title(f"v_{r+1}  σ={s[r]:.1f}", fontsize=8)

        smallest_v = Vt[-1]
        # Project: A_t = X @ v_min should be near constant
        proj = (Xc @ smallest_v)
        summary[cat] = {
            "singular_values": [float(x) for x in s],
            "smallest_eigvec": [float(x) for x in smallest_v],
            "products": plist,
            "proj_std": float(np.std(proj)),
            "ratio_smallest_to_largest": float(s[-1] / s[0]),
        }
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t2_svd.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[T2] saved {out}")
    print(f"\n[T2] Per-category SVD summary (smallest singular value's eigenvector = stable basket):")
    print(f"{'category':<14} {'σ_1':>9} {'σ_2':>9} {'σ_3':>9} {'σ_4':>9} {'σ_5':>9}  σ5/σ1   smallest_eigvec")
    for cat, info in summary.items():
        s = info["singular_values"]
        v = info["smallest_eigvec"]
        vstr = "  ".join(f"{SHORT[p][:3]}={v[i]:+.3f}" for i, p in enumerate(info["products"]))
        print(
            f"{cat:<14} {s[0]:9.1f} {s[1]:9.1f} {s[2]:9.1f} {s[3]:9.1f} {s[4]:9.1f}  "
            f"{info['ratio_smallest_to_largest']:6.4f}  {vstr}"
        )
    return summary


# ---------- T3: Rank invariants ------------------------------------------ #


def t3_rank_invariants(wide: pd.DataFrame, top: int = 6):
    """For each category, sort the 5 products at each tick. Distribution of
    rank orderings reveals strict hierarchies + their rare reversals.
    """
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.ravel()
    summary = {}
    print(f"\n[T3] Rank-order frequencies per category (top {top}):")
    for k, (cat, plist) in enumerate(CATEGORIES.items()):
        sub = wide[plist].dropna()
        # ranks[t,i] = rank of product i at time t (0=lowest, 4=highest)
        ranks = np.argsort(np.argsort(sub.values, axis=1), axis=1)
        # Encode each row as a tuple — count
        rows = [tuple(r) for r in ranks]
        c = Counter(rows)
        total = sum(c.values())
        most = c.most_common(top)

        # Bar chart of top orderings + their share
        ax = axes[k]
        share = [v / total for _, v in most]
        labels = []
        for ord_tuple, _ in most:
            # Reverse rank to recover ascending order of products
            order = np.argsort(ord_tuple)
            labels.append(" < ".join(SHORT[plist[i]][:4] for i in order))
        ax.barh(range(len(most)), share, color="steelblue")
        ax.set_yticks(range(len(most)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{cat}  (n_unique={len(c)})", fontsize=9)
        ax.set_xlabel("share of ticks")

        summary[cat] = {
            "n_unique": len(c),
            "dominant": most[0],
            "dominant_share": most[0][1] / total,
            "top_k": most,
        }
        # Print
        print(f"\n  --- {cat}  (n_unique_orders={len(c)}, total={total}) ---")
        for ord_tuple, count in most:
            order = np.argsort(ord_tuple)
            label = " < ".join(SHORT[plist[i]] for i in order)
            print(f"    {count/total:6.3%}  {label}")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t3_rank_invariants.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[T3] saved {out}")
    return summary


# ---------- T4: Min/max envelope ----------------------------------------- #


def t4_envelope(wide: pd.DataFrame):
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(n_cats, 1, figsize=(16, 2.0 * n_cats), sharex=True)
    summary = {}
    print(f"\n[T4] Per-category envelope diagnostics:")
    print(f"{'category':<14} {'rng_mean':>9} {'rng_std':>8} {'mean_std':>9} {'asym':>7}  {'individual_std_med':>18}")
    for k, (cat, plist) in enumerate(CATEGORIES.items()):
        sub = wide[plist].dropna()
        max_t = sub.max(axis=1)
        min_t = sub.min(axis=1)
        mean_t = sub.mean(axis=1)
        rng = max_t - min_t
        upper = max_t - mean_t
        lower = mean_t - min_t

        ax = axes[k]
        idx = np.arange(len(sub))
        ax.plot(idx, rng.values, lw=0.4, label="max - min", color="C0")
        ax.plot(idx, upper.values, lw=0.4, label="max - mean", color="C1", alpha=0.7)
        ax.plot(idx, lower.values, lw=0.4, label="mean - min", color="C2", alpha=0.7)
        ax.set_title(f"{cat}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        # Mark day boundaries
        days = day_index(sub.index)
        for boundary in np.where(np.diff(days) != 0)[0]:
            ax.axvline(boundary, color="k", lw=0.4, alpha=0.4)

        ind_std = float(np.median([sub[p].std() for p in plist]))
        summary[cat] = {
            "range_mean": float(rng.mean()),
            "range_std": float(rng.std()),
            "mean_std": float(mean_t.std()),
            "asymmetry": float(upper.mean() - lower.mean()),
            "individual_std_median": ind_std,
        }
        print(
            f"{cat:<14} {rng.mean():9.2f} {rng.std():8.2f} {mean_t.std():9.2f} "
            f"{upper.mean()-lower.mean():+7.2f}  {ind_std:18.2f}"
        )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t4_envelope.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[T4] saved {out}")
    return summary


# ---------- T5: Ratio invariants ----------------------------------------- #


def t5_ratio_invariants(wide: pd.DataFrame, top: int = 30):
    n = len(ALL_PRODUCTS)
    X = wide.values.astype(float)
    CV = np.full((n, n), np.nan)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    for i in range(n):
        if means[i] == 0:
            continue
        for j in range(n):
            if i == j or means[j] == 0:
                continue
            r = X[:, i] / X[:, j]
            r = r[np.isfinite(r)]
            if len(r) < 1000:
                continue
            CV[i, j] = np.nanstd(r) / abs(np.nanmean(r))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(CV, cmap="viridis", vmin=0, vmax=0.05)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ALL_PRODUCTS, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ALL_PRODUCTS, fontsize=5)
    plt.colorbar(im, ax=ax, label="CV(P_i / P_j)")
    ax.set_title("T5 — coefficient of variation of P_i / P_j (low = stable ratio)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t5_ratios.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[T5] saved {out}")

    flat = []
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(CV[i, j]):
                continue
            flat.append((CV[i, j], ALL_PRODUCTS[i], ALL_PRODUCTS[j]))
    flat.sort()
    print(f"\n[T5] Top {top} stable-ratio pairs (CV(P_i/P_j) low):")
    print(f"{'CV':>8}  same_cat  {'P_i':<28}  {'P_j':<28}  mean_ratio")
    for cv, pi, pj in flat[:top]:
        i, j = ALL_PRODUCTS.index(pi), ALL_PRODUCTS.index(pj)
        same = "Y" if P2CAT[pi] == P2CAT[pj] else "."
        mr = means[i] / means[j]
        print(f"{cv:8.5f}     {same}      {pi:<28}  {pj:<28}  {mr:9.4f}")
    return {"CV": CV, "top": flat[:top]}


# ---------- T6: Lagged-sum stationarity ---------------------------------- #


def t6_lagged_sum(wide: pd.DataFrame, mode: str = "intra"):
    """For each within-category pair, std of P_i(t) + P_j(t-k) at k ∈ LAGS.
    The shift is per-day (so we don't bridge day boundaries).
    """
    days = day_index(wide.index)

    def shifted_std(pi: str, pj: str, lag: int) -> float:
        # Per-day shift to avoid day-boundary leak
        vals = []
        for d_id in np.unique(days):
            mask = days == d_id
            xi = wide[pi].values[mask]
            xj = wide[pj].values[mask]
            if lag == 0:
                vals.append(xi + xj)
            elif lag > 0:
                # P_i(t) + P_j(t - lag): align shifted xj into xi space
                if len(xj) <= lag:
                    continue
                vals.append(xi[lag:] + xj[:-lag])
        if not vals:
            return float("nan")
        cat = np.concatenate(vals)
        # Demean per-day series isn't easy here; just take overall std after
        # centering (level offset between days irrelevant if both products move
        # together)
        return float(np.nanstd(cat - np.nanmean(cat)))

    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    axes = axes.ravel()
    print(f"\n[T6] Lagged-sum stationarity (intra-category pairs):")
    print(f"  rows = pair, cols = lag, value = std(P_i(t) + P_j(t-k)) / std(P_i)")
    summary = {}
    for k, (cat, plist) in enumerate(CATEGORIES.items()):
        sub = wide[plist].dropna()
        sigmas = {p: float(sub[p].std()) for p in plist}
        labels = []
        M = []
        for i, pi in enumerate(plist):
            for j, pj in enumerate(plist):
                if i == j:
                    continue
                row = []
                for lag in LAGS:
                    s = shifted_std(pi, pj, lag)
                    row.append(s / sigmas[pi] if sigmas[pi] > 0 else np.nan)
                M.append(row)
                labels.append(f"{SHORT[pi]}+{SHORT[pj]}(-k)")
        M = np.array(M)
        ax = axes[k]
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=2.0)
        ax.set_xticks(range(len(LAGS)))
        ax.set_xticklabels(LAGS, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title(cat, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
        # Print best lag per pair (when k>0 beats k=0 by >5%)
        for li, label in enumerate(labels):
            base = M[li, 0]
            best_k_idx = int(np.nanargmin(M[li]))
            best_lag = LAGS[best_k_idx]
            best_val = M[li, best_k_idx]
            if best_lag != 0 and best_val < base * 0.95:
                print(
                    f"  {cat:<14}  {label:<35}  lag0={base:.3f}  best lag={best_lag:>3}  val={best_val:.3f}"
                )
        summary[cat] = {"labels": labels, "M": M.tolist(), "lags": list(LAGS)}
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_t6_lagged_sum.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[T6] saved {out}")
    return summary


# ---------- T7: Cross-category pair scan --------------------------------- #


def t7_cross_category(wide: pd.DataFrame, top: int = 40):
    n = len(ALL_PRODUCTS)
    X = wide.values.astype(float)
    days = day_index(wide.index)
    # Per-day demean to remove level shifts
    Xd = X.copy()
    for p_idx in range(n):
        for d_id in np.unique(days):
            mask = days == d_id
            col = Xd[mask, p_idx]
            Xd[mask, p_idx] = col - np.nanmean(col)

    sigmas = np.nanstd(Xd, axis=0)
    rows = []
    for i, j in combinations(range(n), 2):
        xi = Xd[:, i]
        xj = Xd[:, j]
        s_diff = float(np.nanstd(xi - xj))
        s_sum = float(np.nanstd(xi + xj))
        norm = float(min(sigmas[i], sigmas[j]))
        if norm == 0:
            continue
        rows.append({
            "i": ALL_PRODUCTS[i],
            "j": ALL_PRODUCTS[j],
            "same_cat": P2CAT[ALL_PRODUCTS[i]] == P2CAT[ALL_PRODUCTS[j]],
            "diff": s_diff / norm,
            "sum": s_sum / norm,
            "best": min(s_diff, s_sum) / norm,
            "best_op": "diff" if s_diff < s_sum else "sum",
            "sigma_i": float(sigmas[i]),
            "sigma_j": float(sigmas[j]),
        })

    df = pd.DataFrame(rows)
    cross_df = df[~df["same_cat"]].sort_values("best").reset_index(drop=True)
    intra_df = df[df["same_cat"]].sort_values("best").reset_index(drop=True)

    print(f"\n[T7] Top {top} CROSS-category pair candidates (intra-day demeaned):")
    print(f"  ratio = std(spread) / min(σ_i, σ_j); op = sum or diff")
    print(f"{'best':>6} {'op':>4}  {'P_i':<28}  {'P_j':<28}  σ_i      σ_j")
    for _, r in cross_df.head(top).iterrows():
        print(f"{r['best']:6.3f} {r['best_op']:>4}  {r['i']:<28}  {r['j']:<28}  {r['sigma_i']:7.2f}  {r['sigma_j']:7.2f}")

    print(f"\n[T7] Top {top//2} INTRA-category pairs (sanity check — known sums should appear):")
    print(f"{'best':>6} {'op':>4}  {'P_i':<28}  {'P_j':<28}")
    for _, r in intra_df.head(top // 2).iterrows():
        print(f"{r['best']:6.3f} {r['best_op']:>4}  {r['i']:<28}  {r['j']:<28}")

    # Histogram: cross vs intra
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(intra_df["best"], bins=50, alpha=0.6, label=f"intra-cat (n={len(intra_df)})", color="C0")
    ax.hist(cross_df["best"], bins=50, alpha=0.6, label=f"cross-cat (n={len(cross_df)})", color="C1")
    ax.axvline(1.0, color="k", lw=0.6, ls="--", label="ratio=1 (no improvement)")
    ax.set_xlabel("min(std(diff), std(sum)) / min(σ_i, σ_j)")
    ax.set_ylabel("pair count")
    ax.set_title("T7 — pair-spread tightness, cross vs intra category")
    ax.legend()
    out = os.path.join(OUT_DIR, "round5_t7_cross_category_hist.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[T7] saved {out}")
    return {"cross_top": cross_df.head(top).to_dict("records"), "intra_top": intra_df.head(top).to_dict("records")}


# ---------- Main --------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "candidates"],
        help="Subset of tasks to run.",
    )
    parser.add_argument("--top", type=int, default=30, help="Top-N rows in stdout summaries.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {len(DAYS)} days from {ROUND_DIR} ...", flush=True)
    wide = load_wide()
    print(f"  shape = {wide.shape} (ticks × products), missing = {int(wide.isna().sum().sum())}")

    if "t1" in args.tasks:
        t1_beta_heatmap(wide, top=args.top)
    if "t2" in args.tasks:
        t2_svd_per_category(wide)
    if "t3" in args.tasks:
        t3_rank_invariants(wide)
    if "t4" in args.tasks:
        t4_envelope(wide)
    if "t5" in args.tasks:
        t5_ratio_invariants(wide, top=args.top)
    if "t6" in args.tasks:
        t6_lagged_sum(wide)
    if "t7" in args.tasks:
        t7_cross_category(wide, top=args.top)
    if "candidates" in args.tasks:
        plot_top_candidates(wide)


def plot_top_candidates(wide: pd.DataFrame):
    """Time-series plot of the most promising spread candidates surfaced by
    T1 / T7. Each panel = one spread, showing per-day demeaned series + ±2σ
    band. Useful for visually judging whether the spread is tradable."""
    candidates = [
        # (label, op, p_i, p_j, beta)  — A_t = p_i + beta * p_j
        ("SNACKPACK_VANILLA + CHOCOLATE (sum invariant)", "sum", "SNACKPACK_VANILLA", "SNACKPACK_CHOCOLATE", +1.0),
        ("MICROCHIP_CIRCLE - OXYGEN_SHAKE_CHOCOLATE", "diff", "MICROCHIP_CIRCLE", "OXYGEN_SHAKE_CHOCOLATE", -1.0),
        ("PANEL_4X4 - TRANSLATOR_GRAPHITE_MIST", "diff", "PANEL_4X4", "TRANSLATOR_GRAPHITE_MIST", -1.0),
        ("PANEL_1X4 + ROBOT_MOPPING (sum)", "sum", "PANEL_1X4", "ROBOT_MOPPING", +1.0),
        ("OXYGEN_SHAKE_MORNING_BREATH + PEBBLES_M (sum)", "sum", "OXYGEN_SHAKE_MORNING_BREATH", "PEBBLES_M", +1.0),
        ("PEBBLES_XL + 1.5*PEBBLES_L (intra non-trivial β)", "lin", "PEBBLES_XL", "PEBBLES_L", +1.5),
        ("SNACKPACK_RASPBERRY + STRAWBERRY (sum)", "sum", "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", +1.0),
        ("PANEL_2X4 + 0.5*PEBBLES_XS", "lin", "PANEL_2X4", "PEBBLES_XS", +0.5),
        ("PEBBLES_XS - 1.5*UV_VISOR_AMBER", "lin", "PEBBLES_XS", "UV_VISOR_AMBER", -1.5),
    ]
    days = day_index(wide.index)
    n = len(candidates)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, op, pi, pj, b) in zip(axes, candidates):
        spread = wide[pi].values + b * wide[pj].values
        # Per-day demean so the signal is visible across day shifts
        for d_id in np.unique(days):
            mask = days == d_id
            spread[mask] = spread[mask] - np.nanmean(spread[mask])
        sigma = np.nanstd(spread)
        ax.plot(spread, lw=0.4, color="C0")
        ax.axhline(0, color="k", lw=0.4)
        ax.axhline(2 * sigma, color="r", lw=0.4, ls="--", alpha=0.6)
        ax.axhline(-2 * sigma, color="r", lw=0.4, ls="--", alpha=0.6)
        for boundary in np.where(np.diff(days) != 0)[0]:
            ax.axvline(boundary, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"{label}  σ={sigma:.1f}", fontsize=9)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "round5_candidates.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[candidates] saved {out}")


if __name__ == "__main__":
    main()
