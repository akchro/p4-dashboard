"""Round 5 multi-product spread analysis core.

SVD-based pipeline for surfacing hidden linear constraints across the 50
Round 5 products organized in 10 categories of 5. Store-agnostic — both the
live and historical sub-tabs reuse this module by passing the right store.

Components:
  1. Per-category SV scree (drop-off at the smallest SV → linear constraint)
  2. Smallest-eigenvector recipe + integer-rounding (clean designer recipe)
  3. Spread time series with mean ± σ bands
  4. Global 50-product SVD with sparsified eigenvectors
  5. Per-day stability check (cosine similarity of eigvecs across days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

ALL_PRODUCTS: list[str] = [p for plist in CATEGORIES.values() for p in plist]
P2CAT: dict[str, str] = {p: c for c, plist in CATEGORIES.items() for p in plist}
SHORT: dict[str, str] = {
    p: p.replace(c + "_", "")
    for c, plist in CATEGORIES.items() for p in plist
}

TIMESTAMP_PER_DAY = 1_000_000


# ---------- Data prep ----------------------------------------------------- #


def build_wide(store, days: list[int] | None = None) -> pd.DataFrame:
    """Pivot store activities into a wide mid-price matrix.

    Index: global tick `t = (day - first_day) * TIMESTAMP_PER_DAY + timestamp`.
    Columns: any of `ALL_PRODUCTS` present in the store. Missing products are
    silently skipped so callers can still reason about the available subset.
    """
    frames = []
    for product in ALL_PRODUCTS:
        df = store.get_activities(product)
        if df is None or df.empty or "mid_price" not in df.columns:
            continue
        sub = df[["day", "timestamp", "mid_price"]].copy()
        if days is not None and len(days) > 0:
            sub = sub[sub["day"].isin(days)]
            if sub.empty:
                continue
        sub["product"] = product
        frames.append(sub)
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    first_day = int(df_all["day"].min())
    df_all["t"] = (df_all["day"] - first_day) * TIMESTAMP_PER_DAY + df_all["timestamp"]
    wide = df_all.pivot_table(index="t", columns="product", values="mid_price").sort_index()
    wide.attrs["first_day"] = first_day
    return wide


def day_index(t_index: pd.Index) -> np.ndarray:
    return (np.asarray(t_index) // TIMESTAMP_PER_DAY).astype(int)


def per_day_demean(M: np.ndarray, days: np.ndarray) -> np.ndarray:
    out = M.astype(float).copy()
    for d in np.unique(days):
        mask = days == d
        if mask.sum() == 0:
            continue
        out[mask] = out[mask] - np.nanmean(out[mask], axis=0, keepdims=True)
    return out


# ---------- SVD primitives ----------------------------------------------- #


def category_svd(wide: pd.DataFrame, products: list[str]) -> dict | None:
    """Per-day-demeaned SVD on a category. Returns None if data is missing."""
    cols = [p for p in products if p in wide.columns]
    if len(cols) < 2:
        return None
    sub = wide[cols].dropna()
    if sub.shape[0] < 5:
        return None
    days = day_index(sub.index)
    Xc = per_day_demean(sub.values, days)
    valid = ~np.any(np.isnan(Xc), axis=1)
    Xc = Xc[valid]
    if Xc.shape[0] < 5:
        return None
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return {
        "products": cols,
        "M": Xc,
        "t_index": sub.index[valid],
        "days": days[valid],
        "singular_values": s,
        "eigenvectors": Vt,
        "n_obs": Xc.shape[0],
    }


def find_clean_integer_recipe(eigenvector: np.ndarray, M: np.ndarray,
                              scales=(1, 2, 3, 5, 10), tol_factor: float = 5.0):
    """Try rounding eigvec*scale to nearest integer; pick the smallest scale
    where the recipe std stays within tol_factor × raw std. Returns
    (rounded, new_std) or (None, None)."""
    raw_proj = M @ eigenvector
    raw_std = float(np.std(raw_proj))
    if raw_std == 0:
        return None, None
    for scale in scales:
        rounded = np.round(eigenvector * scale)
        if np.all(rounded == 0):
            continue
        first_nz = next((x for x in rounded if x != 0), 0)
        if first_nz < 0:
            rounded = -rounded
        new_std = float(np.std(M @ rounded))
        if new_std < tol_factor * raw_std:
            return rounded, new_std
    return None, None


def spread_diagnostics(series: np.ndarray) -> dict:
    s = np.asarray(series, dtype=float)
    finite = ~np.isnan(s)
    s2 = s[finite]
    if s2.size < 3:
        return {"mean": np.nan, "std": np.nan, "autocorr_1": np.nan,
                "half_life": np.nan, "frac_2std": np.nan, "n": int(s2.size)}
    mean = float(s2.mean())
    std = float(s2.std())
    if s2.size < 2 or std == 0:
        ac = np.nan
        hl = np.nan
    else:
        cm = np.corrcoef(s2[:-1], s2[1:])
        ac = float(cm[0, 1]) if cm.shape == (2, 2) else np.nan
        if not np.isfinite(ac) or abs(ac) <= 0:
            hl = np.nan
        else:
            hl = -np.log(2) / np.log(max(0.001, abs(ac)))
    frac = float(np.mean(np.abs(s2 - mean) > 2 * std)) if std > 0 else np.nan
    return {"mean": mean, "std": std, "autocorr_1": ac, "half_life": hl,
            "frac_2std": frac, "n": int(s2.size)}


# ---------- Visualization helpers ---------------------------------------- #


def _empty_fig(text: str = "Load round 5 historical data to view") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{"text": text, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"size": 14, "color": "#888"}}],
        margin={"l": 30, "r": 10, "t": 30, "b": 30},
        plot_bgcolor="white",
    )
    return fig


def _is_round5(wide: pd.DataFrame) -> bool:
    """Cheap sanity check: at least one full Round 5 category present."""
    if wide.empty:
        return False
    for plist in CATEGORIES.values():
        if all(p in wide.columns for p in plist):
            return True
    return False


def build_scree_recipe_figure(wide: pd.DataFrame) -> go.Figure:
    """Component 1+2: 4 rows × 5 cols. Rows 1, 3 = scree; rows 2, 4 = recipe."""
    if not _is_round5(wide):
        return _empty_fig()

    cats = list(CATEGORIES.keys())
    titles = []
    for i, cat in enumerate(cats):
        titles.append(cat)
    # We need 4 rows, with rows 1+3 holding the title and rows 2+4 unlabeled.
    subplot_titles = []
    for row_idx in range(4):
        for col in range(5):
            cat_idx = (row_idx // 2) * 5 + col
            if cat_idx >= len(cats):
                subplot_titles.append("")
                continue
            if row_idx % 2 == 0:
                subplot_titles.append(cats[cat_idx])
            else:
                subplot_titles.append("")

    fig = make_subplots(
        rows=4, cols=5,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        row_heights=[0.27, 0.23, 0.27, 0.23],
    )

    for i, cat in enumerate(cats):
        block = i // 5  # 0 or 1 (top 5 vs bottom 5)
        col = (i % 5) + 1
        scree_row = block * 2 + 1
        recipe_row = block * 2 + 2

        plist = CATEGORIES[cat]
        result = category_svd(wide, plist)
        if result is None:
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="missing", showarrow=False,
                font={"color": "#999", "size": 10},
                row=scree_row, col=col,
            )
            continue

        s = result["singular_values"]
        ratio = s[-1] / s[0] if s[0] > 0 else np.nan
        s_norm = s / s.max()
        # color smallest red if ratio < 0.1; otherwise neutral
        bar_colors = ["#5b9bd5"] * len(s)
        if ratio < 0.1:
            bar_colors[-1] = "#d62728"
        elif ratio < 0.2:
            bar_colors[-1] = "#ff9933"
        fig.add_trace(go.Bar(
            x=[f"σ{j+1}" for j in range(len(s))],
            y=s_norm,
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate=("σ%{x}<br>raw=%{customdata:.3f}"
                           "<br>norm=%{y:.3f}<extra></extra>"),
            customdata=s,
        ), row=scree_row, col=col)
        # Annotate ratio
        fig.add_annotation(
            x=0.97, y=0.97, xref="x domain", yref="y domain",
            xanchor="right", yanchor="top",
            text=f"σ₅/σ₁={ratio:.4f}",
            showarrow=False,
            font={"size": 9, "color": "#444"},
            bgcolor="rgba(255,255,255,0.7)",
            row=scree_row, col=col,
        )

        # Recipe: smallest eigenvector + try integer rounding
        v = result["eigenvectors"][-1]
        labels = [SHORT[p] for p in result["products"]]
        rec_colors = ["#2ca02c" if x >= 0 else "#d62728" for x in v]
        fig.add_trace(go.Bar(
            x=labels,
            y=v,
            marker_color=rec_colors,
            showlegend=False,
            hovertemplate="%{x}<br>w=%{y:+.4f}<extra></extra>",
        ), row=recipe_row, col=col)

        # Try integer rounding; annotate result
        rounded, _new_std = find_clean_integer_recipe(v, result["M"])
        if rounded is not None:
            recipe_str = "(" + ",".join(f"{int(x):+d}" for x in rounded) + ")"
            ann_text = f"int: {recipe_str}"
        else:
            ann_text = "no clean int"
        fig.add_annotation(
            x=0.5, y=-0.35, xref="x domain", yref="y domain",
            xanchor="center", yanchor="top",
            text=ann_text,
            showarrow=False,
            font={"size": 8, "color": "#555"},
            row=recipe_row, col=col,
        )

    # Tighten ticks
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].tickfont = {"size": 8}
        if axis.startswith("yaxis"):
            fig.layout[axis].tickfont = {"size": 8}
    for ann in fig.layout.annotations:
        if ann.text in cats:
            ann.font = {"size": 11, "color": "#333"}

    fig.update_layout(
        title="Per-category SVD — singular values (top) + smallest-σ eigenvector recipe (bottom)",
        margin={"l": 30, "r": 10, "t": 60, "b": 30},
        plot_bgcolor="white",
        height=720,
    )
    return fig


def build_spread_series_figure(wide: pd.DataFrame, category: str,
                               use_int_recipe: bool = True) -> go.Figure:
    """Component 3: spread time series for one category."""
    if not _is_round5(wide):
        return _empty_fig()
    if category not in CATEGORIES:
        return _empty_fig(f"Unknown category: {category}")
    result = category_svd(wide, CATEGORIES[category])
    if result is None:
        return _empty_fig(f"{category}: insufficient data")

    v_raw = result["eigenvectors"][-1]
    rounded, new_std = find_clean_integer_recipe(v_raw, result["M"])
    if use_int_recipe and rounded is not None:
        weights = rounded.astype(float)
        recipe_label = "int " + "(" + ",".join(f"{int(x):+d}" for x in rounded) + ")"
    else:
        weights = v_raw
        recipe_label = "raw eigvec"

    series = result["M"] @ weights
    diag = spread_diagnostics(series)
    t = result["t_index"]
    days = result["days"]
    first_day = int(wide.attrs.get("first_day", 0))

    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    unique_days = sorted(np.unique(days).tolist())
    for di, d in enumerate(unique_days):
        mask = days == d
        fig.add_trace(go.Scatter(
            x=np.asarray(t)[mask], y=np.asarray(series)[mask],
            mode="lines",
            name=f"day {int(d) + first_day}",
            line={"color": palette[di % len(palette)], "width": 1},
        ))

    # Reference lines
    if np.isfinite(diag["mean"]) and np.isfinite(diag["std"]) and diag["std"] > 0:
        for k, dash in [(0, "solid"), (1, "dot"), (2, "dash"), (5, "longdash")]:
            for sign in (1, -1) if k != 0 else (1,):
                y = diag["mean"] + sign * k * diag["std"]
                fig.add_hline(y=y, line_color="#999", line_dash=dash, line_width=1,
                              opacity=0.4 if k != 0 else 0.6)

    title = (f"{category} spread (per-day demeaned, {recipe_label}) — "
             f"μ={diag['mean']:+.2f}, σ={diag['std']:.3f}, "
             f"AC1={diag['autocorr_1']:.3f}, "
             f"frac>2σ={diag['frac_2std']:.1%}, n={diag['n']}")
    fig.update_layout(
        title=title,
        xaxis_title="t (global, days × 1e6 + tick)",
        yaxis_title="spread = M @ weights (price units)",
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        plot_bgcolor="white",
        legend={"orientation": "h", "y": -0.15},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_stability_figure(wide: pd.DataFrame, category: str) -> go.Figure:
    """Component 5: per-day SVD comparison. Bars per day showing the smallest
    eigenvector + cosine similarity matrix between days."""
    if not _is_round5(wide):
        return _empty_fig()
    if category not in CATEGORIES:
        return _empty_fig(f"Unknown category: {category}")
    plist = CATEGORIES[category]
    cols = [p for p in plist if p in wide.columns]
    if len(cols) < 2:
        return _empty_fig(f"{category}: missing products")
    sub = wide[cols].dropna()
    if sub.empty:
        return _empty_fig(f"{category}: empty data")

    days_arr = day_index(sub.index)
    unique_days = sorted(np.unique(days_arr).tolist())
    if len(unique_days) < 2:
        return _empty_fig(f"{category}: need ≥2 days for stability check")
    first_day = int(wide.attrs.get("first_day", 0))

    per_day = []
    for d in unique_days:
        mask = days_arr == d
        if mask.sum() < 5:
            continue
        Xd = sub.values[mask]
        Xd = Xd - Xd.mean(axis=0, keepdims=True)
        _, s, Vt = np.linalg.svd(Xd, full_matrices=False)
        per_day.append({"day": int(d) + first_day, "s": s, "v": Vt[-1],
                        "ratio": s[-1] / s[0] if s[0] > 0 else np.nan})
    if len(per_day) < 2:
        return _empty_fig(f"{category}: not enough valid days")

    # Sign-align eigenvectors to first day
    ref = per_day[0]["v"]
    for entry in per_day[1:]:
        if np.dot(ref, entry["v"]) < 0:
            entry["v"] = -entry["v"]

    fig = make_subplots(
        rows=1, cols=len(per_day) + 1,
        column_widths=[1.0] * len(per_day) + [1.2],
        subplot_titles=[f"day {e['day']}  σ₅/σ₁={e['ratio']:.4f}" for e in per_day]
                       + ["pairwise |cos sim|"],
        horizontal_spacing=0.06,
    )
    labels = [SHORT[p] for p in cols]
    for i, entry in enumerate(per_day):
        v = entry["v"]
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in v]
        fig.add_trace(go.Bar(
            x=labels, y=v, marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>w=%{y:+.4f}<extra></extra>",
        ), row=1, col=i + 1)

    # Cosine similarity matrix
    n_d = len(per_day)
    sim = np.zeros((n_d, n_d))
    for i in range(n_d):
        for j in range(n_d):
            vi = per_day[i]["v"]; vj = per_day[j]["v"]
            sim[i, j] = abs(float(np.dot(vi, vj)) /
                            (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))
    day_labels = [f"d{e['day']}" for e in per_day]
    fig.add_trace(go.Heatmap(
        z=sim, x=day_labels, y=day_labels,
        zmin=0, zmax=1, colorscale="Viridis",
        text=[[f"{v:.3f}" for v in row] for row in sim],
        texttemplate="%{text}",
        showscale=False,
        hovertemplate="%{y} ↔ %{x}<br>|cos|=%{z:.3f}<extra></extra>",
    ), row=1, col=len(per_day) + 1)

    # Worst-pair similarity → caption
    if n_d >= 2:
        off_diag = sim[np.triu_indices(n_d, k=1)]
        worst = float(off_diag.min()) if off_diag.size else np.nan
        verdict = ("STABLE (>0.95)" if worst > 0.95
                   else "marginal (0.7–0.95)" if worst > 0.7
                   else "UNSTABLE (<0.7)")
    else:
        verdict = "n/a"

    fig.update_layout(
        title=f"{category} — per-day eigenvector stability   |   worst pair: {verdict}",
        margin={"l": 40, "r": 10, "t": 50, "b": 40},
        plot_bgcolor="white",
        height=320,
    )
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].tickfont = {"size": 9}
    return fig


def build_global_svd_figure(wide: pd.DataFrame,
                             n_smallest: int = 8,
                             sparsify_threshold: float = 0.15) -> go.Figure:
    """Component 4: SVD across all 50 products. Returns scree plot above
    a heatmap of the n_smallest eigenvectors (sparsified)."""
    if not _is_round5(wide):
        return _empty_fig()

    cols = [p for p in ALL_PRODUCTS if p in wide.columns]
    if len(cols) < 5:
        return _empty_fig("global SVD: too few products in store")
    sub = wide[cols].dropna()
    if sub.shape[0] < 10:
        return _empty_fig("global SVD: not enough overlapping ticks")

    days = day_index(sub.index)
    Xc = per_day_demean(sub.values, days)
    valid = ~np.any(np.isnan(Xc), axis=1)
    Xc = Xc[valid]
    if Xc.shape[0] < 10:
        return _empty_fig("global SVD: too few valid rows after demean")

    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Sparsify the n_smallest eigvecs and report their stds
    eigs = []
    for k in range(1, n_smallest + 1):
        v = Vt[-k]
        max_abs = np.abs(v).max()
        if max_abs == 0:
            continue
        sparse = v.copy()
        sparse[np.abs(sparse) < sparsify_threshold * max_abs] = 0
        if np.all(sparse == 0):
            sparse = v
        # renormalize the sparse vector for fair std comparison
        norm = np.linalg.norm(sparse)
        if norm > 0:
            sparse = sparse / norm * np.linalg.norm(v)
        raw_std = float(np.std(Xc @ v))
        sparse_std = float(np.std(Xc @ sparse))
        eigs.append({
            "rank_from_smallest": k,
            "sigma": float(s[-k]),
            "v_raw": v,
            "v_sparse": sparse,
            "raw_std": raw_std,
            "sparse_std": sparse_std,
            "n_nonzero_sparse": int(np.sum(np.abs(sparse) > 1e-9)),
        })

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.25, 0.75],
        subplot_titles=["Singular values (50)", "Smallest eigenvectors (sparsified)"],
        vertical_spacing=0.1,
    )

    # Scree
    fig.add_trace(go.Bar(
        x=[f"σ{i+1}" for i in range(len(s))],
        y=s,
        marker_color=["#5b9bd5" if i < len(s) - n_smallest else "#d62728"
                      for i in range(len(s))],
        showlegend=False,
        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    # Heatmap of sparsified eigenvectors
    if eigs:
        z = np.array([e["v_sparse"] for e in eigs])
        ylabels = [f"σ{len(s)-e['rank_from_smallest']+1} "
                   f"(σ={e['sigma']:.2f}, std_sparse={e['sparse_std']:.2f}, "
                   f"k={e['n_nonzero_sparse']})"
                   for e in eigs]
        fig.add_trace(go.Heatmap(
            z=z,
            x=cols,
            y=ylabels,
            colorscale="RdBu",
            zmid=0,
            colorbar={"len": 0.5, "y": 0.3},
            hovertemplate="%{y}<br>%{x}<br>w=%{z:+.4f}<extra></extra>",
        ), row=2, col=1)

    fig.update_xaxes(tickfont={"size": 7}, row=2, col=1, tickangle=-90)
    fig.update_yaxes(tickfont={"size": 9}, row=2, col=1)

    fig.update_layout(
        title=f"Global SVD across {len(cols)} products (per-day demeaned, sparsified ≥{sparsify_threshold:.0%} of max |w|)",
        margin={"l": 50, "r": 10, "t": 70, "b": 80},
        plot_bgcolor="white",
        height=720,
    )
    return fig


def build_recipe_summary_table(wide: pd.DataFrame) -> go.Figure:
    """Tabular summary of all 10 categories' σ-ratios and clean recipes,
    so you can scan all candidates at a glance."""
    if not _is_round5(wide):
        return _empty_fig()
    rows = []
    for cat, plist in CATEGORIES.items():
        result = category_svd(wide, plist)
        if result is None:
            rows.append([cat, "—", "—", "—", "—", "—", "—", "missing data"])
            continue
        s = result["singular_values"]
        v = result["eigenvectors"][-1]
        ratio = s[-1] / s[0] if s[0] > 0 else np.nan
        rounded, _ = find_clean_integer_recipe(v, result["M"])
        if rounded is not None:
            recipe = "(" + ",".join(f"{int(x):+d}" for x in rounded) + ")"
            k_nz = int(np.sum(np.abs(rounded) > 0))
        else:
            recipe = "—"
            k_nz = 0
        spread = result["M"] @ (rounded if rounded is not None else v)
        diag = spread_diagnostics(spread)
        # Tier requires both a small σ₅/σ₁ ratio AND a non-trivial recipe (k≥2)
        if ratio < 0.05 and k_nz >= 2:
            verdict = "Tier 1 ✓"
        elif ratio < 0.1 and k_nz >= 2:
            verdict = "Tier 2"
        elif k_nz < 2:
            verdict = "trivial (k=1)"
        else:
            verdict = "noise"
        rows.append([
            cat,
            f"{ratio:.4f}",
            f"{s[-1]:.3f}",
            recipe,
            str(k_nz),
            f"{diag['std']:.3f}",
            f"{diag['autocorr_1']:.3f}",
            verdict,
        ])
    fig = go.Figure(data=[go.Table(
        header={
            "values": ["Category", "σ₅/σ₁", "σ₅", "Int recipe", "k", "spread σ", "AC₁", "tier"],
            "fill_color": "#eef2f7",
            "font": {"size": 11, "color": "#222"},
            "align": "left",
        },
        cells={
            "values": list(zip(*rows)) if rows else [],
            "fill_color": [["#fff" if i % 2 == 0 else "#fafafa" for i in range(len(rows))]] * 8,
            "font": {"size": 11, "color": "#222"},
            "align": "left",
        },
    )])
    fig.update_layout(
        title="Per-category ranking (sorted in display order)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig
