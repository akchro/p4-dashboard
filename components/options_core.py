"""Store-agnostic figure builders for the options panels.

Each `build_*` function takes a store module (either `data.store` or
`data.historical_store`) and returns a Plotly figure. Live/historical panel
modules wire these up with their own IDs and callbacks.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import options as opts

# Consistent strike colours so overlay / smile / IV-TS panels share a palette.
_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#17becf", "#8c564b", "#bcbd22",
]


def strike_color(strike):
    try:
        idx = opts.STRIKES.index(int(strike))
    except ValueError:
        idx = 0
    return _PALETTE[idx % len(_PALETTE)]


def _empty(text):
    fig = go.Figure()
    fig.add_annotation(
        text=text, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font={"size": 14, "color": "#999"},
    )
    fig.update_layout(margin={"l": 50, "r": 20, "t": 40, "b": 30},
                      template="plotly_white")
    return fig


def _downsample(df, step):
    if step and step > 1 and not df.empty:
        return df.iloc[::step]
    return df


# ---------------------------------------------------------------------------
# Overlay: voucher mid vs intrinsic floor max(S-K, 0)
# ---------------------------------------------------------------------------

def build_overlay_figure(store_module, strikes, day, downsample, mode="overlay",
                         tte_start=None, sigma_baseline=0.013):
    """Overlay voucher mid with reference lines per strike.

    mode = "overlay"   → voucher mid + intrinsic floor max(S-K, 0); S on y2
    mode = "extrinsic" → C - max(S-K, 0) per strike (bounded to small y-range)
    mode = "implied"   → BS-inverted underlying per voucher at sigma_baseline,
                          plotted alongside observed S on a single y-axis. In a
                          fair market all lines collapse onto S; gaps = mispricing.
    mode = "rebased"   → voucher mid shifted so each line starts at S(t₀).
                          Lets you eyeball relative movement; absolute levels
                          are NOT actual prices.
    """
    if not strikes or not store_module.is_loaded():
        return _empty("Select one or more voucher strikes")

    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")

    ve = _downsample(ve, downsample)
    fig = go.Figure()

    for K in strikes:
        try:
            K_int = int(K)
        except (TypeError, ValueError):
            continue
        color = strike_color(K_int)
        voucher = store_module.get_activities(f"VEV_{K_int}", day)
        if voucher.empty:
            continue
        voucher = _downsample(voucher, downsample)

        merged = pd.merge(
            voucher[["timestamp", "mid_price"]].rename(columns={"mid_price": "C"}),
            ve[["timestamp", "mid_price"]].rename(columns={"mid_price": "S"}),
            on="timestamp", how="inner",
        ).dropna()
        if merged.empty:
            continue

        if mode == "implied":
            T = opts.time_to_expiry(merged["timestamp"].to_numpy(),
                                    tte_start if tte_start else 8.0)
            S_implied = opts.implied_underlying(
                merged["C"].to_numpy(), K_int, T, sigma_baseline,
            )
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=S_implied,
                mode="lines", name=f"VEV_{K_int} → S_impl",
                line={"color": color, "width": 1.6},
                connectgaps=False,
                hovertemplate="t=%{x}<br>S_impl=%{y:.2f}<extra>VEV_%{meta}</extra>",
                meta=K_int,
            ))
            continue

        if mode == "rebased":
            shift = float(merged["S"].iloc[0]) - float(merged["C"].iloc[0])
            y_rebased = merged["C"] + shift
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=y_rebased,
                mode="lines", name=f"VEV_{K_int} (rebased)",
                line={"color": color, "width": 2},
                connectgaps=False,
                customdata=np.stack([merged["C"].to_numpy(),
                                     np.full(len(merged), shift)], axis=-1),
                hovertemplate=("t=%{x}<br>rebased=%{y:.2f}<br>"
                               "actual mid=%{customdata[0]:.2f}<br>"
                               "shift=%{customdata[1]:+.2f}"
                               "<extra>VEV_%{meta}</extra>"),
                meta=K_int,
            ))
            continue

        intrinsic = np.maximum(merged["S"] - K_int, 0.0)
        if mode == "extrinsic":
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["C"] - intrinsic,
                mode="lines", name=f"VEV_{K_int} extrinsic",
                line={"color": color, "width": 2},
                connectgaps=False,
                hovertemplate="t=%{x}<br>extrinsic=%{y:.2f}<extra>VEV_%{meta}</extra>",
                meta=K_int,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["C"],
                mode="lines", name=f"VEV_{K_int} mid",
                line={"color": color, "width": 2},
                connectgaps=False,
                hovertemplate="t=%{x}<br>mid=%{y:.2f}<extra>VEV_%{meta}</extra>",
                meta=K_int,
            ))
            fig.add_trace(go.Scatter(
                x=merged["timestamp"], y=intrinsic,
                mode="lines", name=f"max(S−{K_int}, 0)",
                line={"color": color, "width": 1.3, "dash": "dot"},
                connectgaps=False,
                hovertemplate="t=%{x}<br>S−K=%{y:.2f}<extra>K=%{meta}</extra>",
                meta=K_int,
            ))

    if mode in ("overlay", "implied", "rebased") and not ve.empty:
        s_axis = "y2" if mode == "overlay" else "y"
        fig.add_trace(go.Scatter(
            x=ve["timestamp"], y=ve["mid_price"],
            mode="lines", name=f"{opts.UNDERLYING} mid (observed)",
            line={"color": "#000000", "width": 2.2},
            yaxis=s_axis,
            connectgaps=False,
            hovertemplate="t=%{x}<br>S=%{y:.2f}<extra>VEV underlying</extra>",
        ))

    title_map = {
        "overlay": "mid vs intrinsic floor",
        "extrinsic": "extrinsic value",
        "implied": f"implied underlying (σ={sigma_baseline:.4f}/√day)",
        "rebased": "rebased to S(t₀)",
    }
    yaxis_title_map = {
        "overlay": "Voucher price",
        "extrinsic": "Extrinsic (C − max(S−K, 0))",
        "implied": "Underlying price (observed & BS-implied per voucher)",
        "rebased": "Rebased price (NOT actual — shifted to align at t₀)",
    }
    top_margin = 70 if mode == "rebased" else 40
    layout_kwargs = {
        "title": f"Voucher {title_map.get(mode, mode)}",
        "xaxis_title": "Timestamp",
        "yaxis_title": yaxis_title_map.get(mode, "Price"),
        "hovermode": "x unified",
        "margin": {"l": 50, "r": 60, "t": top_margin, "b": 30},
        "legend": {"orientation": "h", "y": -0.15},
        "template": "plotly_white",
    }
    if mode == "overlay":
        layout_kwargs["yaxis2"] = {
            "title": f"{opts.UNDERLYING} mid",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    fig.update_layout(**layout_kwargs)
    if mode == "extrinsic":
        fig.add_hline(y=0, line={"color": "#999", "width": 1, "dash": "dash"})
    if mode == "rebased":
        fig.add_annotation(
            text=("<b>NORMALIZED VIEW</b> — voucher mids are shifted vertically so each "
                  "starts at S(t₀). Y-axis values are NOT actual voucher prices; "
                  "only the shape and relative movement are meaningful."),
            xref="paper", yref="paper",
            x=0.5, y=1.0, xanchor="center", yanchor="bottom",
            showarrow=False,
            font={"size": 11, "color": "#664d03"},
            bgcolor="#fff3cd", bordercolor="#f0ad4e", borderwidth=1, borderpad=5,
        )
    return fig


# ---------------------------------------------------------------------------
# Volatility smile — full-day IV scatter, color-coded by strike
# ---------------------------------------------------------------------------

def build_smile_figure(store_module, strikes, day, tte_start, downsample, x_axis, show_fit):
    """Scatter cloud of (x, IV) across the whole day for each selected strike.

    x is one of:
        "strike"        → raw strike K
        "moneyness"     → log-moneyness m = ln(K/S) / √T  (call-OTM is positive)
        "moneyness_sk"  → log-moneyness m = ln(S/K) / √T  (call-ITM is positive)
    Each strike gets its own color (shared with overlay / IV-TS). Optional
    quadratic fit is one parabola through the combined cloud.

    Floor-pinned points (voucher price ≤ 0.5 + eps where the tick grid clamps
    deep OTM) are drawn as open markers and excluded from the fit — their IVs
    are artifacts of price quantization, not real surface readings.
    """
    if not strikes or not store_module.is_loaded():
        return _empty("Select one or more voucher strikes")

    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    ve_ds = _downsample(ve, downsample)

    fig = go.Figure()
    fit_x = []   # only non-floor points feed the fit
    fit_y = []
    floor_eps = 1e-3
    floor_count_total = 0
    real_count_total = 0

    for K in strikes:
        try:
            K_int = int(K)
        except (TypeError, ValueError):
            continue
        voucher = store_module.get_activities(f"VEV_{K_int}", day)
        if voucher.empty:
            continue
        voucher = _downsample(voucher, downsample)
        merged = pd.merge(
            voucher[["timestamp", "mid_price"]].rename(columns={"mid_price": "C"}),
            ve_ds[["timestamp", "mid_price"]].rename(columns={"mid_price": "S"}),
            on="timestamp", how="inner",
        ).dropna()
        if merged.empty:
            continue
        T = opts.time_to_expiry(merged["timestamp"].values, float(tte_start))
        K_arr = np.full(len(merged), K_int, dtype=float)
        C_arr = merged["C"].values
        S_arr = merged["S"].values
        iv = opts.implied_vol(C_arr, S_arr, K_arr, T)
        m = opts.log_moneyness(K_arr, S_arr, T)
        valid = np.isfinite(iv)
        if not valid.any():
            continue

        # Floor-pinning: voucher mid sits at the 0.5 tick floor (deep OTM)
        # — IV is a quantization artifact, not a market view.
        is_floor = C_arr <= 0.5 + floor_eps

        if x_axis == "moneyness":
            xs_full = m
        elif x_axis == "moneyness_sk":
            xs_full = -m
        else:
            xs_full = K_arr

        real_mask = valid & ~is_floor
        floor_mask = valid & is_floor
        ts_arr = merged["timestamp"].values

        if real_mask.any():
            real_count_total += int(real_mask.sum())
            fit_x.append(xs_full[real_mask])
            fit_y.append(iv[real_mask])
            fig.add_trace(go.Scatter(
                x=xs_full[real_mask], y=iv[real_mask], mode="markers",
                name=f"VEV_{K_int}",
                marker={"color": strike_color(K_int), "size": 4, "opacity": 0.55},
                customdata=ts_arr[real_mask],
                hovertemplate=(
                    f"K={K_int}<br>"
                    "t=%{customdata:,}<br>"
                    "x=%{x:.4f}<br>"
                    "IV=%{y:.5f}<extra></extra>"
                ),
            ))
        if floor_mask.any():
            floor_count_total += int(floor_mask.sum())
            fig.add_trace(go.Scatter(
                x=xs_full[floor_mask], y=iv[floor_mask], mode="markers",
                name=f"VEV_{K_int} (floor-pinned)",
                marker={
                    "color": strike_color(K_int),
                    "size": 5, "opacity": 0.4,
                    "symbol": "circle-open",
                    "line": {"width": 1, "color": strike_color(K_int)},
                },
                customdata=ts_arr[floor_mask],
                hovertemplate=(
                    f"K={K_int} (FLOOR)<br>"
                    "t=%{customdata:,}<br>"
                    "x=%{x:.4f}<br>"
                    "IV=%{y:.5f} (artifact)<extra></extra>"
                ),
                showlegend=True,
            ))

    if not fit_x and floor_count_total == 0:
        return _empty("No solvable IVs across selected strikes")

    if show_fit and fit_x:
        xs_all = np.concatenate(fit_x)
        ys_all = np.concatenate(fit_y)
        if len(xs_all) >= 3:
            try:
                coef = np.polyfit(xs_all, ys_all, 2)
                xfit = np.linspace(xs_all.min(), xs_all.max(), 80)
                yfit = np.polyval(coef, xfit)
                fig.add_trace(go.Scatter(
                    x=xfit, y=yfit, mode="lines",
                    name=f"fit (excl. floor): {coef[0]:.4g}x² + {coef[1]:.4g}x + {coef[2]:.4g}",
                    line={"color": "#000000", "width": 2.5},
                ))
            except (np.linalg.LinAlgError, ValueError):
                pass

    x_label = {
        "moneyness": "log-moneyness m = ln(K/S)/√T",
        "moneyness_sk": "log-moneyness m = ln(S/K)/√T",
    }.get(x_axis, "Strike K")
    title = "Volatility Smile · full day (color = strike)"
    if floor_count_total:
        title += f"  ·  {real_count_total} real / {floor_count_total} floor-pinned (open markers, excl. from fit)"
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Implied Volatility (per √day)",
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.25},
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# IV time series across strikes + optional realised-vol overlay
# ---------------------------------------------------------------------------

def _realized_sigma(ve_df, window):
    """Rolling realized sigma (per sqrt-day) from mid-price log-returns."""
    if ve_df.empty:
        return None
    df = ve_df[["timestamp", "mid_price"]].dropna().sort_values("timestamp")
    if len(df) < window + 2:
        return None
    prices = df["mid_price"].values
    ts = df["timestamp"].values
    log_ret = np.diff(np.log(prices))
    # dt between consecutive rows (in days) — may vary slightly
    dt = np.diff(ts) / opts.TIMESTAMP_PER_DAY
    # σ_per_√day = sqrt( rolling_mean(r^2) / dt )
    sq = log_ret ** 2
    # Align to ts[1:] (end of each return interval)
    series = pd.Series(sq / np.maximum(dt, 1e-9))
    rolling = series.rolling(window).mean()
    sigma = np.sqrt(rolling.values)
    return pd.DataFrame({"timestamp": ts[1:], "sigma": sigma}).dropna()


def build_iv_ts_figure(store_module, strikes, day, tte_start, downsample, show_realized,
                       realized_window=500):
    if not strikes or not store_module.is_loaded():
        return _empty("Select one or more voucher strikes")

    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    ve_ds = _downsample(ve, downsample)

    fig = go.Figure()

    for K in strikes:
        try:
            K_int = int(K)
        except (TypeError, ValueError):
            continue
        voucher = store_module.get_activities(f"VEV_{K_int}", day)
        if voucher.empty:
            continue
        voucher = _downsample(voucher, downsample)
        merged = pd.merge(
            voucher[["timestamp", "mid_price"]].rename(columns={"mid_price": "C"}),
            ve_ds[["timestamp", "mid_price"]].rename(columns={"mid_price": "S"}),
            on="timestamp", how="inner",
        ).dropna()
        if merged.empty:
            continue
        T = opts.time_to_expiry(merged["timestamp"].values, float(tte_start))
        iv = opts.implied_vol(
            merged["C"].values, merged["S"].values,
            np.full(len(merged), K_int, dtype=float), T,
        )
        fig.add_trace(go.Scatter(
            x=merged["timestamp"], y=iv, mode="lines",
            name=f"VEV_{K_int}",
            line={"color": strike_color(K_int), "width": 1.6},
            connectgaps=False,
            hovertemplate="t=%{x}<br>IV=%{y:.5f}<extra>VEV_%{meta}</extra>",
            meta=K_int,
        ))

    if show_realized:
        rv = _realized_sigma(ve, realized_window)
        if rv is not None and not rv.empty:
            fig.add_trace(go.Scatter(
                x=rv["timestamp"], y=rv["sigma"],
                mode="lines", name=f"Realized σ (window={realized_window})",
                line={"color": "#000000", "width": 2, "dash": "dash"},
                hovertemplate="t=%{x}<br>realized=%{y:.5f}<extra></extra>",
            ))

    fig.update_layout(
        title="Implied Vol Time Series",
        xaxis_title="Timestamp",
        yaxis_title="IV / σ (per √day)",
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 40, "b": 30},
        legend={"orientation": "h", "y": -0.15},
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Snapshot helpers — used by the strategy panels below
# ---------------------------------------------------------------------------

def _snapshot(store_module, day, as_of_ts):
    """Resolve (actual_ts, S, ve_clean) at the requested as-of timestamp.

    Returns (None, None, None) when data is missing. `as_of_ts=None` snaps to
    the latest underlying observation. Any non-NaN underlying mid is acceptable;
    we snap to the nearest timestamp.
    """
    if not store_module.is_loaded():
        return None, None, None
    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return None, None, None
    ve = ve.dropna(subset=["mid_price"]).sort_values("timestamp").reset_index(drop=True)
    if ve.empty:
        return None, None, None
    if as_of_ts is None:
        actual_ts = int(ve["timestamp"].iloc[-1])
        S = float(ve["mid_price"].iloc[-1])
    else:
        idx = (ve["timestamp"] - int(as_of_ts)).abs().idxmin()
        actual_ts = int(ve.loc[idx, "timestamp"])
        S = float(ve.loc[idx, "mid_price"])
    return actual_ts, S, ve


def _voucher_snapshot(store_module, day, K, target_ts):
    """Voucher mid at the timestamp nearest `target_ts` (or NaN)."""
    df = store_module.get_activities(f"VEV_{K}", day)
    if df.empty:
        return float("nan")
    df = df.dropna(subset=["mid_price"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return float("nan")
    idx = (df["timestamp"] - int(target_ts)).abs().idxmin()
    return float(df.loc[idx, "mid_price"])


def _tte_remaining(actual_ts, tte_at_start):
    return max(float(tte_at_start) - actual_ts / opts.TIMESTAMP_PER_DAY, 1e-6)


def _lognormal_cdf(x, S0, sd):
    """P(S_T ≤ x) under driftless lognormal with sd = σ·√T (no risk-neutral drift)."""
    if not np.isfinite(x) or x <= 0 or sd <= 0 or S0 <= 0:
        return float("nan")
    return float(opts._norm_cdf(np.log(x / S0) / sd))


# ---------------------------------------------------------------------------
# Terminal-S distribution forecast
# ---------------------------------------------------------------------------

def build_terminal_dist_figure(store_module, strikes, day, as_of_ts,
                                sigma_r, tte_at_start):
    """Lognormal PDF of S at end-of-round given current S, σ_R, and remaining
    TTE. Strikes overlay as vertical dotted lines with P(S_T > K) annotations.
    Driftless convention: ln(S_T/S₀) ~ N(0, σ²·T).
    """
    actual_ts, S, _ = _snapshot(store_module, day, as_of_ts)
    if actual_ts is None:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    T = _tte_remaining(actual_ts, tte_at_start)
    sigma = float(sigma_r) if sigma_r else 0.0
    if sigma <= 0:
        return _empty("σ_R must be positive")

    sd = sigma * np.sqrt(T)
    s_lo = max(S * np.exp(-3.5 * sd), 1e-3)
    s_hi = S * np.exp(+3.5 * sd)
    s_grid = np.linspace(s_lo, s_hi, 400)
    pdf = (1.0 / (s_grid * sd * np.sqrt(2.0 * np.pi))) * \
        np.exp(-(np.log(s_grid / S) ** 2) / (2.0 * sd * sd))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_grid, y=pdf, mode="lines",
        name="terminal S PDF",
        line={"color": "#377eb8", "width": 2.2},
        fill="tozeroy", fillcolor="rgba(55,126,184,0.18)",
        hovertemplate="S=%{x:.2f}<br>density=%{y:.5f}<extra></extra>",
    ))
    fig.add_vline(
        x=S, line={"color": "black", "width": 1.6, "dash": "dash"},
        annotation_text=f"S₀={S:.1f}",
        annotation_position="top",
    )

    pdf_max = float(np.nanmax(pdf)) if pdf.size else 1.0
    selected = list(strikes) if strikes else list(opts.STRIKES)
    for K in selected:
        try:
            K_int = int(K)
        except (TypeError, ValueError):
            continue
        # Off-grid strikes still get an annotation at the edge
        z = np.log(K_int / S) / sd
        p_itm = float(1.0 - opts._norm_cdf(z))
        color = strike_color(K_int)
        if K_int < s_lo:
            x_anno = s_lo
            on_axis = False
        elif K_int > s_hi:
            x_anno = s_hi
            on_axis = False
        else:
            x_anno = K_int
            on_axis = True
        if on_axis:
            fig.add_vline(x=K_int, line={"color": color, "width": 1.1, "dash": "dot"})
        fig.add_annotation(
            x=x_anno, y=pdf_max * 0.95,
            text=f"K={K_int}<br>P(S_T>K)={p_itm:.0%}",
            showarrow=False,
            font={"size": 10, "color": color},
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color, borderwidth=1, borderpad=2,
            xanchor="center" if on_axis else ("left" if K_int < s_lo else "right"),
        )

    fig.update_layout(
        title=f"Terminal S distribution · t={actual_ts:,} · σ_R={sigma:.4f} · "
              f"TTE_remaining={T:.2f}d",
        xaxis_title="S at end of round",
        yaxis_title="Density",
        margin={"l": 50, "r": 20, "t": 50, "b": 30},
        template="plotly_white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Fly mispricing scan — market vs fair (BS at σ_R)
# ---------------------------------------------------------------------------

def build_fly_mispricing_figure(store_module, day, as_of_ts,
                                 sigma_r, tte_at_start):
    """For every symmetric butterfly (K1, K2, K3) with K2-K1 = K3-K2, compare
    market cost (C₁ − 2C₂ + C₃) to BS-fair cost at σ_R. Cheapest mispricing
    (most negative diff) is the trade candidate.
    """
    actual_ts, S, _ = _snapshot(store_module, day, as_of_ts)
    if actual_ts is None:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    T = _tte_remaining(actual_ts, tte_at_start)
    sigma = float(sigma_r) if sigma_r else 0.0
    if sigma <= 0:
        return _empty("σ_R must be positive")

    Cs = {}
    for K in opts.STRIKES:
        c = _voucher_snapshot(store_module, day, K, actual_ts)
        if np.isfinite(c):
            Cs[K] = c
    if len(Cs) < 3:
        return _empty("Not enough voucher prices at this snapshot")

    sks = sorted(Cs.keys())
    rows = []
    for i, K1 in enumerate(sks):
        for j in range(i + 1, len(sks)):
            K2 = sks[j]
            for k in range(j + 1, len(sks)):
                K3 = sks[k]
                if K2 - K1 != K3 - K2:
                    continue
                market_cost = Cs[K1] - 2.0 * Cs[K2] + Cs[K3]
                fair_cost = float(
                    opts.bs_call(np.array([S]), np.array([K1]),
                                 np.array([T]), np.array([sigma]))[0]
                    - 2.0 * opts.bs_call(np.array([S]), np.array([K2]),
                                         np.array([T]), np.array([sigma]))[0]
                    + opts.bs_call(np.array([S]), np.array([K3]),
                                   np.array([T]), np.array([sigma]))[0]
                )
                rows.append({
                    "fly": f"{K1}/{K2}/{K3}",
                    "K2": K2,
                    "width": K2 - K1,
                    "market": market_cost,
                    "fair": fair_cost,
                    "diff": market_cost - fair_cost,
                })

    if not rows:
        return _empty("No symmetric butterflies in the available strikes")

    df = pd.DataFrame(rows).sort_values(["width", "K2"]).reset_index(drop=True)
    cheapest = df.loc[df["diff"].idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["fly"], y=df["market"],
        name="Market cost (C₁ − 2C₂ + C₃)",
        marker_color="#e41a1c",
        hovertemplate="%{x}<br>market=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["fly"], y=df["fair"],
        name=f"Fair cost (BS @ σ={sigma:.4f})",
        marker_color="#377eb8",
        hovertemplate="%{x}<br>fair=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["fly"], y=df["diff"],
        mode="markers+text",
        name="market − fair",
        marker={"color": "black", "symbol": "diamond", "size": 9},
        text=[f"{d:+.2f}" for d in df["diff"]],
        textposition="top center",
        textfont={"size": 10},
        hovertemplate="%{x}<br>diff=%{y:+.3f}<extra></extra>",
    ))
    fig.add_annotation(
        x=cheapest["fly"], y=max(cheapest["market"], cheapest["fair"]),
        text=f"cheapest:<br>{cheapest['fly']} @ {cheapest['diff']:+.2f}",
        showarrow=True, arrowhead=2, ax=0, ay=-30,
        font={"size": 10, "color": "#005a9c"},
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#005a9c", borderwidth=1,
    )
    fig.update_layout(
        title=f"Fly cost: market vs fair · S={S:.1f} · TTE_remaining={T:.2f}d",
        xaxis_title="Butterfly K1/K2/K3",
        yaxis_title="Cost",
        barmode="group",
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        legend={"orientation": "h", "y": -0.25},
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Per-strike Γ/$ and Θ/$ — long-vol bang-for-buck ranker
# ---------------------------------------------------------------------------

def build_greeks_per_dollar_figure(store_module, day, as_of_ts,
                                    sigma_r, tte_at_start):
    """For each tradeable strike, compute Γ/C and Θ/C at σ_R and rank by Γ/$.
    Two side-by-side panels share the strike order."""
    actual_ts, S, _ = _snapshot(store_module, day, as_of_ts)
    if actual_ts is None:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    T = _tte_remaining(actual_ts, tte_at_start)
    sigma = float(sigma_r) if sigma_r else 0.0
    if sigma <= 0:
        return _empty("σ_R must be positive")

    rows = []
    for K in opts.STRIKES:
        C = _voucher_snapshot(store_module, day, K, actual_ts)
        if not np.isfinite(C) or C <= 0.5 + 1e-6:
            continue
        g = opts.greeks(np.array([S]), np.array([K]),
                        np.array([T]), np.array([sigma]))
        rows.append({
            "K": K,
            "C": C,
            "gamma": float(g["gamma"][0]),
            "theta": float(g["theta"][0]),
            "gamma_per_dollar": float(g["gamma"][0]) / C,
            "theta_per_dollar": float(g["theta"][0]) / C,
        })
    if not rows:
        return _empty("No solvable voucher prices at this snapshot")

    df = pd.DataFrame(rows).sort_values("gamma_per_dollar", ascending=False).reset_index(drop=True)
    colors = [strike_color(K) for K in df["K"]]
    x_labels = [str(K) for K in df["K"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Γ / C  (long-vol gain per $ spent)",
            f"Θ / C  (time-decay per $ spent — negative for long)",
        ),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        x=x_labels, y=df["gamma_per_dollar"],
        marker_color=colors, name="Γ/$",
        hovertemplate=("K=%{x}<br>Γ/C=%{y:.5g}<br>"
                       "Γ=%{customdata[0]:.5g}<br>"
                       "C=%{customdata[1]:.2f}<extra></extra>"),
        customdata=np.stack([df["gamma"], df["C"]], axis=-1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x_labels, y=df["theta_per_dollar"],
        marker_color=colors, name="Θ/$",
        hovertemplate=("K=%{x}<br>Θ/C=%{y:.5g}<br>"
                       "Θ=%{customdata[0]:.5g}<br>"
                       "C=%{customdata[1]:.2f}<extra></extra>"),
        customdata=np.stack([df["theta"], df["C"]], axis=-1),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Strike (sorted by Γ/$)", row=1, col=1)
    fig.update_xaxes(title_text="Strike (same order)", row=1, col=2)
    fig.update_yaxes(title_text="Γ / C", row=1, col=1)
    fig.update_yaxes(title_text="Θ / C", row=1, col=2)
    fig.update_layout(
        title=f"Per-strike Γ/$ and Θ/$ · S={S:.1f} · σ_R={sigma:.4f} · "
              f"TTE_remaining={T:.2f}d",
        margin={"l": 50, "r": 20, "t": 70, "b": 40},
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Probability of profit per candidate structure
# ---------------------------------------------------------------------------

def build_pop_per_structure_figure(store_module, day, as_of_ts,
                                    sigma_r, tte_at_start):
    """Compute P(profit at expiry) for: long calls, short calls, long
    butterflies (symmetric, every available width). Lognormal terminal S
    with σ = σ_R, no drift."""
    actual_ts, S, _ = _snapshot(store_module, day, as_of_ts)
    if actual_ts is None:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    T = _tte_remaining(actual_ts, tte_at_start)
    sigma = float(sigma_r) if sigma_r else 0.0
    if sigma <= 0:
        return _empty("σ_R must be positive")
    sd = sigma * np.sqrt(T)

    Cs = {}
    for K in opts.STRIKES:
        c = _voucher_snapshot(store_module, day, K, actual_ts)
        if np.isfinite(c):
            Cs[K] = c
    if not Cs:
        return _empty("No voucher prices at this snapshot")

    rows = []

    # Long calls — profit if S_T > K + cost
    for K, C in Cs.items():
        if C <= 0.5 + 1e-6:
            continue
        breakeven = K + C
        p = 1.0 - _lognormal_cdf(breakeven, S, sd)
        rows.append({
            "name": f"long call {K}",
            "type": "long_call",
            "cost": C,
            "p_profit": p,
            "max_loss": C,
            "max_gain": float("inf"),
        })

    # Short calls — profit if S_T < K + premium received
    for K, C in Cs.items():
        if C <= 0.5 + 1e-6:
            continue
        breakeven = K + C
        p = _lognormal_cdf(breakeven, S, sd)
        rows.append({
            "name": f"short call {K}",
            "type": "short_call",
            "cost": -C,  # premium received
            "p_profit": p,
            "max_loss": float("inf"),
            "max_gain": C,
        })

    # Long symmetric butterflies — profit if K1+cost < S_T < K3-cost
    sks = sorted(Cs.keys())
    for i, K1 in enumerate(sks):
        for j in range(i + 1, len(sks)):
            K2 = sks[j]
            for k in range(j + 1, len(sks)):
                K3 = sks[k]
                if K2 - K1 != K3 - K2:
                    continue
                w = K2 - K1
                cost = Cs[K1] - 2.0 * Cs[K2] + Cs[K3]
                if cost <= 0:
                    p = 1.0  # can't lose at expiry
                elif cost >= w:
                    p = 0.0  # paid more than max payoff
                else:
                    lo = K1 + cost
                    hi = K3 - cost
                    p = max(_lognormal_cdf(hi, S, sd) - _lognormal_cdf(lo, S, sd), 0.0)
                rows.append({
                    "name": f"long fly {K1}/{K2}/{K3}",
                    "type": "long_fly",
                    "cost": cost,
                    "p_profit": p,
                    "max_loss": cost,
                    "max_gain": max(w - cost, 0.0),
                })

    if not rows:
        return _empty("No structures could be priced")

    df = pd.DataFrame(rows).sort_values("p_profit", ascending=True).reset_index(drop=True)
    colors = {
        "long_call": "#377eb8",
        "short_call": "#e41a1c",
        "long_fly": "#4daf4a",
    }
    fig = go.Figure()
    for t, group in df.groupby("type", sort=False):
        fig.add_trace(go.Bar(
            y=group["name"], x=group["p_profit"],
            orientation="h", name=t.replace("_", " "),
            marker_color=colors.get(t, "#888"),
            customdata=np.stack([group["cost"], group["max_gain"], group["max_loss"]], axis=-1),
            hovertemplate=(
                "%{y}<br>P(profit)=%{x:.1%}<br>"
                "cost=%{customdata[0]:+.2f}<br>"
                "max gain=%{customdata[1]:.2f}<br>"
                "max loss=%{customdata[2]:.2f}<extra></extra>"
            ),
        ))
    fig.add_vline(x=0.5, line={"color": "#999", "width": 1, "dash": "dot"})
    fig.update_layout(
        title=f"P(profit at expiry) · σ_R={sigma:.4f} · TTE_remaining={T:.2f}d · S={S:.1f}",
        xaxis_title="P(profit)",
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        yaxis_title="",
        template="plotly_white",
        margin={"l": 130, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.15},
        barmode="stack",
    )
    return fig


# ---------------------------------------------------------------------------
# Cumulative gamma-PnL vs theta-paid tracker
# ---------------------------------------------------------------------------

def _position_series(store_module, K, day, timestamps):
    """Position-over-time aligned to `timestamps`. Uses real user trades when
    the store has a `side` column with buy/sell entries (live mode); otherwise
    returns a hypothetical 'long 1 voucher held from t=0' series."""
    default_pos = np.ones(len(timestamps), dtype=float)
    is_hypothetical = True
    try:
        trades = store_module.get_trades(f"VEV_{K}", day)
    except Exception:
        return default_pos, is_hypothetical
    if trades is None or trades.empty or "side" not in trades.columns:
        return default_pos, is_hypothetical
    user_trades = trades[trades["side"].isin(["buy", "sell"])]
    if user_trades.empty:
        return default_pos, is_hypothetical

    user_trades = user_trades.sort_values("timestamp")
    pos = np.zeros(len(timestamps), dtype=float)
    cur = 0.0
    ti = 0
    for _, row in user_trades.iterrows():
        ts = int(row["timestamp"])
        side = row["side"]
        qty = float(row["quantity"])
        signed = qty if side == "buy" else -qty
        while ti < len(timestamps) and timestamps[ti] < ts:
            pos[ti] = cur
            ti += 1
        cur += signed
    while ti < len(timestamps):
        pos[ti] = cur
        ti += 1
    return pos, False


def build_gamma_pnl_tracker_figure(store_module, strikes, day, sigma_r,
                                    tte_at_start, downsample=1):
    """Cumulative Γ-PnL vs Θ paid for the selected strikes.

    Live mode: uses actual user voucher trades to derive position(t) when
    available. Otherwise (or for historical data) falls back to a hypothetical
    'long 1 voucher of each selected strike from t=0'. Greeks evaluated at σ_R.
    Step PnL is computed over each underlying mid update using
        gamma_step = ½·Γ(t)·(ΔS)²·pos(t)
        theta_step = Θ(t)·Δt·pos(t)
    and summed across selected strikes.
    """
    if not store_module.is_loaded():
        return _empty("Load data first")
    if not strikes:
        return _empty("Select one or more voucher strikes")

    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    ve = ve.dropna(subset=["mid_price"]).sort_values("timestamp").reset_index(drop=True)
    ve = _downsample(ve, downsample)
    if len(ve) < 2:
        return _empty("Not enough underlying observations")

    timestamps = ve["timestamp"].to_numpy()
    S_arr = ve["mid_price"].to_numpy()
    T_arr = opts.time_to_expiry(timestamps, float(tte_at_start))
    sigma = float(sigma_r) if sigma_r else 0.0
    if sigma <= 0:
        return _empty("σ_R must be positive")

    cum_gamma = np.zeros(len(timestamps))
    cum_theta = np.zeros(len(timestamps))
    n_traced = 0
    used_hypothetical = []
    used_real = []

    for K in strikes:
        try:
            K_int = int(K)
        except (TypeError, ValueError):
            continue
        pos, is_hypo = _position_series(store_module, K_int, day, timestamps)
        if is_hypo:
            used_hypothetical.append(K_int)
        else:
            used_real.append(K_int)
        K_arr = np.full(len(timestamps), float(K_int))
        sig_arr = np.full(len(timestamps), sigma)
        g = opts.greeks(S_arr, K_arr, T_arr, sig_arr)
        gamma = g["gamma"]
        theta = g["theta"]
        dS = np.diff(S_arr)
        dt = np.diff(timestamps) / opts.TIMESTAMP_PER_DAY
        gamma_step = 0.5 * gamma[:-1] * dS * dS * pos[:-1]
        theta_step = theta[:-1] * dt * pos[:-1]
        cum_gamma[1:] += np.cumsum(gamma_step)
        cum_theta[1:] += np.cumsum(theta_step)
        n_traced += 1

    if n_traced == 0:
        return _empty("No strikes available to trace")

    net = cum_gamma + cum_theta

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_gamma, mode="lines",
        name="Cumulative Γ-PnL (gain from realized moves)",
        line={"color": "#4daf4a", "width": 2.4},
        hovertemplate="t=%{x}<br>Γ-PnL=%{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_theta, mode="lines",
        name="Cumulative Θ paid (negative = cost of long-vol)",
        line={"color": "#e41a1c", "width": 2.4},
        hovertemplate="t=%{x}<br>Θ-PnL=%{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=net, mode="lines",
        name="Net (Γ + Θ)",
        line={"color": "black", "width": 2.4, "dash": "dash"},
        hovertemplate="t=%{x}<br>net=%{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": "#999", "width": 1})

    if used_real and not used_hypothetical:
        position_note = f"actual user trades · strikes {used_real}"
    elif used_real and used_hypothetical:
        position_note = (f"actual: {used_real} · hypothetical (+1 from t=0): "
                         f"{used_hypothetical}")
    else:
        position_note = f"hypothetical +1 from t=0 · strikes {used_hypothetical}"

    fig.update_layout(
        title=f"Cumulative Γ-PnL vs Θ paid · σ_R={sigma:.4f} · {position_note}",
        xaxis_title="Timestamp",
        yaxis_title="PnL (in S currency)",
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
    )
    return fig
