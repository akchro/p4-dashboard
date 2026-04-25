"""Store-agnostic figure builders for the options panels.

Each `build_*` function takes a store module (either `data.store` or
`data.historical_store`) and returns a Plotly figure. Live/historical panel
modules wire these up with their own IDs and callbacks.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

    x is either log-moneyness m = ln(K/S)/√T or raw strike K. Each strike gets
    its own color (shared with overlay / IV-TS). Optional quadratic fit is one
    parabola through the combined cloud.
    """
    if not strikes or not store_module.is_loaded():
        return _empty("Select one or more voucher strikes")

    ve = store_module.get_activities(opts.UNDERLYING, day)
    if ve.empty:
        return _empty(f"No {opts.UNDERLYING} data for day {day}")
    ve_ds = _downsample(ve, downsample)

    fig = go.Figure()
    all_x = []
    all_y = []

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
        iv = opts.implied_vol(merged["C"].values, merged["S"].values, K_arr, T)
        m = opts.log_moneyness(K_arr, merged["S"].values, T)
        valid = np.isfinite(iv)
        if not valid.any():
            continue

        xs = m[valid] if x_axis == "moneyness" else K_arr[valid]
        ys = iv[valid]
        ts = merged["timestamp"].values[valid]
        all_x.append(xs)
        all_y.append(ys)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"VEV_{K_int}",
            marker={"color": strike_color(K_int), "size": 4, "opacity": 0.55},
            customdata=ts,
            hovertemplate=(
                f"K={K_int}<br>"
                "t=%{customdata:,}<br>"
                "x=%{x:.4f}<br>"
                "IV=%{y:.5f}<extra></extra>"
            ),
        ))

    if not all_x:
        return _empty("No solvable IVs across selected strikes")

    if show_fit:
        xs_all = np.concatenate(all_x)
        ys_all = np.concatenate(all_y)
        if len(xs_all) >= 3:
            try:
                coef = np.polyfit(xs_all, ys_all, 2)
                xfit = np.linspace(xs_all.min(), xs_all.max(), 80)
                yfit = np.polyval(coef, xfit)
                fig.add_trace(go.Scatter(
                    x=xfit, y=yfit, mode="lines",
                    name=f"fit: {coef[0]:.4g}x² + {coef[1]:.4g}x + {coef[2]:.4g}",
                    line={"color": "#000000", "width": 2.5},
                ))
            except (np.linalg.LinAlgError, ValueError):
                pass

    x_label = "log-moneyness m = ln(K/S)/√T" if x_axis == "moneyness" else "Strike K"
    fig.update_layout(
        title="Volatility Smile · full day (color = strike)",
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
