"""Round 3 — bid-velocity sub-tab.

Shows bid volume / velocity / cross-strike imbalance signal on the OTM voucher
strikes (default VEV_5300/5400/5500). The aggregate sum(bid_vol_1 − ask_vol_1)
on these strikes leads forward returns on every other voucher (verified
+0.05 to +0.55 spread between sig>0 and sig<0 fwd returns, all 3 days).

Reuses `hist-day-selector` and `hist-downsample-slider` from the Historical
sidebar; all other controls are local to this sub-tab.
"""
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import historical_store


CELL = {"border": "1px solid #ddd", "borderRadius": "4px"}

DEFAULT_STRIKES = [5300, 5400, 5500]
ALL_OTM = [5300, 5400, 5500, 6000, 6500]
ALL_VOUCHERS = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDERLYING = "VELVETFRUIT_EXTRACT"

# Distinct colors per strike for line consistency across charts
STRIKE_COLORS = {
    5300: "#1f77b4",
    5400: "#ff7f0e",
    5500: "#2ca02c",
    6000: "#d62728",
    6500: "#9467bd",
}

# Palette for overlay traces (cycles if more than 8 selected)
OVERLAY_PALETTE = ["#000000", "#7f7f7f", "#8c564b", "#e377c2",
                   "#17becf", "#bcbd22", "#9467bd", "#aec7e8"]


def _controls_row():
    overlay_options = [{"label": "VFE (underlying)", "value": UNDERLYING}] + [
        {"label": f"VEV_{k}", "value": f"VEV_{k}"} for k in ALL_VOUCHERS
    ]
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1.3fr 0.6fr 0.6fr 0.6fr 0.8fr 1.3fr",
        "gap": "10px",
        "padding": "8px",
        "borderBottom": "1px solid #ddd",
        "alignItems": "center",
        "fontSize": "12px",
    }, children=[
        html.Div([
            html.Label("Strikes (5300+)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="r3bv-strike-multi",
                options=[{"label": f"VEV_{k}", "value": k} for k in ALL_OTM],
                value=list(DEFAULT_STRIKES),
                multi=True,
                clearable=False,
            ),
        ]),
        html.Div([
            html.Label("Smooth window (ticks)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="r3bv-smooth-window", type="number",
                value=20, min=1, step=1,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Velocity lookback (ticks)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="r3bv-velocity-lag", type="number",
                value=20, min=1, step=1,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Fwd return k (ticks)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="r3bv-fwd-k", type="number",
                value=10, min=1, max=500, step=1,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Aggregate target", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="r3bv-target",
                options=[
                    {"label": "VFE", "value": "VELVETFRUIT_EXTRACT"},
                    {"label": "VEV_5000", "value": "VEV_5000"},
                    {"label": "VEV_5100", "value": "VEV_5100"},
                    {"label": "VEV_5300", "value": "VEV_5300"},
                ],
                value="VEV_5000",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "6px"},
            ),
        ]),
        html.Div([
            html.Label("Price overlay (normalized to t=0)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="r3bv-velocity-overlay",
                options=overlay_options,
                value=[],
                multi=True,
                placeholder="Pick products to overlay…",
            ),
        ]),
    ])


def layout():
    return html.Div(style={
        "padding": "4px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "2px",
    }, children=[
        _controls_row(),
        html.Div(style={"padding": "4px 10px 0", "fontSize": "11px", "color": "#666"},
                 children=("Top: bid_volume_1 (smoothed) per strike. "
                           "Middle: bid velocity = Δsmoothed_bid_vol over the lookback "
                           "(positive = bids growing → bullish; negative = bids shrinking → bearish). "
                           "Bottom-left: aggregate sum(b1−a1) across selected strikes "
                           "vs target product mid. Bottom-right: mean fwd return of target product, "
                           "bucketed by aggregate-signal sign — confirms predictive direction.")),
        html.Div(style={**CELL, "height": "30vh"}, children=[
            dcc.Graph(id="r3bv-volume-chart", style={"height": "100%"}),
        ]),
        html.Div(style={**CELL, "height": "30vh"}, children=[
            dcc.Graph(id="r3bv-velocity-chart", style={"height": "100%"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "2fr 1fr",
            "gap": "2px",
            "height": "32vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="r3bv-signal-vs-target", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="r3bv-signal-fwdret", style={"height": "100%"}),
            ]),
        ]),
    ])


def _load_strike(strike: int, day: int):
    """Return DataFrame indexed by timestamp with bid_volume_1, ask_volume_1, mid_price.
    Empty if missing."""
    df = historical_store.get_activities(f"VEV_{strike}", day)
    if df.empty: return df
    df = df.sort_values("timestamp").set_index("timestamp")
    for c in ["bid_volume_1", "ask_volume_1", "bid_price_1", "ask_price_1", "mid_price"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def _load_target(product: str, day: int):
    df = historical_store.get_activities(product, day)
    if df.empty: return df
    return df.sort_values("timestamp").set_index("timestamp")


def _empty_fig(text: str = "Load historical data to view") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{"text": text, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False,
                       "font": {"size": 14, "color": "#888"}}],
        margin={"l": 30, "r": 10, "t": 30, "b": 30},
        plot_bgcolor="white",
    )
    return fig


def _build_volume(strikes, day, smooth_w, overlay=None):
    if not strikes or day is None:
        return _empty_fig()
    overlay = list(overlay or [])
    use_secondary = len(overlay) > 0
    if use_secondary:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    smooth_w = max(int(smooth_w or 1), 1)
    for k in strikes:
        df = _load_strike(k, day)
        if df.empty or "bid_volume_1" not in df.columns: continue
        sm = df["bid_volume_1"].rolling(smooth_w, min_periods=1).mean()
        color = STRIKE_COLORS.get(k, "#444")
        trace = go.Scatter(
            x=sm.index, y=sm.values, mode="lines",
            name=f"VEV_{k} bid_vol_1 (sm{smooth_w})",
            line={"color": color, "width": 1.5},
        )
        if use_secondary:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)

    # Normalized price overlays — raw mid_price minus its first observation,
    # so every overlay starts at 0 at the first available timestamp. Rendered
    # on a secondary y-axis, independent of the bid-volume y-scale.
    for i, prod in enumerate(overlay):
        odf = _load_target(prod, day)
        if odf.empty or "mid_price" not in odf.columns: continue
        mid = odf["mid_price"].astype(float).dropna()
        if mid.empty: continue
        norm = mid - mid.iloc[0]
        color = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, mode="lines",
            name=f"{prod} Δmid (norm)",
            line={"color": color, "width": 1.2, "dash": "dot"},
        ), secondary_y=True)

    fig.update_layout(
        title=f"Bid volume (level 1) — Day {day}, smoothed over {smooth_w} ticks"
              + ("  +  normalized price overlay" if use_secondary else ""),
        xaxis_title="timestamp",
        margin={"l": 50, "r": 50 if use_secondary else 10, "t": 40, "b": 40},
        plot_bgcolor="white", legend={"orientation": "h", "y": -0.15},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    if use_secondary:
        fig.update_yaxes(title_text="bid_volume_1", secondary_y=False,
                         showgrid=True, gridcolor="#eee")
        fig.update_yaxes(title_text="Δmid (from t=0)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="bid_volume_1", showgrid=True, gridcolor="#eee")
    return fig


def _build_velocity(strikes, day, smooth_w, lag, overlay=None):
    if not strikes or day is None:
        return _empty_fig()
    overlay = list(overlay or [])
    use_secondary = len(overlay) > 0
    if use_secondary:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    smooth_w = max(int(smooth_w or 1), 1)
    lag = max(int(lag or 1), 1)
    for k in strikes:
        df = _load_strike(k, day)
        if df.empty or "bid_volume_1" not in df.columns: continue
        sm = df["bid_volume_1"].rolling(smooth_w, min_periods=1).mean()
        v = sm - sm.shift(lag)
        color = STRIKE_COLORS.get(k, "#444")
        trace = go.Scatter(
            x=v.index, y=v.values, mode="lines",
            name=f"VEV_{k} velocity",
            line={"color": color, "width": 1.5},
        )
        if use_secondary:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)

    # Normalized price overlays — raw mid_price minus its first observation,
    # so every overlay starts at 0 at the first available timestamp. Rendered
    # on a secondary y-axis, independent of the velocity y-scale.
    for i, prod in enumerate(overlay):
        odf = _load_target(prod, day)
        if odf.empty or "mid_price" not in odf.columns: continue
        mid = odf["mid_price"].astype(float).dropna()
        if mid.empty: continue
        norm = mid - mid.iloc[0]
        color = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, mode="lines",
            name=f"{prod} Δmid (norm)",
            line={"color": color, "width": 1.2, "dash": "dot"},
        ), secondary_y=True)

    fig.add_hline(y=0, line_color="#999", line_dash="dash", line_width=1,
                  **({"secondary_y": False} if use_secondary else {}))
    fig.update_layout(
        title=f"Bid velocity (Δsmoothed bid_vol over {lag}-tick lookback)"
              + ("  +  normalized price overlay" if use_secondary else ""),
        xaxis_title="timestamp",
        margin={"l": 50, "r": 50 if use_secondary else 10, "t": 40, "b": 40},
        plot_bgcolor="white", legend={"orientation": "h", "y": -0.15},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    if use_secondary:
        fig.update_yaxes(title_text="Δbid_volume", secondary_y=False,
                         showgrid=True, gridcolor="#eee")
        fig.update_yaxes(title_text="Δmid (from t=0)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Δbid_volume", showgrid=True, gridcolor="#eee")
    return fig


def _aggregate_signal(strikes, day):
    """Per-tick sum(bid_vol_1 - ask_vol_1) across strikes.
    Returns pd.Series indexed by timestamp."""
    series = None
    for k in strikes:
        df = _load_strike(k, day)
        if df.empty: continue
        if "bid_volume_1" not in df.columns or "ask_volume_1" not in df.columns: continue
        s = df["bid_volume_1"] - df["ask_volume_1"]
        series = s if series is None else series.add(s, fill_value=0)
    return series


def _build_signal_vs_target(strikes, day, target):
    if not strikes or day is None:
        return _empty_fig()
    sig = _aggregate_signal(strikes, day)
    if sig is None or sig.empty:
        return _empty_fig("No data for selected strikes")
    tgt_df = _load_target(target, day)
    if tgt_df.empty or "mid_price" not in tgt_df.columns:
        return _empty_fig(f"No mid_price for {target}")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig.values, mode="lines",
        name=f"Σ(b1−a1) across {len(strikes)} strikes",
        line={"color": "#1f77b4", "width": 1},
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tgt_df.index, y=tgt_df["mid_price"].values, mode="lines",
        name=f"{target} mid",
        line={"color": "#d62728", "width": 1.2},
    ), secondary_y=True)
    fig.add_hline(y=0, line_color="#999", line_dash="dash", line_width=1, secondary_y=False)
    fig.update_layout(
        title=f"Aggregate OTM signal vs {target} mid — Day {day}",
        xaxis_title="timestamp",
        margin={"l": 50, "r": 50, "t": 40, "b": 40},
        plot_bgcolor="white", legend={"orientation": "h", "y": -0.15},
    )
    fig.update_yaxes(title_text="signal", secondary_y=False, showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text=f"{target} mid", secondary_y=True)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    return fig


def _build_signal_fwdret(strikes, day, target, k):
    if not strikes or day is None:
        return _empty_fig()
    sig = _aggregate_signal(strikes, day)
    if sig is None or sig.empty: return _empty_fig("No signal")
    tgt_df = _load_target(target, day)
    if tgt_df.empty or "mid_price" not in tgt_df.columns:
        return _empty_fig(f"No mid for {target}")
    k = max(int(k or 1), 1)
    mid = tgt_df["mid_price"].astype(float)
    # Align indices
    common = sig.index.intersection(mid.index)
    s = sig.loc[common]
    m = mid.loc[common]
    fwd = m.shift(-k) - m
    sig_arr = np.asarray(s.values, dtype=float)
    fwd_arr = np.asarray(fwd.values, dtype=float)
    mask = ~np.isnan(fwd_arr)
    sig_arr, fwd_arr = sig_arr[mask], fwd_arr[mask]
    # Three buckets: sig<0, sig==0, sig>0
    buckets = []
    for lab, sel in [("sig < 0", sig_arr < 0), ("sig = 0", sig_arr == 0), ("sig > 0", sig_arr > 0)]:
        if sel.sum() == 0:
            buckets.append((lab, 0, np.nan, np.nan))
            continue
        buckets.append((lab, int(sel.sum()), float(fwd_arr[sel].mean()),
                        float(fwd_arr[sel].std())))
    fig = go.Figure()
    labels = [b[0] for b in buckets]
    means  = [b[2] for b in buckets]
    counts = [b[1] for b in buckets]
    colors = ["#d62728", "#888888", "#2ca02c"]
    fig.add_trace(go.Bar(
        x=labels, y=means, marker_color=colors,
        text=[f"n={c}<br>μ={m:+.3f}" for c, m in zip(counts, means)],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="#999", line_dash="dash", line_width=1)
    fig.update_layout(
        title=f"Mean {target} fwd return at k={k} ticks",
        yaxis_title="mean fwd return (ticks)",
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        plot_bgcolor="white", showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def register_callbacks(app):
    @app.callback(
        Output("r3bv-volume-chart", "figure"),
        [Input("r3bv-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("r3bv-smooth-window", "value"),
         Input("r3bv-velocity-overlay", "value")],
    )
    def _vol(strikes, day, smooth, overlay):
        if not historical_store.is_loaded(): raise PreventUpdate
        return _build_volume(strikes, day, smooth, overlay)

    @app.callback(
        Output("r3bv-velocity-chart", "figure"),
        [Input("r3bv-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("r3bv-smooth-window", "value"),
         Input("r3bv-velocity-lag", "value"),
         Input("r3bv-velocity-overlay", "value")],
    )
    def _vel(strikes, day, smooth, lag, overlay):
        if not historical_store.is_loaded(): raise PreventUpdate
        return _build_velocity(strikes, day, smooth, lag, overlay)

    @app.callback(
        Output("r3bv-signal-vs-target", "figure"),
        [Input("r3bv-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("r3bv-target", "value")],
    )
    def _sig_vs_tgt(strikes, day, target):
        if not historical_store.is_loaded(): raise PreventUpdate
        return _build_signal_vs_target(strikes, day, target)

    @app.callback(
        Output("r3bv-signal-fwdret", "figure"),
        [Input("r3bv-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("r3bv-target", "value"),
         Input("r3bv-fwd-k", "value")],
    )
    def _sig_fwd(strikes, day, target, k):
        if not historical_store.is_loaded(): raise PreventUpdate
        return _build_signal_fwdret(strikes, day, target, k)
