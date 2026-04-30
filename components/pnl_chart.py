import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div(style={
        "height": "100%",
        "display": "grid",
        "gridTemplateColumns": "1fr 170px",
    }, children=[
        html.Div(style={"height": "100%"}, children=[
            dcc.Graph(id="pnl-chart", style={"height": "calc(100% - 28px)"}),
            html.Div(
                id="pnl-range-stats",
                children="Zoom into a range to see std dev, CV & max drawdown",
                style={
                    "fontSize": "11px", "padding": "2px 10px",
                    "color": "#666", "height": "24px", "lineHeight": "24px",
                    "background": "#f8f9fa", "borderTop": "1px solid #eee",
                },
            ),
        ]),
        html.Div(id="pnl-toggles-panel", style={
            "padding": "6px 8px",
            "borderLeft": "1px solid #eee",
            "overflowY": "auto",
        }, children=[
            html.Label("PnL Products", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(
                id="pnl-product-toggles",
                options=[],
                value=[],
                style={"fontSize": "11px", "marginTop": "4px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"display": "block", "marginBottom": "2px"},
            ),
        ]),
    ])


def _stats(ys):
    if len(ys) < 2:
        return None
    changes = np.diff(ys)
    chg_std = float(np.std(changes))
    chg_mean = float(np.mean(changes))
    chg_cv = (chg_std / abs(chg_mean) * 100) if chg_mean != 0 else 0

    peak = ys[0]
    max_dd = 0.0
    dd_peak = peak
    for y in ys:
        if y > peak:
            peak = y
        dd = peak - y
        if dd > max_dd:
            max_dd = dd
            dd_peak = peak
    pct = (max_dd / abs(dd_peak) * 100) if dd_peak != 0 else 0
    return chg_std, chg_cv, max_dd, pct


def register_callbacks(app):
    @app.callback(
        [Output("pnl-product-toggles", "options"),
         Output("pnl-product-toggles", "value")],
        [Input("product-selector", "options"),
         Input("product-selector", "value")],
    )
    def sync_toggles(options, current_product):
        if not options or not current_product:
            raise PreventUpdate
        return options, [current_product]

    @app.callback(
        Output("pnl-chart", "figure"),
        [Input("pnl-product-toggles", "value"),
         Input("day-selector", "value"),
         Input("downsample-slider", "value")],
    )
    def update_pnl(products, day, downsample):
        if not products or not store.is_loaded():
            return go.Figure()

        pivot = store.get_pnl_pivot(products, day)
        if pivot.empty:
            return go.Figure()

        # Engine writes cumulative PnL across all days; reset each column to
        # start at 0 within the selected day so the chart shows that day's
        # contribution rather than the running total.
        pivot = pivot - pivot.bfill().iloc[0]

        if downsample and downsample > 1:
            pivot = pivot.iloc[::downsample]

        fig = go.Figure()
        for product in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[product],
                mode="lines", name=product,
                line={"width": 1.2},
            ))
        if len(pivot.columns) > 1:
            total = pivot.sum(axis=1)
            fig.add_trace(go.Scatter(
                x=pivot.index, y=total,
                mode="lines", name="Total",
                line={"color": "black", "width": 2.5},
            ))

        fig.update_layout(
            title="Profit & Loss",
            xaxis_title="Timestamp",
            yaxis_title="PnL",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    @app.callback(
        Output("pnl-range-stats", "children"),
        Input("pnl-chart", "relayoutData"),
        [State("pnl-product-toggles", "value"),
         State("day-selector", "value")],
    )
    def pnl_range_stats(relayout, products, day):
        if not relayout or not products or not store.is_loaded():
            raise PreventUpdate

        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        if relayout.get("xaxis.autorange") or x0 is None or x1 is None:
            return "Zoom into a range to see std dev, CV & max drawdown"

        pivot = store.get_pnl_pivot(products, day)
        if pivot.empty:
            return "No PnL data"
        total = pivot.sum(axis=1)
        ts = pd.to_datetime(total.index)
        x0, x1 = pd.to_datetime(x0), pd.to_datetime(x1)
        ys = total.values[(ts >= x0) & (ts <= x1)]
        ys = ys[~pd.isna(ys)]

        s = _stats(ys)
        if s is None:
            return "Zoom range has fewer than 2 points"
        chg_std, chg_cv, max_dd, pct = s
        return (
            f"{len(ys)} pts | "
            f"ΔPnL Std Dev: {chg_std:,.2f} | ΔPnL CV: {chg_cv:.2f}% | "
            f"Max Drawdown: {max_dd:,.2f} ({pct:.1f}%)"
        )

    @app.callback(
        Output("pnl-overall-stats", "children"),
        [Input("pnl-product-toggles", "value"),
         Input("day-selector", "value")],
    )
    def pnl_overall_stats(products, day):
        if not products or not store.is_loaded():
            raise PreventUpdate

        pivot = store.get_pnl_pivot(products, day)
        if pivot.empty:
            raise PreventUpdate
        total = pivot.sum(axis=1)
        ys = total.dropna().values

        s = _stats(ys)
        if s is None:
            return "Not enough PnL data"
        chg_std, chg_cv, max_dd, pct = s

        label = "Total" if len(products) > 1 else products[0]
        return html.Div([
            html.Div(f"PnL Stats — {label}", style={"fontWeight": "bold", "marginBottom": "4px"}),
            html.Div(f"ΔPnL Std Dev: {chg_std:,.2f} | ΔPnL CV: {chg_cv:.2f}%"),
            html.Div(f"Max Drawdown: {max_dd:,.2f} ({pct:.1f}%)"),
        ])
