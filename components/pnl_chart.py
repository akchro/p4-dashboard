import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div([
        dcc.Graph(id="pnl-chart", style={"height": "calc(100% - 28px)"}),
        html.Div(
            id="pnl-range-stats",
            children="Zoom into a range to see variance & max drawdown",
            style={
                "fontSize": "11px", "padding": "2px 10px",
                "color": "#666", "height": "24px", "lineHeight": "24px",
                "background": "#f8f9fa", "borderTop": "1px solid #eee",
            },
        ),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("pnl-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value"),
         Input("downsample-slider", "value")],
    )
    def update_pnl(product, day, downsample):
        if not product or not store.is_loaded():
            raise PreventUpdate

        acts = store.get_activities(product, day)
        if downsample and downsample > 1:
            acts = acts.iloc[::downsample]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acts["timestamp"], y=acts["profit_and_loss"],
            mode="lines", name="PnL",
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
        [State("product-selector", "value"),
         State("day-selector", "value")],
    )
    def pnl_range_stats(relayout, product, day):
        if not relayout or not product or not store.is_loaded():
            raise PreventUpdate

        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")

        if relayout.get("xaxis.autorange") or x0 is None or x1 is None:
            return "Zoom into a range to see variance & max drawdown"

        acts = store.get_activities(product, day)
        ts = pd.to_datetime(acts["timestamp"])
        x0, x1 = pd.to_datetime(x0), pd.to_datetime(x1)
        ys = acts.loc[(ts >= x0) & (ts <= x1), "profit_and_loss"].dropna().values

        if len(ys) < 2:
            return "Zoom range has fewer than 2 points"

        variance = float(np.var(ys))

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

        return (
            f"{len(ys)} pts | "
            f"Variance: {variance:,.2f} | "
            f"Max Drawdown: {max_dd:,.2f} ({pct:.1f}%)"
        )

    @app.callback(
        Output("pnl-overall-stats", "children"),
        [Input("product-selector", "value"),
         Input("day-selector", "value")],
    )
    def pnl_overall_stats(product, day):
        if not product or not store.is_loaded():
            raise PreventUpdate

        acts = store.get_activities(product, day)
        ys = acts["profit_and_loss"].dropna().values

        if len(ys) < 2:
            return "Not enough PnL data"

        variance = float(np.var(ys))

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

        return html.Div([
            html.Div("PnL Stats", style={"fontWeight": "bold", "marginBottom": "4px"}),
            html.Div(f"Variance: {variance:,.2f}"),
            html.Div(f"Max Drawdown: {max_dd:,.2f} ({pct:.1f}%)"),
        ])
