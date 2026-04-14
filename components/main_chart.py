import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from data import store

BID_COLORS = ["#0000FF", "#4444FF", "#8888FF"]
ASK_COLORS = ["#FF0000", "#FF4444", "#FF8888"]

LEVEL_MAP = {
    "bid_1": ("bid_price_1", "Bid 1", BID_COLORS[0]),
    "bid_2": ("bid_price_2", "Bid 2", BID_COLORS[1]),
    "bid_3": ("bid_price_3", "Bid 3", BID_COLORS[2]),
    "ask_1": ("ask_price_1", "Ask 1", ASK_COLORS[0]),
    "ask_2": ("ask_price_2", "Ask 2", ASK_COLORS[1]),
    "ask_3": ("ask_price_3", "Ask 3", ASK_COLORS[2]),
}

TRADE_STYLES = {
    "buy":    {"color": "orange", "symbol": "triangle-up",   "name": "Buy"},
    "sell":   {"color": "cyan",   "symbol": "triangle-down", "name": "Sell"},
    "market": {"color": "yellow", "symbol": "circle",        "name": "Market"},
}


def layout():
    return html.Div([
        dcc.Graph(id="main-chart", style={"height": "calc(100% - 28px)"}),
        html.Div(
            id="main-range-stats",
            children="Zoom into a range to see mid-price variance & max drawdown",
            style={
                "fontSize": "11px", "padding": "2px 10px",
                "color": "#666", "height": "24px", "lineHeight": "24px",
                "background": "#f8f9fa", "borderTop": "1px solid #eee",
            },
        ),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("main-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value"),
         Input("downsample-slider", "value"),
         Input("level-toggles", "value"),
         Input("trade-toggle", "value"),
         Input("qty-filter", "value"),
         Input("wallmid-toggle", "value")],
    )
    def update_main_chart(product, day, downsample, levels, trade_toggle, qty_range, wallmid_toggle):
        if not product or not store.is_loaded():
            raise PreventUpdate

        fig = go.Figure()
        acts = store.get_activities(product, day)
        if downsample and downsample > 1:
            acts = acts.iloc[::downsample]

        # Mid price
        fig.add_trace(go.Scatter(
            x=acts["timestamp"], y=acts["mid_price"],
            mode="lines", name="Mid",
            line={"color": "black", "width": 2.5},
            connectgaps=False,
        ))

        # Wallmid overlay (optional — only if data exists and toggle is on)
        if wallmid_toggle and "show" in wallmid_toggle and "wallmid" in acts.columns:
            wm = acts.dropna(subset=["wallmid"])
            if not wm.empty:
                fig.add_trace(go.Scatter(
                    x=wm["timestamp"], y=wm["wallmid"],
                    mode="lines", name="Wallmid",
                    line={"color": "#FF00FF", "width": 2},
                    connectgaps=False,
                ))

        # Order book levels
        if levels:
            for lvl in levels:
                col, name, color = LEVEL_MAP[lvl]
                if col in acts.columns:
                    fig.add_trace(go.Scatter(
                        x=acts["timestamp"], y=acts[col],
                        mode="lines", name=name,
                        line={"color": color, "width": 2},
                        connectgaps=False,
                    ))

        # Trade markers
        if trade_toggle and "show" in trade_toggle:
            trades = store.get_trades(product, day)
            if not trades.empty:
                if qty_range:
                    trades = trades[
                        (trades["quantity"] >= qty_range[0]) &
                        (trades["quantity"] <= qty_range[1])
                    ]
                for side, style in TRADE_STYLES.items():
                    side_trades = trades[trades["side"] == side]
                    if side_trades.empty:
                        continue
                    buyer = side_trades["buyer"].fillna("-").replace("", "-")
                    seller = side_trades["seller"].fillna("-").replace("", "-")
                    fig.add_trace(go.Scatter(
                        x=side_trades["timestamp"],
                        y=side_trades["price"],
                        mode="markers",
                        name=style["name"],
                        marker={
                            "color": style["color"],
                            "symbol": style["symbol"],
                            "size": 8,
                            "line": {"width": 1, "color": "black"},
                        },
                        hovertemplate=(
                            "t=%{x}<br>price=%{y}<br>"
                            "qty=%{customdata[0]}<br>"
                            "buyer=%{customdata[1]}<br>"
                            "seller=%{customdata[2]}"
                            "<extra></extra>"
                        ),
                        customdata=list(zip(
                            side_trades["quantity"],
                            buyer,
                            seller,
                        )),
                    ))

        fig.update_layout(
            title=f"{product} Order Book",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            hovermode="x unified",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            legend={"orientation": "h", "y": -0.15},
            template="plotly_white",
        )
        return fig

    @app.callback(
        Output("main-range-stats", "children"),
        Input("main-chart", "relayoutData"),
        [State("product-selector", "value"),
         State("day-selector", "value")],
    )
    def main_range_stats(relayout, product, day):
        if not relayout or not product or not store.is_loaded():
            raise PreventUpdate

        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")

        if relayout.get("xaxis.autorange") or x0 is None or x1 is None:
            return "Zoom into a range to see mid-price variance & max drawdown"

        acts = store.get_activities(product, day)
        ts = pd.to_datetime(acts["timestamp"])
        x0, x1 = pd.to_datetime(x0), pd.to_datetime(x1)
        ys = acts.loc[(ts >= x0) & (ts <= x1), "mid_price"].dropna().values

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
            f"Variance: {variance:,.4f} | "
            f"Max Drawdown: {max_dd:,.4f} ({pct:.1f}%)"
        )
