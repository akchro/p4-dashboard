import plotly.graph_objects as go
from dash import html, dcc, Input, Output
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
    "buy":    {"color": "green",  "symbol": "triangle-up",   "name": "Buy"},
    "sell":   {"color": "red",    "symbol": "triangle-down", "name": "Sell"},
    "market": {"color": "gray",   "symbol": "circle",        "name": "Market"},
}


def layout():
    return html.Div([
        dcc.Graph(id="main-chart", style={"height": "100%"}),
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
