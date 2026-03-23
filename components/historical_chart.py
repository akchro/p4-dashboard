import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import historical_store

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


def layout():
    return html.Div([
        dcc.Graph(id="hist-main-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("hist-main-chart", "figure"),
        [Input("hist-product-selector", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-downsample-slider", "value"),
         Input("hist-level-toggles", "value"),
         Input("hist-trade-toggle", "value"),
         Input("hist-qty-filter", "value")],
    )
    def update_hist_chart(product, day, downsample, levels, trade_toggle, qty_range):
        if not product or not historical_store.is_loaded():
            raise PreventUpdate

        fig = go.Figure()
        acts = historical_store.get_activities(product, day)
        if downsample and downsample > 1:
            acts = acts.iloc[::downsample]

        # Mid price
        fig.add_trace(go.Scatter(
            x=acts["timestamp"], y=acts["mid_price"],
            mode="lines", name="Mid",
            line={"color": "black", "width": 1.5},
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
                        line={"color": color, "width": 1},
                        connectgaps=False,
                    ))

        # Trade markers (all shown as gray circles — no buy/sell distinction)
        if trade_toggle and "show" in trade_toggle:
            trades = historical_store.get_trades(product, day)
            if not trades.empty:
                if qty_range:
                    trades = trades[
                        (trades["quantity"] >= qty_range[0]) &
                        (trades["quantity"] <= qty_range[1])
                    ]
                if not trades.empty:
                    fig.add_trace(go.Scatter(
                        x=trades["timestamp"],
                        y=trades["price"],
                        mode="markers",
                        name="Trade",
                        marker={"color": "gray", "symbol": "circle", "size": 8},
                        hovertemplate=(
                            "t=%{x}<br>price=%{y}<br>"
                            "qty=%{customdata[0]}"
                            "<extra></extra>"
                        ),
                        customdata=list(zip(trades["quantity"],)),
                    ))

        fig.update_layout(
            title=f"{product} Order Book (Historical)",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            hovermode="x unified",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            legend={"orientation": "h", "y": -0.15},
            template="plotly_white",
        )
        return fig
