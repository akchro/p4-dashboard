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

OVERLAY_COLORS = [
    "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#ff7f0e", "#1f77b4", "#d62728", "#aec7e8", "#98df8a",
]


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
         Input("hist-qty-filter", "value"),
         Input("hist-qty-filter-exact", "value"),
         Input("hist-wallmid-toggle", "value"),
         Input("hist-r3-overlay-products", "value")],
    )
    def update_hist_chart(product, day, downsample, levels, trade_toggle, qty_range, qty_exact, wallmid_toggle, overlay_products):
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
            line={"color": "black", "width": 2.5},
            connectgaps=False,
        ))

        # Wallmid overlay
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

        # Trade markers (all shown as gray circles — no buy/sell distinction)
        if trade_toggle and "show" in trade_toggle:
            trades = historical_store.get_trades(product, day)
            if not trades.empty:
                if qty_exact is not None:
                    trades = trades[trades["quantity"] == qty_exact]
                elif qty_range:
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
                        marker={
                            "color": "yellow",
                            "symbol": "circle",
                            "size": 8,
                            "line": {"width": 1, "color": "black"},
                        },
                        hovertemplate=(
                            "t=%{x}<br>price=%{y}<br>"
                            "qty=%{customdata[0]}"
                            "<extra></extra>"
                        ),
                        customdata=list(zip(trades["quantity"],)),
                    ))

        # R3 trade-time overlays (dotted vertical lines per selected product)
        if overlay_products:
            for i, op in enumerate(overlay_products):
                if op == product:
                    continue
                ot = historical_store.get_trades(op, day)
                if ot.empty:
                    continue
                color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                xs, ys = [], []
                for t in ot["timestamp"]:
                    xs.extend([t, t, None])
                    ys.extend([0, 1, None])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=f"{op} ({len(ot)})",
                    line={"color": color, "width": 1, "dash": "dot"},
                    yaxis="y2",
                    hoverinfo="skip",
                    showlegend=True,
                ))
            fig.update_layout(
                yaxis2={
                    "overlaying": "y", "range": [0, 1],
                    "showgrid": False, "showticklabels": False, "fixedrange": True,
                },
            )

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
