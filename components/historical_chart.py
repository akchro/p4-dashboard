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

# Light/neon trader fills, chosen to pop against the darker bid (blue),
# ask (red), and mid (black) lines.
TRADER_PALETTE = [
    "#FFFF00",  # yellow
    "#00FFFF",  # cyan
    "#FF66FF",  # light magenta
    "#99FF66",  # light lime
    "#FFA500",  # orange
    "#FF99CC",  # light pink
    "#66CCFF",  # light blue
    "#FFD700",  # gold
    "#DDA0DD",  # plum
    "#00FF99",  # mint
]


def _trader_color_map(names):
    return {n: TRADER_PALETTE[i % len(TRADER_PALETTE)] for i, n in enumerate(sorted(names))}


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
         Input("hist-r3-overlay-products", "value"),
         Input("hist-trader-toggles", "value")],
    )
    def update_hist_chart(product, day, downsample, levels, trade_toggle, qty_range, qty_exact, wallmid_toggle, overlay_products, selected_traders):
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
                # Trader filter: keep trades where buyer OR seller is selected.
                # If trader column has names but none are selected, hide all.
                # If all trades have empty names (e.g. R3 historical), no-op.
                if selected_traders is not None and {"buyer", "seller"}.issubset(trades.columns):
                    has_named = (
                        trades["buyer"].fillna("").astype(str).str.len().gt(0) |
                        trades["seller"].fillna("").astype(str).str.len().gt(0)
                    ).any()
                    if has_named:
                        sel = set(selected_traders)
                        mask = (
                            trades["buyer"].fillna("").astype(str).isin(sel) |
                            trades["seller"].fillna("").astype(str).isin(sel)
                        )
                        trades = trades[mask]
                if not trades.empty:
                    has_buyer_col = "buyer" in trades.columns
                    has_seller_col = "seller" in trades.columns
                    buyer = trades["buyer"].fillna("").astype(str) if has_buyer_col else None
                    seller = trades["seller"].fillna("").astype(str) if has_seller_col else None
                    has_named = (
                        ((buyer is not None and buyer.str.len().gt(0).any()) or
                         (seller is not None and seller.str.len().gt(0).any()))
                    )
                    if has_named:
                        # Use the dashboard's full trader list (not just this product's) so
                        # colors stay consistent across product/day switches.
                        all_traders = historical_store.get_traders()
                        cmap = _trader_color_map(all_traders)
                        # Group dots by buyer; outline color encodes seller. Both visible at once.
                        for buyer_name in sorted(buyer.unique()):
                            mask = buyer == buyer_name
                            sub = trades[mask]
                            if sub.empty:
                                continue
                            display = buyer_name if buyer_name else "(unknown)"
                            sub_seller = (
                                sub["seller"].fillna("").astype(str) if has_seller_col
                                else [""] * len(sub)
                            )
                            fill_color = cmap.get(buyer_name, "yellow")
                            fig.add_trace(go.Scatter(
                                x=sub["timestamp"],
                                y=sub["price"],
                                mode="markers",
                                name=f"buyer: {display}",
                                marker={
                                    "color": fill_color,
                                    "symbol": "circle",
                                    "size": 9,
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
                                    sub["quantity"],
                                    [display] * len(sub),
                                    [s if s else "(unknown)" for s in sub_seller],
                                )),
                            ))
                    else:
                        # Anonymous trades (e.g. ROUND_3) — keep the original single yellow trace.
                        b_label = (
                            buyer.replace("", "-") if buyer is not None else ["-"] * len(trades)
                        )
                        s_label = (
                            seller.replace("", "-") if seller is not None else ["-"] * len(trades)
                        )
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
                                "qty=%{customdata[0]}<br>"
                                "buyer=%{customdata[1]}<br>"
                                "seller=%{customdata[2]}"
                                "<extra></extra>"
                            ),
                            customdata=list(zip(trades["quantity"], b_label, s_label)),
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
