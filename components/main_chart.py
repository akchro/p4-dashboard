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
}

# Light/neon trader fills, chosen to pop against the darker bid (blue),
# ask (red), and mid (black) lines. Mirror of historical_chart.TRADER_PALETTE
# so colors are consistent between Live and Historical tabs.
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

# Distinct colors for R3 trade-time overlays (avoids colors used elsewhere
# on the chart: black mid, red/blue book levels, orange/cyan/yellow trades,
# magenta/turquoise wallmids).
OVERLAY_COLORS = [
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
    "#ff7f0e",  # dark orange
    "#1f77b4",  # steel blue
    "#d62728",  # brick red
    "#aec7e8",  # light steel
    "#98df8a",  # light green
]


def layout():
    return html.Div([
        dcc.Graph(id="main-chart", style={"height": "calc(100% - 28px)"}),
        html.Div(
            id="main-range-stats",
            children="Zoom into a range to see std dev, CV & max drawdown",
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
         Input("qty-filter-exact", "value"),
         Input("wallmid-toggle", "value"),
         Input("dashboard-wallmid-toggle", "value"),
         Input("r3-overlay-products", "value"),
         Input("trader-toggles", "value")],
    )
    def update_main_chart(product, day, downsample, levels, trade_toggle, qty_range, qty_exact, wallmid_toggle, dashboard_wallmid_toggle, overlay_products, selected_traders):
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

        # Dashboard wallmid overlay (computed from order book levels)
        if dashboard_wallmid_toggle and "show" in dashboard_wallmid_toggle and "dashboard_wallmid" in acts.columns:
            dwm = acts.dropna(subset=["dashboard_wallmid"])
            if not dwm.empty:
                fig.add_trace(go.Scatter(
                    x=dwm["timestamp"], y=dwm["dashboard_wallmid"],
                    mode="lines", name="Dashboard Wallmid",
                    line={"color": "#00CED1", "width": 2},
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
                if qty_exact is not None:
                    trades = trades[trades["quantity"] == qty_exact]
                elif qty_range:
                    trades = trades[
                        (trades["quantity"] >= qty_range[0]) &
                        (trades["quantity"] <= qty_range[1])
                    ]
                # Trader filter: keep trades where buyer OR seller is selected.
                # SUBMISSION (our bot) is treated as a trader and shows up in the list.
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
                # Our own (SUBMISSION) trades stay as orange/cyan triangles.
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

                # Market trades (between two non-SUBMISSION parties) — color by
                # buyer, with seller available in the hover. Mirrors the
                # historical chart so the same trader keeps the same color
                # across Live and Historical tabs.
                market_trades = trades[trades["side"] == "market"]
                if not market_trades.empty:
                    buyer = market_trades["buyer"].fillna("").astype(str)
                    seller = market_trades["seller"].fillna("").astype(str)
                    has_named = buyer.str.len().gt(0).any() or seller.str.len().gt(0).any()

                    if has_named:
                        cmap = _trader_color_map(store.get_traders())
                        for buyer_name in sorted(buyer.unique()):
                            mask = buyer == buyer_name
                            sub = market_trades[mask]
                            if sub.empty:
                                continue
                            display = buyer_name if buyer_name else "(unknown)"
                            sub_seller = sub["seller"].fillna("").astype(str)
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
                        # Anonymous market trades — fall back to single yellow trace.
                        b_label = buyer.replace("", "-")
                        s_label = seller.replace("", "-")
                        fig.add_trace(go.Scatter(
                            x=market_trades["timestamp"],
                            y=market_trades["price"],
                            mode="markers",
                            name="Market",
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
                            customdata=list(zip(market_trades["quantity"], b_label, s_label)),
                        ))

        # R3 trade-time overlays (dotted vertical lines per selected product)
        if overlay_products:
            for i, op in enumerate(overlay_products):
                if op == product:
                    continue  # would just clutter own trades; user already sees them
                ot = store.get_trades(op, day)
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
            return "Zoom into a range to see std dev, CV & max drawdown"

        acts = store.get_activities(product, day)
        ts = pd.to_datetime(acts["timestamp"])
        x0, x1 = pd.to_datetime(x0), pd.to_datetime(x1)
        ys = acts.loc[(ts >= x0) & (ts <= x1), "mid_price"].dropna().values

        if len(ys) < 2:
            return "Zoom range has fewer than 2 points"

        std_dev = float(np.std(ys))
        mean = float(np.mean(ys))
        cv = (std_dev / abs(mean) * 100) if mean != 0 else 0

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
            f"Std Dev: {std_dev:,.4f} | CV: {cv:.2f}% | "
            f"Max Drawdown: {max_dd:,.4f} ({pct:.1f}%)"
        )
