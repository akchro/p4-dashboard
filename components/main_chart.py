import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from data import store
from components import zscore_overlay

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

# Mid-saturation trader colors. Each marker has a thin black outer outline
# so dark colors still pop against the bid/ask/mid lines. Mirror of
# historical_chart.TRADER_PALETTE so colors are consistent across tabs.
TRADER_PALETTE = [
    "#D32F2F",  # red
    "#1976D2",  # blue
    "#388E3C",  # green
    "#7B1FA2",  # purple
    "#F57C00",  # orange
    "#0097A7",  # teal
    "#FBC02D",  # yellow
    "#5D4037",  # brown
    "#C2185B",  # pink
    "#455A64",  # blue-gray
]


def _trader_color_map(names):
    return {n: TRADER_PALETTE[i % len(TRADER_PALETTE)] for i, n in enumerate(sorted(names))}


def _add_split_trade_markers(fig, trades, buyer, seller, cmap):
    # Stack two markers per trade:
    #   outer (size 11) — seller color fill, thin black outline (graph contrast)
    #   inner (size 6)  — buyer color fill, no outline
    # Result: black ring → seller color ring → buyer color core.
    if trades.empty:
        return

    qty = trades["quantity"].tolist()
    ts = trades["timestamp"].tolist()
    px = trades["price"].tolist()
    buyers = list(buyer)
    sellers = list(seller)
    buyer_disp = [b if b else "(unknown)" for b in buyers]
    seller_disp = [s if s else "(unknown)" for s in sellers]
    hover = (
        "t=%{x}<br>price=%{y}<br>"
        "qty=%{customdata[0]}<br>"
        "buyer=%{customdata[1]}<br>"
        "seller=%{customdata[2]}"
        "<extra></extra>"
    )

    for buyer_name in sorted(set(buyers)):
        idxs = [i for i in range(len(trades)) if buyers[i] == buyer_name]
        if not idxs:
            continue
        display = buyer_name if buyer_name else "(unknown)"
        fill_color = cmap.get(buyer_name, "yellow") if buyer_name else "lightgray"
        seller_colors = [
            cmap.get(sellers[i], "lightgray") if sellers[i] else "lightgray"
            for i in idxs
        ]
        xs = [ts[i] for i in idxs]
        ys = [px[i] for i in idxs]
        cd = [(qty[i], buyer_disp[i], seller_disp[i]) for i in idxs]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"buyer: {display}", legendgroup=display, showlegend=False,
            marker={
                "color": seller_colors,
                "symbol": "circle",
                "size": 11,
                "line": {"width": 1, "color": "black"},
            },
            hovertemplate=hover, customdata=cd,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"buyer: {display}", legendgroup=display, showlegend=True,
            marker={
                "color": fill_color,
                "symbol": "circle",
                "size": 6,
                "line": {"width": 0},
            },
            hovertemplate=hover, customdata=cd,
        ))

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
         Input("bull-signal-toggle", "value"),
         Input("r3-overlay-products", "value"),
         Input("r3-overlay-points-toggle", "value"),
         Input("product-overlay-list", "value"),
         Input("product-overlay-mode", "value"),
         Input("product-overlay-sum-toggle", "value"),
         Input("product-overlay-spread-toggle", "value"),
         Input("trader-toggles", "value"),
         Input("live-zscore-toggle", "value"),
         Input("live-zscore-mean", "value"),
         Input("live-zscore-entry", "value"),
         Input("live-zscore-exit", "value")],
    )
    def update_main_chart(product, day, downsample, levels, trade_toggle, qty_range, qty_exact, wallmid_toggle, dashboard_wallmid_toggle, bull_signal_toggle, overlay_products, overlay_points_toggle, overlay_list, overlay_mode, overlay_sum_toggle, overlay_spread_toggle, selected_traders, zscore_toggle, zscore_mu, zscore_entry, zscore_exit):
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
                        _add_split_trade_markers(fig, market_trades, buyer, seller, cmap)
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

        need_yaxis2 = False

        # Product price overlay — mid-price lines of other products on the same chart.
        # rebased: shifted so each starts at the main product's first mid (shape comparison)
        # raw:     plotted on the main y-axis alongside the current product
        if overlay_list:
            main_first = None
            if overlay_mode == "rebased":
                main_mid = acts["mid_price"].dropna() if "mid_price" in acts.columns else pd.Series(dtype=float)
                if not main_mid.empty:
                    main_first = float(main_mid.iloc[0])
            sum_panel = None
            for i, op in enumerate(overlay_list):
                if op == product:
                    continue
                op_acts = store.get_activities(op, day)
                if op_acts.empty:
                    continue
                if downsample and downsample > 1:
                    op_acts = op_acts.iloc[::downsample]
                op_mid = op_acts.dropna(subset=["mid_price"])
                if op_mid.empty:
                    continue
                color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                if overlay_mode == "rebased" and main_first is not None:
                    shift = main_first - float(op_mid["mid_price"].iloc[0])
                    ys = op_mid["mid_price"] + shift
                    fig.add_trace(go.Scatter(
                        x=op_mid["timestamp"], y=ys,
                        mode="lines", name=f"{op} (rebased {shift:+.1f})",
                        line={"color": color, "width": 2, "dash": "dash"},
                        connectgaps=False,
                        customdata=op_mid["mid_price"],
                        hovertemplate=(
                            "t=%{x}<br>rebased=%{y:.2f}<br>"
                            "actual mid=%{customdata:.2f}"
                            f"<extra>{op}</extra>"
                        ),
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=op_mid["timestamp"], y=op_mid["mid_price"],
                        mode="lines", name=op,
                        line={"color": color, "width": 2, "dash": "dash"},
                        connectgaps=False,
                        hovertemplate=f"t=%{{x}}<br>mid=%{{y:.2f}}<extra>{op}</extra>",
                    ))

                # Accumulate aligned panel for the sum overlay (timestamp-merged)
                if overlay_sum_toggle and "show" in overlay_sum_toggle:
                    col = op_mid[["timestamp", "mid_price"]].rename(
                        columns={"mid_price": op})
                    sum_panel = col if sum_panel is None else sum_panel.merge(
                        col, on="timestamp", how="outer")

            # Sum-of-overlay-mids line — only timestamps where every selected
            # overlay product has data (inner-merge semantics via dropna).
            if (overlay_sum_toggle and "show" in overlay_sum_toggle
                    and sum_panel is not None and len(sum_panel.columns) > 1):
                s = sum_panel.sort_values("timestamp").dropna()
                if not s.empty:
                    n_legs = len(s.columns) - 1
                    ts = s["timestamp"]
                    total = s.drop(columns="timestamp").sum(axis=1)
                    if overlay_mode == "rebased" and main_first is not None:
                        shift = main_first - float(total.iloc[0])
                        ys = total + shift
                        fig.add_trace(go.Scatter(
                            x=ts, y=ys, mode="lines",
                            name=f"Σ overlay mids · {n_legs} legs (rebased {shift:+.1f})",
                            line={"color": "#000000", "width": 2.5},
                            customdata=total,
                            hovertemplate=(
                                "t=%{x}<br>rebased Σ=%{y:.2f}<br>"
                                "actual Σ=%{customdata:.2f}"
                                "<extra>Σ overlay</extra>"
                            ),
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=ts, y=total, mode="lines",
                            name=f"Σ overlay mids · {n_legs} legs",
                            line={"color": "#000000", "width": 2.5},
                            hovertemplate="t=%{x}<br>Σ=%{y:.2f}<extra>Σ overlay</extra>",
                        ))

        # Spread trace: current_product_mid - first_overlay_mid, timestamp-merged
        if (overlay_spread_toggle and "show" in overlay_spread_toggle
                and overlay_list):
            first_op = next((o for o in overlay_list if o != product), None)
            if first_op is not None and "mid_price" in acts.columns:
                op_acts = store.get_activities(first_op, day)
                if not op_acts.empty:
                    if downsample and downsample > 1:
                        op_acts = op_acts.iloc[::downsample]
                    main_mid = acts[["timestamp", "mid_price"]].dropna(subset=["mid_price"])
                    op_mid = op_acts[["timestamp", "mid_price"]].dropna(subset=["mid_price"])
                    merged = main_mid.merge(op_mid, on="timestamp", suffixes=("_main", "_op"))
                    if not merged.empty:
                        spread = merged["mid_price_main"].values - merged["mid_price_op"].values
                        is_abs = "abs" in overlay_spread_toggle
                        is_rebase = "rebase" in overlay_spread_toggle
                        raw = abs(spread) if is_abs else spread
                        base_label = "|spread|" if is_abs else "spread"
                        pair = (f"|{product} − {first_op}|" if is_abs
                                else f"{product} − {first_op}")
                        if is_rebase and len(raw) > 0:
                            shift = float(merged["mid_price_main"].iloc[0]) - float(raw[0])
                            ys = raw + shift
                            fig.add_trace(go.Scatter(
                                x=merged["timestamp"], y=ys, mode="lines",
                                name=f"{base_label}: {pair} (rebased {shift:+.1f})",
                                line={"color": "#FF6F00", "width": 2, "dash": "dot"},
                                customdata=raw,
                                hovertemplate=(
                                    "t=%{x}<br>rebased=%{y:.2f}<br>"
                                    f"actual {base_label}=%{{customdata:.2f}}<extra></extra>"
                                ),
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=merged["timestamp"], y=raw, mode="lines",
                                name=f"{base_label}: {pair}",
                                line={"color": "#FF6F00", "width": 2, "dash": "dot"},
                                hovertemplate=f"t=%{{x}}<br>{base_label}=%{{y:.2f}}<extra></extra>",
                            ))

        # Bull-signal up-arrows pinned to ask_price_1 of the displayed product
        # (signal is global; the y-anchor is per-product so the marker sits on
        # the touch ask at that tick).
        if bull_signal_toggle and "show" in bull_signal_toggle and "ask_price_1" in acts.columns:
            bull_df = store.get_bull_signals(day)
            bull_df = bull_df[bull_df["bull"] > 0]
            if not bull_df.empty:
                ask1 = acts[["timestamp", "ask_price_1"]].dropna(subset=["ask_price_1"])
                merged = bull_df.merge(ask1, on="timestamp", how="inner")
                if not merged.empty:
                    fig.add_trace(go.Scatter(
                        x=merged["timestamp"],
                        y=merged["ask_price_1"],
                        mode="markers",
                        name=f"Bull Signal ({len(merged)})",
                        marker={
                            "color": "#00C853",
                            "symbol": "triangle-up",
                            "size": 10,
                            "line": {"width": 1, "color": "black"},
                        },
                        hovertemplate="t=%{x}<br>ask_1=%{y}<br>bull=%{customdata}<extra></extra>",
                        customdata=merged["bull"].tolist(),
                    ))

        # R3 trade-time overlays (dotted vertical lines per selected product;
        # optional price markers at raw trade prices on the main y-axis).
        if overlay_products:
            show_points = bool(overlay_points_toggle) and "show" in overlay_points_toggle
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
                    legendgroup=op,
                    hoverinfo="skip",
                    showlegend=True,
                ))

                if show_points:
                    fig.add_trace(go.Scatter(
                        x=ot["timestamp"],
                        y=ot["price"].astype(float),
                        mode="markers",
                        name=f"{op} trades",
                        marker={
                            "color": color, "symbol": "diamond",
                            "size": 6, "opacity": 0.65,
                            "line": {"width": 0.5, "color": "black"},
                        },
                        legendgroup=op,
                        hovertemplate=("t=%{x}<br>price=%{y:.2f}<br>"
                                       "qty=%{customdata}"
                                       f"<extra>{op}</extra>"),
                        customdata=ot["quantity"].tolist(),
                    ))
            need_yaxis2 = True

        if need_yaxis2:
            fig.update_layout(
                yaxis2={
                    "overlaying": "y", "range": [0, 1],
                    "showgrid": False, "showticklabels": False, "fixedrange": True,
                },
            )

        zscore_overlay.add_lines(fig, product, zscore_toggle,
                                 zscore_mu, zscore_entry, zscore_exit)

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
