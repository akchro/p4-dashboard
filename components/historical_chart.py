import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import historical_store
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

OVERLAY_COLORS = [
    "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#ff7f0e", "#1f77b4", "#d62728", "#aec7e8", "#98df8a",
]

# Mid-saturation trader colors. Each marker has a thin black outer outline
# so dark colors still pop against the bid/ask/mid lines; darker palette
# gives buyer-fill vs seller-ring more contrast against each other.
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
         Input("hist-product-overlay-list", "value"),
         Input("hist-product-overlay-mode", "value"),
         Input("hist-product-overlay-sum-toggle", "value"),
         Input("hist-product-overlay-spread-toggle", "value"),
         Input("hist-trader-toggles", "value"),
         Input("hist-zscore-toggle", "value"),
         Input("hist-zscore-mean", "value"),
         Input("hist-zscore-entry", "value"),
         Input("hist-zscore-exit", "value")],
    )
    def update_hist_chart(product, day, downsample, levels, trade_toggle, qty_range, qty_exact, wallmid_toggle, overlay_products, overlay_list, overlay_mode, overlay_sum_toggle, overlay_spread_toggle, selected_traders, zscore_toggle, zscore_mu, zscore_entry, zscore_exit):
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
                        _add_split_trade_markers(fig, trades, buyer, seller, cmap)
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
                op_acts = historical_store.get_activities(op, day)
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
                op_acts = historical_store.get_activities(first_op, day)
                if not op_acts.empty:
                    if downsample and downsample > 1:
                        op_acts = op_acts.iloc[::downsample]
                    main_mid = acts[["timestamp", "mid_price"]].dropna(subset=["mid_price"])
                    op_mid = op_acts[["timestamp", "mid_price"]].dropna(subset=["mid_price"])
                    merged = main_mid.merge(op_mid, on="timestamp", suffixes=("_main", "_op"))
                    if not merged.empty:
                        spread = merged["mid_price_main"].values - merged["mid_price_op"].values
                        is_abs = "abs" in overlay_spread_toggle
                        ys = abs(spread) if is_abs else spread
                        label = (f"|spread|: |{product} − {first_op}|"
                                 if is_abs else f"spread: {product} − {first_op}")
                        fig.add_trace(go.Scatter(
                            x=merged["timestamp"], y=ys, mode="lines",
                            name=label,
                            line={"color": "#FF6F00", "width": 2, "dash": "dot"},
                            hovertemplate=f"t=%{{x}}<br>{'|spread|' if is_abs else 'spread'}=%{{y:.2f}}<extra></extra>",
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

        zscore_overlay.add_lines(fig, product, zscore_toggle,
                                 zscore_mu, zscore_entry, zscore_exit)

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
