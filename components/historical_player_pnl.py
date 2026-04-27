import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import historical_store

TRADER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _trader_color_map(names):
    return {n: TRADER_PALETTE[i % len(TRADER_PALETTE)] for i, n in enumerate(sorted(names))}


def layout():
    return html.Div(style={
        "display": "grid",
        "gridTemplateRows": "auto 70vh",
        "gap": "4px",
        "padding": "8px",
    }, children=[
        html.Div(style={
            "display": "flex", "alignItems": "center", "gap": "16px",
            "flexWrap": "wrap",
        }, children=[
            html.Div([
                html.Label("Mode", style={"fontWeight": "bold", "marginRight": "6px"}),
                dcc.RadioItems(
                    id="hist-pnl-mode",
                    options=[
                        {"label": "Overall (sum across products)", "value": "overall"},
                        {"label": "Per Product", "value": "per_product"},
                    ],
                    value="overall",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px"},
                ),
            ]),
            html.Div([
                html.Label("Product (Per-Product mode)", style={
                    "fontWeight": "bold", "marginRight": "6px", "fontSize": "12px",
                }),
                dcc.Dropdown(
                    id="hist-pnl-product",
                    clearable=False,
                    style={"width": "240px", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="hist-pnl-summary", style={
                "fontSize": "12px", "color": "#444",
            }),
        ]),
        dcc.Graph(id="hist-pnl-chart", style={"height": "100%"}),
    ])


def _compute_player_pnl(trades, prices, products):
    """Compute time series of PnL per player.

    trades: df with columns timestamp, buyer, seller, symbol, price, quantity
    prices: df with columns timestamp, product, mid_price
    products: list of products to mark to market against

    Returns: (pnl_df indexed by timestamp with one column per player,
              summary dict {player: {trades, vol, final_pnl, final_pos_total}})
    """
    if trades.empty or prices.empty:
        return pd.DataFrame(), {}

    # Per-product mid time series, forward-filled.
    mid = prices.pivot_table(
        index="timestamp", columns="product", values="mid_price", aggfunc="last",
    ).sort_index().ffill()

    buyer_flow = pd.DataFrame({
        "timestamp": trades["timestamp"].values,
        "player": trades["buyer"].astype(str).values,
        "product": trades["symbol"].values,
        "pos_change": trades["quantity"].astype(float).values,
        "cash_change": (-trades["price"].astype(float) * trades["quantity"].astype(float)).values,
    })
    seller_flow = pd.DataFrame({
        "timestamp": trades["timestamp"].values,
        "player": trades["seller"].astype(str).values,
        "product": trades["symbol"].values,
        "pos_change": (-trades["quantity"].astype(float)).values,
        "cash_change": (trades["price"].astype(float) * trades["quantity"].astype(float)).values,
    })
    flow = pd.concat([buyer_flow, seller_flow], ignore_index=True)
    flow = flow[flow["player"].str.len() > 0]
    if flow.empty:
        return pd.DataFrame(), {}

    players = sorted(flow["player"].unique())
    common_idx = mid.index

    pnl_cols = {}
    summary = {}
    for player in players:
        p_flow = flow[flow["player"] == player]
        if p_flow.empty:
            continue

        total_trades = len(p_flow)
        total_vol = float(p_flow["pos_change"].abs().sum())

        pnl_series = pd.Series(0.0, index=common_idx)
        final_pos_abs = 0.0

        for prod, sub in p_flow.groupby("product"):
            agg = sub.groupby("timestamp").agg(
                pos_change=("pos_change", "sum"),
                cash_change=("cash_change", "sum"),
            ).cumsum()

            pos = agg["pos_change"].reindex(common_idx, method="ffill").fillna(0.0)
            cash = agg["cash_change"].reindex(common_idx, method="ffill").fillna(0.0)

            if prod in mid.columns:
                mtm = pos * mid[prod].fillna(0.0)
            else:
                mtm = pd.Series(0.0, index=common_idx)

            pnl_series = pnl_series + cash + mtm
            if not pos.empty:
                final_pos_abs += abs(float(pos.iloc[-1]))

        pnl_cols[player] = pnl_series
        summary[player] = {
            "trades": total_trades,
            "volume": total_vol,
            "final_pnl": float(pnl_series.iloc[-1]) if not pnl_series.empty else 0.0,
            "final_pos_abs": final_pos_abs,
        }

    return pd.DataFrame(pnl_cols), summary


def register_callbacks(app):
    @app.callback(
        [Output("hist-pnl-product", "options"),
         Output("hist-pnl-product", "value")],
        Input("hist-product-selector", "options"),
    )
    def populate_pnl_product(options):
        if not options:
            raise PreventUpdate
        return options, options[0]["value"]

    @app.callback(
        [Output("hist-pnl-chart", "figure"),
         Output("hist-pnl-summary", "children")],
        [Input("hist-pnl-mode", "value"),
         Input("hist-pnl-product", "value"),
         Input("hist-day-selector", "value")],
    )
    def update_pnl_chart(mode, product, day):
        if not historical_store.is_loaded() or day is None:
            raise PreventUpdate

        trades = historical_store.get_all_trades(day=day)
        prices = historical_store.get_all_activities(day=day)

        if mode == "per_product":
            if not product:
                return go.Figure(), "Select a product."
            trades = trades[trades["symbol"] == product]
            prices = prices[prices["product"] == product]
            products = [product]
            title = f"Player PnL — {product} (day {day})"
        else:
            products = sorted(trades["symbol"].unique().tolist()) if not trades.empty else []
            title = f"Player PnL — Overall (day {day})"

        pnl_df, summary = _compute_player_pnl(trades, prices, products)

        fig = go.Figure()
        if pnl_df.empty:
            fig.update_layout(
                title=title + " — no data",
                template="plotly_white",
                margin={"l": 50, "r": 20, "t": 40, "b": 30},
            )
            return fig, "No trades for this view."

        cmap = _trader_color_map(pnl_df.columns)
        # Sort traces by final PnL descending so legend ranks them.
        ranked = sorted(pnl_df.columns, key=lambda p: summary[p]["final_pnl"], reverse=True)
        for player in ranked:
            fig.add_trace(go.Scatter(
                x=pnl_df.index,
                y=pnl_df[player],
                mode="lines",
                name=f"{player} ({summary[player]['final_pnl']:+,.0f})",
                line={"color": cmap[player], "width": 2},
            ))

        fig.add_hline(y=0, line={"color": "#888", "width": 1, "dash": "dash"})
        fig.update_layout(
            title=title,
            xaxis_title="Timestamp",
            yaxis_title="PnL (XIRECS)",
            margin={"l": 60, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "v", "x": 1.02, "y": 1, "font": {"size": 11}},
        )

        rows = []
        for player in ranked:
            s = summary[player]
            rows.append(html.Div(
                f"{player}: PnL {s['final_pnl']:+,.0f} | "
                f"trades {s['trades']:,} | vol {s['volume']:,.0f} | "
                f"|final pos| {s['final_pos_abs']:.0f}",
                style={"color": cmap[player], "fontWeight": "bold"},
            ))
        summary_block = html.Div(rows, style={"display": "flex", "flexDirection": "column"})

        return fig, summary_block
