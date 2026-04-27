import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div([
        dcc.Graph(id="volume-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("volume-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value"),
         Input("trade-toggle", "value"),
         Input("qty-filter", "value"),
         Input("qty-filter-exact", "value"),
         Input("volume-bucket-slider", "value"),
         Input("volume-our-trades-toggle", "value"),
         Input("trader-toggles", "value")],
    )
    def update_volume_chart(product, day, trade_toggle, qty_range, qty_exact, bucket_size, our_trades_toggle, selected_traders):
        if not product or not store.is_loaded():
            raise PreventUpdate

        trades = store.get_trades(product, day)
        fig = go.Figure()
        bucket_size = bucket_size or 10000

        if not trades.empty and trade_toggle and "show" in trade_toggle:
            filtered = trades.copy()
            if not (our_trades_toggle and "include" in our_trades_toggle):
                if "side" in filtered.columns:
                    filtered = filtered[filtered["side"] == "market"]
            if qty_exact is not None:
                filtered = filtered[filtered["quantity"] == qty_exact]
            elif qty_range:
                filtered = filtered[
                    (filtered["quantity"] >= qty_range[0]) &
                    (filtered["quantity"] <= qty_range[1])
                ]
            if selected_traders is not None and {"buyer", "seller"}.issubset(filtered.columns):
                has_named = (
                    filtered["buyer"].fillna("").astype(str).str.len().gt(0) |
                    filtered["seller"].fillna("").astype(str).str.len().gt(0)
                ).any()
                if has_named:
                    sel = set(selected_traders)
                    mask = (
                        filtered["buyer"].fillna("").astype(str).isin(sel) |
                        filtered["seller"].fillna("").astype(str).isin(sel)
                    )
                    filtered = filtered[mask]
            if not filtered.empty:
                filtered = filtered.copy()
                filtered["bucket"] = (filtered["timestamp"] // bucket_size) * bucket_size
                vol = filtered.groupby("bucket")["quantity"].sum().reset_index()

                fig.add_trace(go.Bar(
                    x=vol["bucket"],
                    y=vol["quantity"],
                    name="Volume",
                    marker_color="steelblue",
                    opacity=0.7,
                    width=bucket_size * 0.8,
                ))

        fig.update_layout(
            title=f"{product} Market Volume (bucket={bucket_size:,})" if product else "Market Volume",
            xaxis_title="Timestamp",
            yaxis_title="Total Quantity",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
        )
        return fig
