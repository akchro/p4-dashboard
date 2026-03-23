import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import historical_store


def layout():
    return html.Div([
        dcc.Graph(id="hist-volume-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("hist-volume-chart", "figure"),
        [Input("hist-product-selector", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-trade-toggle", "value"),
         Input("hist-qty-filter", "value"),
         Input("hist-volume-bucket-slider", "value")],
    )
    def update_volume_chart(product, day, trade_toggle, qty_range, bucket_size):
        if not product or not historical_store.is_loaded():
            raise PreventUpdate

        trades = historical_store.get_trades(product, day)
        fig = go.Figure()
        bucket_size = bucket_size or 10000

        if not trades.empty and trade_toggle and "show" in trade_toggle:
            filtered = trades.copy()
            if qty_range:
                filtered = filtered[
                    (filtered["quantity"] >= qty_range[0]) &
                    (filtered["quantity"] <= qty_range[1])
                ]
            if not filtered.empty:
                # Bucket trades into intervals and sum quantities
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
