import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div([
        dcc.Graph(id="order-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("order-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value"),
         Input("order-bucket-slider", "value")],
    )
    def update_order_chart(product, day, bucket_size):
        if not product or not store.is_loaded():
            raise PreventUpdate

        trades = store.get_trades(product, day)
        fig = go.Figure()
        bucket_size = bucket_size or 10000

        if not trades.empty and "side" in trades.columns:
            ours = trades[trades["side"].isin(["buy", "sell"])].copy()
            if not ours.empty:
                ours["signed_qty"] = ours.apply(
                    lambda r: r["quantity"] if r["side"] == "buy" else -r["quantity"],
                    axis=1,
                )
                ours["bucket"] = (ours["timestamp"] // bucket_size) * bucket_size

                buys = ours[ours["signed_qty"] > 0].groupby("bucket")["signed_qty"].sum().reset_index()
                sells = ours[ours["signed_qty"] < 0].groupby("bucket")["signed_qty"].sum().reset_index()

                if not buys.empty:
                    fig.add_trace(go.Bar(
                        x=buys["bucket"],
                        y=buys["signed_qty"],
                        name="Buy",
                        marker_color="green",
                        opacity=0.7,
                        width=bucket_size * 0.8,
                    ))
                if not sells.empty:
                    fig.add_trace(go.Bar(
                        x=sells["bucket"],
                        y=sells["signed_qty"],
                        name="Sell",
                        marker_color="red",
                        opacity=0.7,
                        width=bucket_size * 0.8,
                    ))

        fig.update_layout(
            title=f"{product} Our Orders (bucket={bucket_size:,})" if product else "Our Orders",
            xaxis_title="Timestamp",
            yaxis_title="Quantity",
            barmode="relative",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
        )
        return fig
