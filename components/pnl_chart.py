import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div([
        dcc.Graph(id="pnl-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("pnl-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value"),
         Input("downsample-slider", "value")],
    )
    def update_pnl(product, day, downsample):
        if not product or not store.is_loaded():
            raise PreventUpdate

        acts = store.get_activities(product, day)
        if downsample and downsample > 1:
            acts = acts.iloc[::downsample]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acts["timestamp"], y=acts["profit_and_loss"],
            mode="lines", name="PnL",
            line={"color": "black", "width": 1.5},
        ))
        fig.update_layout(
            title="Profit & Loss",
            xaxis_title="Timestamp",
            yaxis_title="PnL",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
            hovermode="x unified",
        )
        return fig
