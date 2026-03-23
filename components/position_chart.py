import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import store
from utils.position import cumulative_position


def layout():
    return html.Div([
        dcc.Graph(id="position-chart", style={"height": "100%"}),
    ], style={"height": "100%"})


def register_callbacks(app):
    @app.callback(
        Output("position-chart", "figure"),
        [Input("product-selector", "value"),
         Input("day-selector", "value")],
    )
    def update_position(product, day):
        if not product or not store.is_loaded():
            raise PreventUpdate

        trades = store.get_trades(product, day)
        pos = cumulative_position(trades)

        fig = go.Figure()
        if not pos.empty:
            fig.add_trace(go.Scatter(
                x=pos["timestamp"], y=pos["position"],
                mode="lines", name="Position",
                line={"color": "black", "width": 1.5},
            ))
        fig.update_layout(
            title="Net Position",
            xaxis_title="Timestamp",
            yaxis_title="Position",
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            template="plotly_white",
            hovermode="x unified",
        )
        return fig
