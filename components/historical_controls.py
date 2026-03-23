from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from data import historical_store
from data.historical_loader import get_rounds


def layout():
    return html.Div(id="hist-controls-panel", children=[
        html.Label("Round", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="hist-round-selector", clearable=False),
        html.Br(),
        html.Label("Product", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="hist-product-selector", clearable=False),
        html.Br(),
        html.Label("Day", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="hist-day-selector", clearable=False),
    ])


def register_callbacks(app):
    @app.callback(
        Output("hist-round-selector", "options"),
        Input("hist-round-selector", "id"),
    )
    def populate_rounds(_):
        rounds = get_rounds()
        return [{"label": r, "value": r} for r in rounds]

    @app.callback(
        [Output("hist-product-selector", "options"),
         Output("hist-product-selector", "value"),
         Output("hist-day-selector", "options"),
         Output("hist-day-selector", "value")],
        Input("hist-round-selector", "value"),
    )
    def load_round(round_name):
        if not round_name:
            raise PreventUpdate
        historical_store.load(round_name)
        products = historical_store.get_products()
        days = historical_store.get_days()
        return (
            [{"label": p, "value": p} for p in products],
            products[0] if products else None,
            [{"label": str(d), "value": d} for d in days],
            days[0] if days else None,
        )
