from dash import html, dcc, Input, Output, callback_context
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div(id="controls-panel", children=[
        html.Label("Log File", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="file-selector", clearable=False),
        html.Br(),
        html.Label("Product", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="product-selector", clearable=False),
        html.Br(),
        html.Label("Day", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="day-selector", clearable=False),
    ])


def register_callbacks(app):
    @app.callback(
        Output("file-selector", "options"),
        Input("file-selector", "id"),  # fires once on load
    )
    def populate_files(_):
        files = store.get_log_files()
        return [{"label": f, "value": f} for f in files]

    @app.callback(
        [Output("product-selector", "options"),
         Output("product-selector", "value"),
         Output("day-selector", "options"),
         Output("day-selector", "value")],
        Input("file-selector", "value"),
    )
    def load_file(filepath):
        if not filepath:
            raise PreventUpdate
        store.load(filepath)
        products = store.get_products()
        days = store.get_days()
        return (
            [{"label": p, "value": p} for p in products],
            products[0] if products else None,
            [{"label": str(d), "value": d} for d in days],
            days[0] if days else None,
        )
