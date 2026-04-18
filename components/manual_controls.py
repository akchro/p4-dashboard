from dash import html, dcc, Input, Output

ROUNDS = [
    {"label": "Round 2 — Invest & Expand", "value": "round2"},
]


def layout():
    return html.Div(id="manual-controls-panel", children=[
        html.Label("Round", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="manual-round-selector",
            options=ROUNDS,
            value="round2",
            clearable=False,
        ),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("manual-round2-container", "style"),
         Output("manual-round2-controls", "style")],
        [Input("manual-round-selector", "value")],
    )
    def toggle_round(round_val):
        show = {"display": "block"}
        hide = {"display": "none"}
        round2_style = show if round_val == "round2" else hide
        return [round2_style, round2_style]
