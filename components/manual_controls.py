from dash import html, dcc, Input, Output

ROUNDS = [
    {"label": "Round 2 — Invest & Expand", "value": "round2"},
    {"label": "Round 3 — Celestial Gardeners", "value": "round3"},
    {"label": "Round 4 — Vanilla Just Isn't Exotic Enough", "value": "round4"},
]

R2_CONTAINER_STYLE = {
    "display": "grid",
    "gridTemplateRows": "45vh 25vh 25vh",
    "gap": "2px",
    "height": "100%",
}
R3_CONTAINER_STYLE = {
    "display": "grid",
    "gridTemplateRows": "55vh 30vh auto",
    "gap": "2px",
    "height": "100%",
}
R4_CONTAINER_STYLE = {
    "display": "grid",
    "gridTemplateRows": "32vh 38vh 28vh",
    "gap": "2px",
    "height": "100%",
}


def layout():
    return html.Div(id="manual-controls-panel", children=[
        html.Label("Round", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="manual-round-selector",
            options=ROUNDS,
            value="round4",
            clearable=False,
        ),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("manual-round2-container", "style"),
         Output("manual-round2-controls", "style"),
         Output("manual-round3-container", "style"),
         Output("manual-round3-controls", "style"),
         Output("manual-round4-container", "style"),
         Output("manual-round4-controls", "style")],
        [Input("manual-round-selector", "value")],
    )
    def toggle_round(round_val):
        show = {"display": "block"}
        hide = {"display": "none"}
        r2_visible = dict(R2_CONTAINER_STYLE)
        r2_hidden = {**R2_CONTAINER_STYLE, "display": "none"}
        r3_visible = dict(R3_CONTAINER_STYLE)
        r3_hidden = {**R3_CONTAINER_STYLE, "display": "none"}
        r4_visible = dict(R4_CONTAINER_STYLE)
        r4_hidden = {**R4_CONTAINER_STYLE, "display": "none"}
        if round_val == "round4":
            return [r2_hidden, hide, r3_hidden, hide, r4_visible, show]
        if round_val == "round3":
            return [r2_hidden, hide, r3_visible, show, r4_hidden, hide]
        return [r2_visible, show, r3_hidden, hide, r4_hidden, hide]
