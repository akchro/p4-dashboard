from dash import html, dcc


def layout():
    return html.Div(id="performance-panel", children=[
        html.Label("Downsample Factor", style={"fontWeight": "bold"}),
        dcc.Slider(
            id="downsample-slider",
            min=1, max=20, step=1, value=1,
            marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
            tooltip={"placement": "bottom"},
        ),
    ])


def register_callbacks(app):
    pass  # slider is consumed by chart callbacks
