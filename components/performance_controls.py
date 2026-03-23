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
        html.Br(),
        html.Label("Volume Bucket Interval", style={"fontWeight": "bold"}),
        dcc.Slider(
            id="volume-bucket-slider",
            min=1000, max=50000, step=1000, value=10000,
            marks={1000: "1k", 10000: "10k", 25000: "25k", 50000: "50k"},
            tooltip={"placement": "bottom"},
        ),
    ])


def register_callbacks(app):
    pass  # slider is consumed by chart callbacks
