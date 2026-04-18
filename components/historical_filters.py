from dash import html, dcc, Input, Output


def layout():
    return html.Div(id="hist-trade-filters-panel", children=[
        html.Label("Order Book Levels", style={"fontWeight": "bold"}),
        html.Div([
            dcc.Checklist(
                id="hist-level-toggles",
                options=[
                    {"label": "Bid 1", "value": "bid_1"},
                    {"label": "Bid 2", "value": "bid_2"},
                    {"label": "Bid 3", "value": "bid_3"},
                    {"label": "Ask 1", "value": "ask_1"},
                    {"label": "Ask 2", "value": "ask_2"},
                    {"label": "Ask 3", "value": "ask_3"},
                ],
                value=["bid_1", "bid_2", "bid_3", "ask_1", "ask_2", "ask_3"],
                inline=True,
                style={"fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "10px"},
            ),
        ]),
        html.Br(),
        html.Label("Overlays", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="hist-wallmid-toggle",
            options=[{"label": "Wallmid", "value": "show"}],
            value=[],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Label("Trades", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="hist-trade-toggle",
            options=[{"label": "Show Trades", "value": "show"}],
            value=["show"],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Label("Volume Filter", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="hist-volume-our-trades-toggle",
            options=[{"label": "Include Our Trades", "value": "include"}],
            value=["include"],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Div([
            html.Label("Qty Filter", style={"fontWeight": "bold"}),
            html.Button(
                "Reset",
                id="hist-qty-filter-reset",
                n_clicks=0,
                style={"marginLeft": "8px", "fontSize": "11px", "padding": "2px 6px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.RangeSlider(
            id="hist-qty-filter",
            min=0, max=100, step=1,
            value=[0, 100],
            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
            tooltip={"placement": "bottom"},
        ),
    ])


def register_callbacks(app):
    @app.callback(
        Output("hist-qty-filter", "value"),
        Input("hist-qty-filter-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_hist_qty_filter(_):
        return [0, 100]
