from dash import html, dcc, Input, Output


def layout():
    return html.Div(id="trade-filters-panel", children=[
        html.Label("Order Book Levels", style={"fontWeight": "bold"}),
        html.Div([
            dcc.Checklist(
                id="level-toggles",
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
            id="wallmid-toggle",
            options=[{"label": "Wallmid", "value": "show"}],
            value=[],
            inputStyle={"marginRight": "4px"},
        ),
        dcc.Checklist(
            id="dashboard-wallmid-toggle",
            options=[{"label": "Dashboard Wallmid", "value": "show"}],
            value=[],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Label("Trades", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="trade-toggle",
            options=[{"label": "Show Trades", "value": "show"}],
            value=["show"],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Label("R3 Trade-Time Overlay", style={"fontWeight": "bold"}),
        html.Div(
            "Dotted vertical lines at trade times of selected products",
            style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"},
        ),
        dcc.Dropdown(
            id="r3-overlay-products",
            multi=True,
            placeholder="Select products to overlay…",
            style={"fontSize": "12px"},
        ),
        html.Br(),
        html.Label("Volume Filter", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="volume-our-trades-toggle",
            options=[{"label": "Include Our Trades", "value": "include"}],
            value=["include"],
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Div([
            html.Label("Qty Filter", style={"fontWeight": "bold"}),
            html.Button(
                "Reset",
                id="qty-filter-reset",
                n_clicks=0,
                style={"marginLeft": "8px", "fontSize": "11px", "padding": "2px 6px"},
            ),
            html.Label("Exact:", style={"marginLeft": "12px", "fontSize": "12px"}),
            dcc.Input(
                id="qty-filter-exact",
                type="number",
                min=0, step=1,
                placeholder="—",
                value=None,
                style={"width": "60px", "marginLeft": "4px", "fontSize": "12px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.RangeSlider(
            id="qty-filter",
            min=0, max=50, step=1,
            value=[0, 50],
            marks={0: "0", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
            tooltip={"placement": "bottom"},
        ),
    ])


def register_callbacks(app):
    @app.callback(
        Output("qty-filter", "value"),
        Output("qty-filter-exact", "value"),
        Input("qty-filter-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_qty_filter(_):
        return [0, 50], None

    @app.callback(
        Output("r3-overlay-products", "options"),
        Input("product-selector", "options"),
    )
    def populate_overlay_products(product_options):
        return product_options or []
