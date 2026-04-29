from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from data import store


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
        dcc.Checklist(
            id="bull-signal-toggle",
            options=[{"label": "Bull Signal", "value": "show"}],
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
        html.Div([
            html.Label("Traders", style={"fontWeight": "bold"}),
            html.Button(
                "All", id="traders-all", n_clicks=0,
                style={"marginLeft": "8px", "fontSize": "11px", "padding": "2px 6px"},
            ),
            html.Button(
                "None", id="traders-none", n_clicks=0,
                style={"marginLeft": "4px", "fontSize": "11px", "padding": "2px 6px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(
            "Filter by buyer/seller identity",
            style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"},
        ),
        dcc.Checklist(
            id="trader-toggles",
            options=[],
            value=[],
            style={"fontSize": "12px"},
            inputStyle={"marginRight": "4px"},
            labelStyle={"display": "block"},
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
        dcc.Checklist(
            id="r3-overlay-points-toggle",
            options=[{"label": "Show price markers (rebased to main mid)", "value": "show"}],
            value=[],
            style={"fontSize": "12px", "marginTop": "4px"},
            inputStyle={"marginRight": "4px"},
        ),
        html.Br(),
        html.Label("Product Price Overlay", style={"fontWeight": "bold"}),
        html.Div(
            "Mid-price lines of other products on the same chart",
            style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"},
        ),
        dcc.Dropdown(
            id="product-overlay-list",
            multi=True,
            placeholder="Select products to overlay mids…",
            style={"fontSize": "12px"},
        ),
        dcc.RadioItems(
            id="product-overlay-mode",
            options=[
                {"label": "Rebased (shifted to start at main mid)", "value": "rebased"},
                {"label": "Right axis (raw prices on y2)", "value": "y2"},
            ],
            value="rebased",
            style={"fontSize": "12px", "marginTop": "4px"},
            inputStyle={"marginRight": "4px"},
            labelStyle={"display": "block"},
        ),
        dcc.Checklist(
            id="product-overlay-sum-toggle",
            options=[{"label": "Show sum of overlay mids", "value": "show"}],
            value=[],
            style={"fontSize": "12px", "marginTop": "4px"},
            inputStyle={"marginRight": "4px"},
        ),
        dcc.Checklist(
            id="product-overlay-spread-toggle",
            options=[
                {"label": "Show spread (current − first overlay)", "value": "show"},
                {"label": "Absolute value", "value": "abs"},
            ],
            value=[],
            style={"fontSize": "12px", "marginTop": "4px"},
            inputStyle={"marginRight": "4px"},
            labelStyle={"display": "block"},
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

    @app.callback(
        Output("product-overlay-list", "options"),
        Input("product-selector", "options"),
    )
    def populate_product_overlay(product_options):
        return product_options or []

    @app.callback(
        [Output("trader-toggles", "options"),
         Output("trader-toggles", "value")],
        Input("product-selector", "options"),
    )
    def populate_traders(_product_options):
        # Re-run after a log loads (product list update triggers this).
        if not store.is_loaded():
            raise PreventUpdate
        traders = store.get_traders()
        opts = [{"label": t, "value": t} for t in traders]
        return opts, list(traders)

    @app.callback(
        Output("trader-toggles", "value", allow_duplicate=True),
        [Input("traders-all", "n_clicks"),
         Input("traders-none", "n_clicks")],
        State("trader-toggles", "options"),
        prevent_initial_call=True,
    )
    def toggle_all_traders(all_clicks, none_clicks, options):
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if "traders-all" in triggered:
            return [o["value"] for o in (options or [])]
        if "traders-none" in triggered:
            return []
        raise PreventUpdate
