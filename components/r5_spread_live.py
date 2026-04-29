"""Live Round 5 sub-tab — multi-product spread analysis (SVD pipeline).

Mirrors `r5_spread_hist` but bound to the live store. Useful when running an
algo that exposes multiple Round-5 products in the same log; lets you re-run
the SVD on the live data to confirm a candidate recipe still holds.
"""
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate

from data import store
from components import r5_spread_core as core


CELL = {"border": "1px solid #ddd", "borderRadius": "4px"}


def _controls_row():
    cat_options = [{"label": c, "value": c} for c in core.CATEGORIES.keys()]
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1.0fr 1.1fr 0.7fr 0.7fr",
        "gap": "10px",
        "padding": "8px",
        "borderBottom": "1px solid #ddd",
        "alignItems": "center",
        "fontSize": "12px",
    }, children=[
        html.Div([
            html.Label("Days (multi-select)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="r5l-days",
                multi=True,
                placeholder="All available days",
            ),
        ]),
        html.Div([
            html.Label("Focus category (spread + stability)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="r5l-focus-cat",
                options=cat_options,
                value="PEBBLES",
                clearable=False,
            ),
        ]),
        html.Div([
            html.Label("Recipe", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="r5l-recipe-mode",
                options=[
                    {"label": "integer (rounded)", "value": "int"},
                    {"label": "raw eigvec", "value": "raw"},
                ],
                value="int",
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"display": "block"},
            ),
        ]),
        html.Div([
            html.Label("Sparsify thr (global SVD)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Slider(
                id="r5l-sparsify",
                min=0.05, max=0.4, step=0.05, value=0.15,
                marks={0.05: "0.05", 0.15: "0.15", 0.25: "0.25", 0.4: "0.4"},
                tooltip={"placement": "bottom"},
            ),
        ]),
    ])


def layout():
    return html.Div(style={
        "padding": "4px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "2px",
    }, children=[
        _controls_row(),
        html.Div(style={"padding": "4px 10px 0", "fontSize": "11px", "color": "#666"},
                 children=("SVD on the live log's Round-5 mids (per-day demeaned). "
                           "Categories partially or fully missing from the log are "
                           "skipped silently. Stability check requires ≥2 days.")),
        html.Div(style={**CELL, "minHeight": "300px"}, children=[
            dcc.Graph(id="r5l-summary-table", style={"height": "300px"}),
        ]),
        html.Div(style={**CELL, "minHeight": "720px"}, children=[
            dcc.Graph(id="r5l-scree-recipe", style={"height": "720px"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1.6fr 1.0fr",
            "gap": "2px",
        }, children=[
            html.Div(style={**CELL, "minHeight": "360px"}, children=[
                dcc.Graph(id="r5l-spread-series", style={"height": "360px"}),
            ]),
            html.Div(style={**CELL, "minHeight": "360px"}, children=[
                dcc.Graph(id="r5l-stability", style={"height": "360px"}),
            ]),
        ]),
        html.Div(style={**CELL, "minHeight": "720px"}, children=[
            dcc.Graph(id="r5l-global-svd", style={"height": "720px"}),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("r5l-days", "options"),
         Output("r5l-days", "value")],
        Input("file-selector", "value"),
    )
    def _populate_days(filepath):
        if not store.is_loaded():
            raise PreventUpdate
        days = store.get_days()
        return [{"label": str(d), "value": d} for d in days], list(days)

    @app.callback(
        Output("r5l-summary-table", "figure"),
        Input("r5l-days", "value"),
    )
    def _summary(days):
        if not store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(store, days)
        return core.build_recipe_summary_table(wide)

    @app.callback(
        Output("r5l-scree-recipe", "figure"),
        Input("r5l-days", "value"),
    )
    def _scree(days):
        if not store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(store, days)
        return core.build_scree_recipe_figure(wide)

    @app.callback(
        Output("r5l-spread-series", "figure"),
        [Input("r5l-days", "value"),
         Input("r5l-focus-cat", "value"),
         Input("r5l-recipe-mode", "value")],
    )
    def _spread(days, cat, recipe_mode):
        if not store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(store, days)
        return core.build_spread_series_figure(
            wide, cat, use_int_recipe=(recipe_mode == "int"),
        )

    @app.callback(
        Output("r5l-stability", "figure"),
        [Input("r5l-days", "value"),
         Input("r5l-focus-cat", "value")],
    )
    def _stability(days, cat):
        if not store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(store, days)
        return core.build_stability_figure(wide, cat)

    @app.callback(
        Output("r5l-global-svd", "figure"),
        [Input("r5l-days", "value"),
         Input("r5l-sparsify", "value")],
    )
    def _global(days, threshold):
        if not store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(store, days)
        return core.build_global_svd_figure(
            wide, sparsify_threshold=threshold or 0.15,
        )
