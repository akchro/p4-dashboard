"""Historical Round 5 sub-tab — multi-product spread analysis (SVD pipeline).

Surfaces hidden linear constraints across the 10 categories of 5 Round 5
products. The smallest-σ eigenvector per category is the candidate basket
("recipe"); PEBBLES will appear as a sanity check, anything else is a
candidate edge.

Day selection is local to this sub-tab (multi-select, defaults to all days)
so SVD can run across days regardless of the single-day Trading sub-tab
selector.
"""
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate

from data import historical_store
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
                id="r5h-days",
                multi=True,
                placeholder="All available days",
            ),
        ]),
        html.Div([
            html.Label("Focus category (spread + stability)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="r5h-focus-cat",
                options=cat_options,
                value="PEBBLES",
                clearable=False,
            ),
        ]),
        html.Div([
            html.Label("Recipe", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="r5h-recipe-mode",
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
                id="r5h-sparsify",
                min=0.05, max=0.4, step=0.05, value=0.15,
                marks={0.05: "0.05", 0.15: "0.15", 0.25: "0.25", 0.4: "0.4"},
                tooltip={"placement": "bottom"},
            ),
        ]),
    ])


def _raw_spread_section():
    cat_options = [{"label": c, "value": c} for c in core.CATEGORIES.keys()]
    return html.Div(style={**CELL, "padding": "8px", "display": "flex",
                           "flexDirection": "column", "gap": "6px"}, children=[
        html.Div("Raw Spread Analysis", style={
            "fontWeight": "bold", "fontSize": "13px", "color": "#333",
        }),
        html.Div(("Pick a primary product in a batch, toggle other batch members "
                  "as raw price lines (left axis), and toggle pairwise spreads "
                  "vs the primary (right axis). 'Avg of selected spreads' overlays "
                  "the unweighted mean."),
                 style={"fontSize": "11px", "color": "#666"}),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "10px",
        }, children=[
            html.Div([
                html.Label("Category", style={"fontWeight": "bold", "fontSize": "11px"}),
                dcc.Dropdown(
                    id="r5h-raw-cat",
                    options=cat_options,
                    value="PANEL",
                    clearable=False,
                ),
            ]),
            html.Div([
                html.Label("Primary product", style={"fontWeight": "bold", "fontSize": "11px"}),
                dcc.Dropdown(
                    id="r5h-raw-primary",
                    clearable=False,
                ),
            ]),
        ]),
        html.Div([
            html.Label("Show price lines (other batch members)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(
                id="r5h-raw-lines",
                options=[],
                value=[],
                inline=True,
                style={"fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),
        ]),
        html.Div([
            html.Label("Show spreads (primary − other)",
                       style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(
                id="r5h-raw-spreads",
                options=[],
                value=[],
                inline=True,
                style={"fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),
        ]),
        dcc.Checklist(
            id="r5h-raw-avg",
            options=[
                {"label": "Show avg of selected spreads", "value": "show"},
                {"label": "Rebase avg onto primary (shifted to start at primary mid)",
                 "value": "rebase"},
            ],
            value=[],
            style={"fontSize": "12px"},
            inputStyle={"marginRight": "4px"},
            labelStyle={"display": "block"},
        ),
        dcc.Graph(id="r5h-raw-fig", style={"height": "440px"}),
        html.Hr(style={"margin": "8px 0", "borderColor": "#ddd"}),
        html.Div("Mean-reversion scatter (primary vs avg spread)", style={
            "fontWeight": "bold", "fontSize": "12px", "color": "#333",
        }),
        html.Div(("x = z-scored avg of selected spreads, y = primary mid, "
                  "points colored by tick. Red line is the binned conditional "
                  "mean E[primary | z]. A flat red line ⇒ primary level is "
                  "independent of the spread; a clear slope ⇒ a level "
                  "relationship worth probing for mean reversion."),
                 style={"fontSize": "11px", "color": "#666"}),
        dcc.Graph(id="r5h-meanrev-fig", style={"height": "440px"}),
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
                 children=("Per-category SVD on per-day-demeaned mids. The smallest "
                           "σ's eigenvector is the candidate basket — flat = hard "
                           "constraint. Bottom annotation shows the integer-rounded "
                           "recipe if it survives. Tier-1 candidates have σ₅/σ₁ < 0.05 "
                           "AND eigvec stays stable across days (>0.95 cosine sim). "
                           "PEBBLES should appear as the obvious confirmation case.")),
        html.Div(style={**CELL, "minHeight": "300px"}, children=[
            dcc.Graph(id="r5h-summary-table", style={"height": "300px"}),
        ]),
        html.Div(style={**CELL, "minHeight": "720px"}, children=[
            dcc.Graph(id="r5h-scree-recipe", style={"height": "720px"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1.6fr 1.0fr",
            "gap": "2px",
        }, children=[
            html.Div(style={**CELL, "minHeight": "360px"}, children=[
                dcc.Graph(id="r5h-spread-series", style={"height": "360px"}),
            ]),
            html.Div(style={**CELL, "minHeight": "360px"}, children=[
                dcc.Graph(id="r5h-stability", style={"height": "360px"}),
            ]),
        ]),
        _raw_spread_section(),
        html.Div(style={**CELL, "minHeight": "720px"}, children=[
            dcc.Graph(id="r5h-global-svd", style={"height": "720px"}),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("r5h-days", "options"),
         Output("r5h-days", "value")],
        Input("hist-round-selector", "value"),
    )
    def _populate_days(round_name):
        if not historical_store.is_loaded():
            raise PreventUpdate
        days = historical_store.get_days()
        return [{"label": str(d), "value": d} for d in days], list(days)

    @app.callback(
        Output("r5h-summary-table", "figure"),
        Input("r5h-days", "value"),
    )
    def _summary(days):
        if not historical_store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_recipe_summary_table(wide)

    @app.callback(
        Output("r5h-scree-recipe", "figure"),
        Input("r5h-days", "value"),
    )
    def _scree(days):
        if not historical_store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_scree_recipe_figure(wide)

    @app.callback(
        Output("r5h-spread-series", "figure"),
        [Input("r5h-days", "value"),
         Input("r5h-focus-cat", "value"),
         Input("r5h-recipe-mode", "value")],
    )
    def _spread(days, cat, recipe_mode):
        if not historical_store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_spread_series_figure(
            wide, cat, use_int_recipe=(recipe_mode == "int"),
        )

    @app.callback(
        Output("r5h-stability", "figure"),
        [Input("r5h-days", "value"),
         Input("r5h-focus-cat", "value")],
    )
    def _stability(days, cat):
        if not historical_store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_stability_figure(wide, cat)

    @app.callback(
        Output("r5h-global-svd", "figure"),
        [Input("r5h-days", "value"),
         Input("r5h-sparsify", "value")],
    )
    def _global(days, threshold):
        if not historical_store.is_loaded():
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_global_svd_figure(
            wide, sparsify_threshold=threshold or 0.15,
        )

    @app.callback(
        [Output("r5h-raw-primary", "options"),
         Output("r5h-raw-primary", "value")],
        Input("r5h-raw-cat", "value"),
    )
    def _raw_primary(category):
        plist = core.CATEGORIES.get(category, [])
        opts = [{"label": core.SHORT.get(p, p), "value": p} for p in plist]
        value = plist[0] if plist else None
        return opts, value

    @app.callback(
        [Output("r5h-raw-lines", "options"),
         Output("r5h-raw-lines", "value"),
         Output("r5h-raw-spreads", "options"),
         Output("r5h-raw-spreads", "value")],
        [Input("r5h-raw-cat", "value"),
         Input("r5h-raw-primary", "value")],
    )
    def _raw_toggles(category, primary):
        plist = core.CATEGORIES.get(category, [])
        others = [p for p in plist if p != primary]
        opts = [{"label": core.SHORT.get(p, p), "value": p} for p in others]
        return opts, [], opts, []

    @app.callback(
        Output("r5h-raw-fig", "figure"),
        [Input("r5h-days", "value"),
         Input("r5h-raw-cat", "value"),
         Input("r5h-raw-primary", "value"),
         Input("r5h-raw-lines", "value"),
         Input("r5h-raw-spreads", "value"),
         Input("r5h-raw-avg", "value")],
    )
    def _raw_fig(days, category, primary, lines, spreads, avg_toggle):
        if not historical_store.is_loaded():
            raise PreventUpdate
        if not category or not primary:
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        show_avg = bool(avg_toggle) and "show" in avg_toggle
        rebase_avg = bool(avg_toggle) and "rebase" in avg_toggle
        return core.build_raw_spread_figure(
            wide, category, primary,
            show_lines=lines or [],
            show_spreads=spreads or [],
            show_avg_spread=show_avg,
            rebase_avg=rebase_avg,
        )

    @app.callback(
        Output("r5h-meanrev-fig", "figure"),
        [Input("r5h-days", "value"),
         Input("r5h-raw-cat", "value"),
         Input("r5h-raw-primary", "value"),
         Input("r5h-raw-spreads", "value")],
    )
    def _meanrev_fig(days, category, primary, spreads):
        if not historical_store.is_loaded():
            raise PreventUpdate
        if not category or not primary:
            raise PreventUpdate
        wide = core.build_wide(historical_store, days)
        return core.build_meanrev_scatter_figure(
            wide, category, primary, show_spreads=spreads or [],
        )
