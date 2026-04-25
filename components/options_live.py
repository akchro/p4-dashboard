"""Live Options sub-tab: overlay, IV smile, IV time series.

Shares round/day with the Trading sub-tab via the existing `day-selector` and
`downsample-slider` IDs. All options-specific controls live in a local row at
the top of this sub-tab.
"""
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate

from data import store
from components import options_core
from utils import options as opts


CELL = {"border": "1px solid #ddd", "borderRadius": "4px"}


def _controls_row():
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1.4fr 0.6fr 0.7fr 0.6fr 1fr 1fr",
        "gap": "10px",
        "padding": "8px",
        "borderBottom": "1px solid #ddd",
        "alignItems": "center",
        "fontSize": "12px",
    }, children=[
        html.Div([
            html.Label("Strikes", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="opt-strike-multi",
                options=[{"label": f"VEV_{k}", "value": k} for k in opts.STRIKES],
                value=list(opts.STRIKES),
                multi=True,
                clearable=False,
            ),
        ]),
        html.Div([
            html.Label("TTE@start (d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="opt-tte-start", type="number",
                value=opts.LIVE_TTE_AT_START, min=0.1, max=30, step=0.1,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Overlay mode", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="opt-overlay-mode",
                options=[
                    {"label": "mid + floor", "value": "overlay"},
                    {"label": "extrinsic", "value": "extrinsic"},
                    {"label": "implied S", "value": "implied"},
                    {"label": "rebased", "value": "rebased"},
                ],
                value="overlay",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "8px"},
            ),
        ]),
        html.Div([
            html.Label("σ baseline (/√d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="opt-sigma-baseline", type="number",
                value=0.013, min=0.001, max=1.0, step=0.001,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Smile x-axis", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="opt-smile-x",
                options=[
                    {"label": "strike", "value": "strike"},
                    {"label": "log-mny", "value": "moneyness"},
                ],
                value="moneyness",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "8px"},
            ),
            dcc.Checklist(
                id="opt-smile-fit",
                options=[{"label": "quadratic fit", "value": "show"}],
                value=["show"],
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
            ),
        ]),
        html.Div([
            html.Label("Other", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(
                id="opt-realized",
                options=[{"label": "realized σ on IV-TS", "value": "show"}],
                value=[],
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
            ),
        ]),
    ])


def layout():
    return html.Div(style={
        "display": "grid",
        "gridTemplateRows": "auto auto 1fr 1fr",
        "height": "calc(100vh - 110px)",
        "gap": "2px",
    }, children=[
        _controls_row(),
        html.Div(style={"padding": "4px 10px 0", "fontSize": "11px", "color": "#666"},
                 children="Overlay modes — 'mid + floor': solid voucher mid vs dotted intrinsic max(S−K,0); "
                          "'extrinsic': C − max(S−K,0); 'implied S': BS-inverted underlying per voucher at "
                          "the chosen σ baseline, alongside observed S (gaps = mispricing); "
                          "'rebased': voucher mids shifted to start at S(t₀) for visual co-movement comparison "
                          "(absolute levels are NOT actual prices)."),
        html.Div(style={**CELL, "minHeight": "30vh"}, children=[
            dcc.Graph(id="opt-overlay-chart", style={"height": "100%"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "minHeight": "40vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-iv-ts-chart", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-smile-chart", style={"height": "100%"}),
            ]),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("opt-overlay-chart", "figure"),
        [Input("opt-strike-multi", "value"),
         Input("day-selector", "value"),
         Input("downsample-slider", "value"),
         Input("opt-overlay-mode", "value"),
         Input("opt-tte-start", "value"),
         Input("opt-sigma-baseline", "value")],
    )
    def _overlay(strikes, day, downsample, mode, tte_start, sigma_baseline):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_overlay_figure(
            store, strikes, day, downsample, mode=mode,
            tte_start=tte_start or opts.LIVE_TTE_AT_START,
            sigma_baseline=sigma_baseline or 0.013,
        )

    @app.callback(
        Output("opt-iv-ts-chart", "figure"),
        [Input("opt-strike-multi", "value"),
         Input("day-selector", "value"),
         Input("opt-tte-start", "value"),
         Input("downsample-slider", "value"),
         Input("opt-realized", "value")],
    )
    def _iv_ts(strikes, day, tte_start, downsample, realized):
        if not store.is_loaded():
            raise PreventUpdate
        show_realized = bool(realized) and "show" in realized
        return options_core.build_iv_ts_figure(
            store, strikes, day, tte_start or opts.LIVE_TTE_AT_START,
            downsample, show_realized,
        )

    @app.callback(
        Output("opt-smile-chart", "figure"),
        [Input("opt-strike-multi", "value"),
         Input("day-selector", "value"),
         Input("opt-tte-start", "value"),
         Input("downsample-slider", "value"),
         Input("opt-smile-x", "value"),
         Input("opt-smile-fit", "value")],
    )
    def _smile(strikes, day, tte_start, downsample, x_axis, fit_toggle):
        if not store.is_loaded():
            raise PreventUpdate
        show_fit = bool(fit_toggle) and "show" in fit_toggle
        return options_core.build_smile_figure(
            store, strikes, day, tte_start or opts.LIVE_TTE_AT_START,
            downsample, x_axis, show_fit,
        )
