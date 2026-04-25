"""Historical Options sub-tab — mirrors options_live but bound to historical_store.

Reuses `hist-day-selector` and `hist-downsample-slider` from the Trading sub-tab
sidebar; all options-specific controls are local to this sub-tab. The TTE
default auto-populates from the selected (round, day).
"""
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate

from data import historical_store
from components import options_core
from utils import options as opts


CELL = {"border": "1px solid #ddd", "borderRadius": "4px"}


def _controls_row():
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1.5fr 0.7fr 0.7fr 1fr 1fr",
        "gap": "10px",
        "padding": "8px",
        "borderBottom": "1px solid #ddd",
        "alignItems": "center",
        "fontSize": "12px",
    }, children=[
        html.Div([
            html.Label("Strikes", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(
                id="hist-opt-strike-multi",
                options=[{"label": f"VEV_{k}", "value": k} for k in opts.STRIKES],
                value=[5200, 5300],
                multi=True,
                clearable=False,
            ),
        ]),
        html.Div([
            html.Label("TTE@start (d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="hist-opt-tte-start", type="number",
                value=8.0, min=0.1, max=30, step=0.1,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Overlay mode", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="hist-opt-overlay-mode",
                options=[
                    {"label": "mid + floor", "value": "overlay"},
                    {"label": "extrinsic", "value": "extrinsic"},
                ],
                value="overlay",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "8px"},
            ),
        ]),
        html.Div([
            html.Label("Smile x-axis", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="hist-opt-smile-x",
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
                id="hist-opt-smile-fit",
                options=[{"label": "quadratic fit", "value": "show"}],
                value=["show"],
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
            ),
        ]),
        html.Div([
            html.Label("Other", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(
                id="hist-opt-realized",
                options=[{"label": "realized σ on IV-TS", "value": "show"}],
                value=[],
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
            ),
        ]),
    ])


def _smile_ts_row():
    return html.Div([
        html.Label("Smile timestamp", style={"fontSize": "11px", "fontWeight": "bold",
                                              "marginRight": "10px"}),
        dcc.Slider(
            id="hist-opt-smile-ts", min=0, max=opts.TIMESTAMP_PER_DAY - 100, step=100,
            value=opts.TIMESTAMP_PER_DAY // 2,
            marks={i * opts.TIMESTAMP_PER_DAY // 10: f"{i}" for i in range(11)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], style={"padding": "0 14px 4px"})


def layout():
    return html.Div(style={
        "display": "grid",
        "gridTemplateRows": "auto auto 1fr auto 1fr",
        "height": "calc(100vh - 110px)",
        "gap": "2px",
    }, children=[
        _controls_row(),
        html.Div(style={"padding": "4px 10px 0", "fontSize": "11px", "color": "#666"},
                 children="Overlay — voucher mid vs intrinsic floor per selected strike. "
                          "Dotted = max(S−K, 0); solid = voucher mid. Gap = extrinsic."),
        html.Div(style={**CELL, "minHeight": "30vh"}, children=[
            dcc.Graph(id="hist-opt-overlay-chart", style={"height": "100%"}),
        ]),
        _smile_ts_row(),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "minHeight": "40vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-iv-ts-chart", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-smile-chart", style={"height": "100%"}),
            ]),
        ]),
    ])


def register_callbacks(app):
    # Auto-populate TTE default when (round, day) changes.
    @app.callback(
        Output("hist-opt-tte-start", "value"),
        [Input("hist-round-selector", "value"),
         Input("hist-day-selector", "value")],
        prevent_initial_call=True,
    )
    def _auto_tte(round_name, day):
        if round_name is None or day is None:
            raise PreventUpdate
        return opts.tte_for_day(round_name, day)

    @app.callback(
        Output("hist-opt-overlay-chart", "figure"),
        [Input("hist-opt-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-downsample-slider", "value"),
         Input("hist-opt-overlay-mode", "value")],
    )
    def _overlay(strikes, day, downsample, mode):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_overlay_figure(historical_store, strikes, day, downsample, mode=mode)

    @app.callback(
        Output("hist-opt-iv-ts-chart", "figure"),
        [Input("hist-opt-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-opt-tte-start", "value"),
         Input("hist-downsample-slider", "value"),
         Input("hist-opt-realized", "value")],
    )
    def _iv_ts(strikes, day, tte_start, downsample, realized):
        if not historical_store.is_loaded():
            raise PreventUpdate
        show_realized = bool(realized) and "show" in realized
        return options_core.build_iv_ts_figure(
            historical_store, strikes, day, tte_start or 8.0,
            downsample, show_realized,
        )

    @app.callback(
        Output("hist-opt-smile-chart", "figure"),
        [Input("hist-opt-smile-ts", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-opt-tte-start", "value"),
         Input("hist-opt-smile-x", "value"),
         Input("hist-opt-smile-fit", "value")],
    )
    def _smile(ts, day, tte_start, x_axis, fit_toggle):
        if not historical_store.is_loaded():
            raise PreventUpdate
        show_fit = bool(fit_toggle) and "show" in fit_toggle
        return options_core.build_smile_figure(
            historical_store, ts, day, tte_start or 8.0,
            x_axis, show_fit,
        )
