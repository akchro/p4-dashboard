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
                id="opt-strike-multi",
                options=[{"label": f"VEV_{k}", "value": k} for k in opts.STRIKES],
                value=[5200, 5300],
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


def _smile_ts_row():
    return html.Div([
        html.Label("Smile timestamp", style={"fontSize": "11px", "fontWeight": "bold",
                                              "marginRight": "10px"}),
        dcc.Slider(
            id="opt-smile-ts", min=0, max=opts.TIMESTAMP_PER_DAY - 100, step=100,
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
            dcc.Graph(id="opt-overlay-chart", style={"height": "100%"}),
        ]),
        _smile_ts_row(),
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
         Input("opt-overlay-mode", "value")],
    )
    def _overlay(strikes, day, downsample, mode):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_overlay_figure(store, strikes, day, downsample, mode=mode)

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
        [Input("opt-smile-ts", "value"),
         Input("day-selector", "value"),
         Input("opt-tte-start", "value"),
         Input("opt-smile-x", "value"),
         Input("opt-smile-fit", "value")],
    )
    def _smile(ts, day, tte_start, x_axis, fit_toggle):
        if not store.is_loaded():
            raise PreventUpdate
        show_fit = bool(fit_toggle) and "show" in fit_toggle
        return options_core.build_smile_figure(
            store, ts, day, tte_start or opts.LIVE_TTE_AT_START,
            x_axis, show_fit,
        )
