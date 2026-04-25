"""Live Options sub-tab: overlay, IV smile, IV time series, plus strategy
panels (terminal-S forecast, fly mispricing, Γ/$, P(profit), Γ-PnL tracker).

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
        "gridTemplateColumns": "1.4fr 0.55fr 0.55fr 0.55fr 0.7fr 0.55fr 0.95fr 0.95fr",
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
            html.Label("σ baseline (/√d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="opt-sigma-baseline", type="number",
                value=0.013, min=0.001, max=1.0, step=0.001,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("σ_R (/√d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="opt-sigma-realized", type="number",
                value=0.018, min=0.001, max=1.0, step=0.001,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("As-of t (blank=latest)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="opt-asof-ts", type="number",
                value=None, min=0, step=100,
                placeholder="latest",
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Overlay mode", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="opt-overlay-mode",
                options=[
                    {"label": "mid+floor", "value": "overlay"},
                    {"label": "extr.", "value": "extrinsic"},
                    {"label": "impl.S", "value": "implied"},
                    {"label": "rebased", "value": "rebased"},
                ],
                value="overlay",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "6px"},
            ),
        ]),
        html.Div([
            html.Label("Smile x-axis", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="opt-smile-x",
                options=[
                    {"label": "K", "value": "strike"},
                    {"label": "ln(K/S)/√T", "value": "moneyness"},
                    {"label": "ln(S/K)/√T", "value": "moneyness_sk"},
                ],
                value="moneyness",
                inline=True,
                style={"fontSize": "11px"},
                inputStyle={"marginRight": "3px"},
                labelStyle={"marginRight": "6px"},
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
        "padding": "4px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "2px",
    }, children=[
        _controls_row(),
        html.Div(style={"padding": "4px 10px 0", "fontSize": "11px", "color": "#666"},
                 children=("Overlay modes — mid+floor / extrinsic / implied-S / rebased; "
                           "σ baseline drives implied-S, σ_R drives the strategy panels "
                           "(terminal distribution, fair fly cost, P(profit)). "
                           "'As-of t' freezes the snapshot timestamp; leave blank to use the latest tick. "
                           "Γ-PnL tracker assumes long 1 voucher per selected strike when no user trades exist.")),
        html.Div(style={**CELL, "height": "32vh"}, children=[
            dcc.Graph(id="opt-overlay-chart", style={"height": "100%"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "32vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-iv-ts-chart", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-smile-chart", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "32vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-terminal-dist", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-greeks-per-dollar", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "34vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-fly-mispricing", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="opt-pop-structures", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={**CELL, "height": "32vh"}, children=[
            dcc.Graph(id="opt-gamma-pnl-tracker", style={"height": "100%"}),
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

    @app.callback(
        Output("opt-terminal-dist", "figure"),
        [Input("opt-strike-multi", "value"),
         Input("day-selector", "value"),
         Input("opt-asof-ts", "value"),
         Input("opt-sigma-realized", "value"),
         Input("opt-tte-start", "value")],
    )
    def _terminal_dist(strikes, day, as_of_ts, sigma_r, tte_start):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_terminal_dist_figure(
            store, strikes, day, as_of_ts,
            sigma_r or 0.018, tte_start or opts.LIVE_TTE_AT_START,
        )

    @app.callback(
        Output("opt-fly-mispricing", "figure"),
        [Input("day-selector", "value"),
         Input("opt-asof-ts", "value"),
         Input("opt-sigma-realized", "value"),
         Input("opt-tte-start", "value")],
    )
    def _fly_mispricing(day, as_of_ts, sigma_r, tte_start):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_fly_mispricing_figure(
            store, day, as_of_ts,
            sigma_r or 0.018, tte_start or opts.LIVE_TTE_AT_START,
        )

    @app.callback(
        Output("opt-greeks-per-dollar", "figure"),
        [Input("day-selector", "value"),
         Input("opt-asof-ts", "value"),
         Input("opt-sigma-realized", "value"),
         Input("opt-tte-start", "value")],
    )
    def _greeks_per_dollar(day, as_of_ts, sigma_r, tte_start):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_greeks_per_dollar_figure(
            store, day, as_of_ts,
            sigma_r or 0.018, tte_start or opts.LIVE_TTE_AT_START,
        )

    @app.callback(
        Output("opt-pop-structures", "figure"),
        [Input("day-selector", "value"),
         Input("opt-asof-ts", "value"),
         Input("opt-sigma-realized", "value"),
         Input("opt-tte-start", "value")],
    )
    def _pop_structures(day, as_of_ts, sigma_r, tte_start):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_pop_per_structure_figure(
            store, day, as_of_ts,
            sigma_r or 0.018, tte_start or opts.LIVE_TTE_AT_START,
        )

    @app.callback(
        Output("opt-gamma-pnl-tracker", "figure"),
        [Input("opt-strike-multi", "value"),
         Input("day-selector", "value"),
         Input("opt-sigma-realized", "value"),
         Input("opt-tte-start", "value"),
         Input("downsample-slider", "value")],
    )
    def _gamma_tracker(strikes, day, sigma_r, tte_start, downsample):
        if not store.is_loaded():
            raise PreventUpdate
        return options_core.build_gamma_pnl_tracker_figure(
            store, strikes, day,
            sigma_r or 0.018, tte_start or opts.LIVE_TTE_AT_START,
            downsample=downsample or 1,
        )
