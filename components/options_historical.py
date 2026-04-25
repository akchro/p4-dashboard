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
                id="hist-opt-strike-multi",
                options=[{"label": f"VEV_{k}", "value": k} for k in opts.STRIKES],
                value=list(opts.STRIKES),
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
            html.Label("σ baseline (/√d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="hist-opt-sigma-baseline", type="number",
                value=0.013, min=0.001, max=1.0, step=0.001,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("σ_R (/√d)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="hist-opt-sigma-realized", type="number",
                value=0.018, min=0.001, max=1.0, step=0.001,
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("As-of t (blank=latest)", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Input(
                id="hist-opt-asof-ts", type="number",
                value=None, min=0, step=100,
                placeholder="latest",
                style={"width": "100%"},
            ),
        ]),
        html.Div([
            html.Label("Overlay mode", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.RadioItems(
                id="hist-opt-overlay-mode",
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
                id="hist-opt-smile-x",
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
                           "Γ-PnL tracker assumes long 1 voucher per selected strike (historical has no user trades).")),
        html.Div(style={**CELL, "height": "32vh"}, children=[
            dcc.Graph(id="hist-opt-overlay-chart", style={"height": "100%"}),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "32vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-iv-ts-chart", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-smile-chart", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "32vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-terminal-dist", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-greeks-per-dollar", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "2px",
            "height": "34vh",
        }, children=[
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-fly-mispricing", style={"height": "100%"}),
            ]),
            html.Div(style=CELL, children=[
                dcc.Graph(id="hist-opt-pop-structures", style={"height": "100%"}),
            ]),
        ]),
        html.Div(style={**CELL, "height": "32vh"}, children=[
            dcc.Graph(id="hist-opt-gamma-pnl-tracker", style={"height": "100%"}),
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
         Input("hist-opt-overlay-mode", "value"),
         Input("hist-opt-tte-start", "value"),
         Input("hist-opt-sigma-baseline", "value")],
    )
    def _overlay(strikes, day, downsample, mode, tte_start, sigma_baseline):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_overlay_figure(
            historical_store, strikes, day, downsample, mode=mode,
            tte_start=tte_start or 8.0,
            sigma_baseline=sigma_baseline or 0.013,
        )

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
        [Input("hist-opt-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-opt-tte-start", "value"),
         Input("hist-downsample-slider", "value"),
         Input("hist-opt-smile-x", "value"),
         Input("hist-opt-smile-fit", "value")],
    )
    def _smile(strikes, day, tte_start, downsample, x_axis, fit_toggle):
        if not historical_store.is_loaded():
            raise PreventUpdate
        show_fit = bool(fit_toggle) and "show" in fit_toggle
        return options_core.build_smile_figure(
            historical_store, strikes, day, tte_start or 8.0,
            downsample, x_axis, show_fit,
        )

    @app.callback(
        Output("hist-opt-terminal-dist", "figure"),
        [Input("hist-opt-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-opt-asof-ts", "value"),
         Input("hist-opt-sigma-realized", "value"),
         Input("hist-opt-tte-start", "value")],
    )
    def _terminal_dist(strikes, day, as_of_ts, sigma_r, tte_start):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_terminal_dist_figure(
            historical_store, strikes, day, as_of_ts,
            sigma_r or 0.018, tte_start or 8.0,
        )

    @app.callback(
        Output("hist-opt-fly-mispricing", "figure"),
        [Input("hist-day-selector", "value"),
         Input("hist-opt-asof-ts", "value"),
         Input("hist-opt-sigma-realized", "value"),
         Input("hist-opt-tte-start", "value")],
    )
    def _fly_mispricing(day, as_of_ts, sigma_r, tte_start):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_fly_mispricing_figure(
            historical_store, day, as_of_ts,
            sigma_r or 0.018, tte_start or 8.0,
        )

    @app.callback(
        Output("hist-opt-greeks-per-dollar", "figure"),
        [Input("hist-day-selector", "value"),
         Input("hist-opt-asof-ts", "value"),
         Input("hist-opt-sigma-realized", "value"),
         Input("hist-opt-tte-start", "value")],
    )
    def _greeks_per_dollar(day, as_of_ts, sigma_r, tte_start):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_greeks_per_dollar_figure(
            historical_store, day, as_of_ts,
            sigma_r or 0.018, tte_start or 8.0,
        )

    @app.callback(
        Output("hist-opt-pop-structures", "figure"),
        [Input("hist-day-selector", "value"),
         Input("hist-opt-asof-ts", "value"),
         Input("hist-opt-sigma-realized", "value"),
         Input("hist-opt-tte-start", "value")],
    )
    def _pop_structures(day, as_of_ts, sigma_r, tte_start):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_pop_per_structure_figure(
            historical_store, day, as_of_ts,
            sigma_r or 0.018, tte_start or 8.0,
        )

    @app.callback(
        Output("hist-opt-gamma-pnl-tracker", "figure"),
        [Input("hist-opt-strike-multi", "value"),
         Input("hist-day-selector", "value"),
         Input("hist-opt-sigma-realized", "value"),
         Input("hist-opt-tte-start", "value"),
         Input("hist-downsample-slider", "value")],
    )
    def _gamma_tracker(strikes, day, sigma_r, tte_start, downsample):
        if not historical_store.is_loaded():
            raise PreventUpdate
        return options_core.build_gamma_pnl_tracker_figure(
            historical_store, strikes, day,
            sigma_r or 0.018, tte_start or 8.0,
            downsample=downsample or 1,
        )
