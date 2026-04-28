from dash import Dash, html, dcc
from components import (
    controls, trade_filters, performance_controls,
    main_chart, pnl_chart, position_chart, volume_chart, order_chart, log_viewer,
)
from components import (
    historical_controls, historical_chart, historical_volume,
    historical_filters, historical_performance,
)
from components import manual_controls, manual_round2, manual_round3, manual_round4
from components import options_live, options_historical, r3_bid_velocity, historical_player_pnl
from components import zscore_overlay

app = Dash(__name__)
app.config.suppress_callback_exceptions = True

SIDEBAR_STYLE = {
    "overflowY": "auto",
    "padding": "10px",
    "border": "1px solid #ddd",
    "borderRadius": "4px",
}

CELL_STYLE = {
    "border": "1px solid #ddd",
    "borderRadius": "4px",
}

live_trading_content = html.Div(style={
    "display": "grid",
    "gridTemplateRows": "60vh 40vh 30vh 25vh 25vh",
    "gap": "2px",
}, children=[
    html.Div(main_chart.layout(), style=CELL_STYLE),
    html.Div(volume_chart.layout(), style=CELL_STYLE),
    html.Div(order_chart.layout(), style=CELL_STYLE),
    html.Div(pnl_chart.layout(), style=CELL_STYLE),
    html.Div(position_chart.layout(), style=CELL_STYLE),
])

live_tab = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gap": "2px",
    "padding": "4px",
}, children=[
    html.Div(style={"gridColumn": "1"}, children=[
        dcc.Tabs(id="live-subtabs", value="trading", children=[
            dcc.Tab(label="Trading", value="trading", children=[live_trading_content]),
            dcc.Tab(label="Options", value="options", children=[options_live.layout()]),
        ], style={"height": "36px"}),
    ]),
    html.Div(style={
        **SIDEBAR_STYLE, "gridColumn": "2",
        "position": "sticky", "top": "40px", "height": "calc(100vh - 60px)",
    }, children=[
        log_viewer.layout(),
        html.Hr(),
        html.Div(id="pnl-overall-stats", style={
            "fontSize": "12px", "padding": "6px 0",
        }),
        html.Hr(),
        controls.layout(),
        html.Hr(),
        trade_filters.layout(),
        html.Hr(),
        zscore_overlay.layout("live"),
        html.Hr(),
        performance_controls.layout(),
    ]),
])

historical_trading_content = html.Div(style={
    "display": "grid",
    "gridTemplateRows": "60vh 40vh",
    "gap": "2px",
}, children=[
    html.Div(historical_chart.layout(), style=CELL_STYLE),
    html.Div(historical_volume.layout(), style=CELL_STYLE),
])

historical_tab = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gap": "2px",
    "padding": "4px",
}, children=[
    html.Div(style={"gridColumn": "1"}, children=[
        dcc.Tabs(id="hist-subtabs", value="trading", children=[
            dcc.Tab(label="Trading", value="trading", children=[historical_trading_content]),
            dcc.Tab(label="Options", value="options", children=[options_historical.layout()]),
            dcc.Tab(label="Player PnL", value="player_pnl", children=[historical_player_pnl.layout()]),
            dcc.Tab(label="R3 Bid Velocity", value="r3bv", children=[r3_bid_velocity.layout()]),
        ], style={"height": "36px"}),
    ]),
    html.Div(style={
        **SIDEBAR_STYLE, "gridColumn": "2",
        "position": "sticky", "top": "40px", "height": "calc(100vh - 60px)",
    }, children=[
        historical_controls.layout(),
        html.Hr(),
        historical_filters.layout(),
        html.Hr(),
        zscore_overlay.layout("hist"),
        html.Hr(),
        historical_performance.layout(),
    ]),
])

manual_tab = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gap": "2px",
    "padding": "4px",
    "minHeight": "calc(100vh - 60px)",
}, children=[
    html.Div(style={"gridColumn": "1"}, children=[
        manual_round2.charts_layout(),
        manual_round3.charts_layout(),
        manual_round4.charts_layout(),
    ]),
    html.Div(style={
        **SIDEBAR_STYLE, "gridColumn": "2",
        "position": "sticky", "top": "40px", "height": "calc(100vh - 60px)",
    }, children=[
        manual_controls.layout(),
        html.Hr(),
        manual_round2.controls_layout(),
        manual_round3.controls_layout(),
        manual_round4.controls_layout(),
    ]),
])

app.layout = html.Div(style={
    "fontFamily": "Arial, sans-serif",
}, children=[
    dcc.Tabs(id="mode-tabs", value="live", children=[
        dcc.Tab(label="Live", value="live", children=[live_tab]),
        dcc.Tab(label="Historical", value="historical", children=[historical_tab]),
        dcc.Tab(label="Manual", value="manual", children=[manual_tab]),
    ], style={"height": "40px"}),
])

# Register all callbacks
for module in [controls, main_chart, pnl_chart, position_chart, volume_chart,
               order_chart, log_viewer, trade_filters, performance_controls]:
    module.register_callbacks(app)

for module in [historical_controls, historical_chart, historical_volume,
               historical_filters, historical_performance]:
    module.register_callbacks(app)

for module in [options_live, options_historical, r3_bid_velocity, historical_player_pnl,
               zscore_overlay]:
    module.register_callbacks(app)

for module in [manual_controls, manual_round2, manual_round3, manual_round4]:
    module.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
