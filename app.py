from dash import Dash, html, dcc
from components import (
    controls, trade_filters, performance_controls,
    main_chart, pnl_chart, position_chart, volume_chart, order_chart, log_viewer,
)
from components import (
    historical_controls, historical_chart, historical_volume,
    historical_filters, historical_performance,
)

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

live_tab = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gridTemplateRows": "60vh 40vh 30vh 25vh 25vh",
    "gap": "2px",
    "padding": "4px",
}, children=[
    html.Div(main_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "1",
    }),
    html.Div(volume_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "2",
    }),
    html.Div(order_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "3",
    }),
    html.Div(pnl_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "4",
    }),
    html.Div(position_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "5",
    }),
    html.Div(style={
        **SIDEBAR_STYLE, "gridColumn": "2", "gridRow": "1 / 6",
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
        performance_controls.layout(),
    ]),
])

historical_tab = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gridTemplateRows": "60vh 40vh",
    "gap": "2px",
    "padding": "4px",
}, children=[
    html.Div(historical_chart.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "1",
    }),
    html.Div(historical_volume.layout(), style={
        **CELL_STYLE, "gridColumn": "1", "gridRow": "2",
    }),
    html.Div(style={
        **SIDEBAR_STYLE, "gridColumn": "2", "gridRow": "1 / 3",
        "position": "sticky", "top": "40px", "height": "calc(100vh - 60px)",
    }, children=[
        historical_controls.layout(),
        html.Hr(),
        historical_filters.layout(),
        html.Hr(),
        historical_performance.layout(),
    ]),
])

app.layout = html.Div(style={
    "fontFamily": "Arial, sans-serif",
}, children=[
    dcc.Tabs(id="mode-tabs", value="live", children=[
        dcc.Tab(label="Live", value="live", children=[live_tab]),
        dcc.Tab(label="Historical", value="historical", children=[historical_tab]),
    ], style={"height": "40px"}),
])

# Register all callbacks
for module in [controls, main_chart, pnl_chart, position_chart, volume_chart,
               order_chart, log_viewer, trade_filters, performance_controls]:
    module.register_callbacks(app)

for module in [historical_controls, historical_chart, historical_volume,
               historical_filters, historical_performance]:
    module.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
