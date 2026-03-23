from dash import Dash, html
from components import (
    controls, trade_filters, performance_controls,
    main_chart, pnl_chart, position_chart, log_viewer,
)

app = Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(style={
    "display": "grid",
    "gridTemplateColumns": "3fr 1fr",
    "gridTemplateRows": "60vh 18vh 18vh",
    "height": "100vh",
    "gap": "4px",
    "padding": "4px",
    "fontFamily": "Arial, sans-serif",
}, children=[
    # Left column - charts
    html.Div(main_chart.layout(), style={
        "gridColumn": "1", "gridRow": "1",
        "border": "1px solid #ddd", "borderRadius": "4px",
    }),
    html.Div(pnl_chart.layout(), style={
        "gridColumn": "1", "gridRow": "2",
        "border": "1px solid #ddd", "borderRadius": "4px",
    }),
    html.Div(position_chart.layout(), style={
        "gridColumn": "1", "gridRow": "3",
        "border": "1px solid #ddd", "borderRadius": "4px",
    }),

    # Right column - controls & log
    html.Div(style={
        "gridColumn": "2", "gridRow": "1 / 4",
        "overflowY": "auto",
        "padding": "10px",
        "border": "1px solid #ddd",
        "borderRadius": "4px",
    }, children=[
        log_viewer.layout(),
        html.Hr(),
        controls.layout(),
        html.Hr(),
        trade_filters.layout(),
        html.Hr(),
        performance_controls.layout(),
    ]),
])

# Register all callbacks
for module in [controls, main_chart, pnl_chart, position_chart, log_viewer,
               trade_filters, performance_controls]:
    module.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
