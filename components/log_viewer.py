from dash import html, Input, Output
from dash.exceptions import PreventUpdate
from data import store


def layout():
    return html.Div(id="log-viewer-panel", children=[
        html.Div(id="log-timestamp", style={
            "fontWeight": "bold", "marginBottom": "8px", "fontSize": "14px",
        }),
        html.Pre(id="log-content", style={
            "whiteSpace": "pre-wrap",
            "wordBreak": "break-word",
            "fontSize": "11px",
            "maxHeight": "400px",
            "overflowY": "auto",
            "backgroundColor": "#f5f5f5",
            "padding": "8px",
            "borderRadius": "4px",
        }),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("log-timestamp", "children"),
         Output("log-content", "children")],
        Input("main-chart", "hoverData"),
    )
    def update_log(hover_data):
        if not hover_data or not store.is_loaded():
            raise PreventUpdate
        point = hover_data["points"][0]
        ts = point.get("x")
        if ts is None:
            raise PreventUpdate
        log = store.get_log_at(int(ts))
        lambda_log = log.get("lambdaLog", "") or ""
        sandbox_log = log.get("sandboxLog", "") or ""
        content_parts = []
        if lambda_log:
            content_parts.append(f"--- Lambda Log ---\n{lambda_log}")
        if sandbox_log:
            content_parts.append(f"--- Sandbox Log ---\n{sandbox_log}")
        content = "\n\n".join(content_parts) if content_parts else "(no log at this timestamp)"
        return f"Timestamp: {ts}", content
