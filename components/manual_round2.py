import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

COST_PER_PCT = 500
LOG101 = np.log(101)


def _research(r):
    return 200_000 * np.log(1 + r) / LOG101


def _scale(s):
    return 7 * s / 100


def _optimal_rs(budget_left):
    if budget_left <= 0:
        return 0.0, 0.0, 0.0
    gr = (np.sqrt(5) + 1) / 2
    a, b = 0.0, float(budget_left)
    tol = 1e-6
    while b - a > tol:
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        if _research(c) * _scale(budget_left - c) < _research(d) * _scale(budget_left - d):
            a = c
        else:
            b = d
    r = (a + b) / 2
    s = budget_left - r
    return r, s, _research(r) * _scale(s)


def _sample_distribution(dist_type, a, b, n, seed):
    rng = np.random.default_rng(int(seed))
    if dist_type == "Uniform":
        lo, hi = sorted([float(a), float(b)])
        if hi <= lo:
            return np.full(n, lo)
        return rng.uniform(lo, hi, size=n)
    if dist_type == "Normal":
        mean, std = float(a), max(float(b), 1e-3)
        x = rng.normal(mean, std, size=n)
        return np.clip(x, 0, 100)
    if dist_type == "Beta":
        alpha, beta = max(float(a), 0.1), max(float(b), 0.1)
        return rng.beta(alpha, beta, size=n) * 100
    return rng.uniform(0, 100, size=n)


def _speed_mult(sp_self, others):
    n_total = len(others) + 1
    if n_total < 2:
        return 0.9
    rank = 1 + int(np.sum(others > sp_self))
    return 0.9 - 0.8 * (rank - 1) / (n_total - 1)


def _pnl_curve(dist_type, a, b, n_players, seed, budget_cap):
    others = _sample_distribution(dist_type, a, b, int(n_players), seed)
    sp_grid = np.arange(0, int(budget_cap) + 1, 1, dtype=float)
    pnl = np.zeros_like(sp_grid)
    r_grid = np.zeros_like(sp_grid)
    s_grid = np.zeros_like(sp_grid)
    speed_mults = np.zeros_like(sp_grid)
    for i, sp in enumerate(sp_grid):
        budget_left = budget_cap - sp
        r, s, prod = _optimal_rs(budget_left)
        m = _speed_mult(sp, others)
        speed_mults[i] = m
        r_grid[i] = r
        s_grid[i] = s
        pnl[i] = prod * m - COST_PER_PCT * (r + s + sp)
    return sp_grid, pnl, r_grid, s_grid, speed_mults, others


def _param_label(dist_type):
    return {
        "Uniform": ("Min", "Max"),
        "Normal": ("Mean", "Std"),
        "Beta": ("Alpha (α)", "Beta (β)"),
    }.get(dist_type, ("Param A", "Param B"))


def _dist_description(dist_type):
    from dash import html
    if dist_type == "Uniform":
        return html.Span("Every speed value between Min and Max is equally likely.")
    if dist_type == "Normal":
        return html.Span(
            "Bell curve centered at Mean with spread Std (clipped to [0, 100])."
        )
    if dist_type == "Beta":
        return html.Div([
            html.B("Beta(α, β) "), "is a flexible distribution on [0, 100]:",
            html.Ul(style={"margin": "4px 0", "paddingLeft": "16px"}, children=[
                html.Li("α = β = 1 → uniform"),
                html.Li("α > β → skewed toward 100 (most invest heavily)"),
                html.Li("β > α → skewed toward 0 (most underinvest)"),
                html.Li("α = β > 1 → bell-shaped, centered at 50"),
                html.Li("α = β < 1 → bathtub, mass at extremes"),
                html.Li("mean = α / (α + β) × 100"),
            ]),
            "Useful for modeling fields where players cluster or split.",
        ])
    return ""


def controls_layout():
    return html.Div(id="manual-round2-controls", children=[
        html.Label("Distribution Type", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="r2-dist-type",
            options=[
                {"label": "Uniform", "value": "Uniform"},
                {"label": "Normal", "value": "Normal"},
                {"label": "Beta", "value": "Beta"},
            ],
            value="Uniform",
            clearable=False,
        ),
        html.Div(id="r2-dist-description", style={
            "fontSize": "11px", "color": "#555", "padding": "6px 4px",
            "lineHeight": "1.4",
        }),
        html.Br(),
        html.Label(id="r2-label-a", style={"fontWeight": "bold"}, children="Min"),
        dcc.Slider(
            id="r2-dist-a", min=0, max=100, step=1, value=0,
            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
            tooltip={"placement": "bottom"},
        ),
        html.Br(),
        html.Label(id="r2-label-b", style={"fontWeight": "bold"}, children="Max"),
        dcc.Slider(
            id="r2-dist-b", min=0, max=100, step=1, value=100,
            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
            tooltip={"placement": "bottom"},
        ),
        html.Br(),
        html.Label("Number of Other Players", style={"fontWeight": "bold"}),
        dcc.Slider(
            id="r2-num-players", min=2, max=200, step=1, value=20,
            marks={2: "2", 20: "20", 50: "50", 100: "100", 200: "200"},
            tooltip={"placement": "bottom"},
        ),
        html.Br(),
        html.Label("Budget Cap (%)", style={"fontWeight": "bold"}),
        dcc.Slider(
            id="r2-budget-cap", min=10, max=100, step=1, value=100,
            marks={10: "10", 50: "50", 100: "100"},
            tooltip={"placement": "bottom"},
        ),
        html.Br(),
        html.Label("Sample Seed", style={"fontWeight": "bold"}),
        dcc.Slider(
            id="r2-seed", min=0, max=99, step=1, value=0,
            marks={0: "0", 50: "50", 99: "99"},
            tooltip={"placement": "bottom"},
        ),
    ])


def charts_layout():
    return html.Div(
        id="manual-round2-container",
        style={
            "display": "grid",
            "gridTemplateRows": "45vh 25vh 25vh",
            "gap": "2px",
            "height": "100%",
        },
        children=[
            html.Div(dcc.Graph(id="r2-pnl-chart", style={"height": "100%"}),
                     style={"border": "1px solid #ddd", "borderRadius": "4px"}),
            html.Div(dcc.Graph(id="r2-dist-chart", style={"height": "100%"}),
                     style={"border": "1px solid #ddd", "borderRadius": "4px"}),
            html.Div(dcc.Graph(id="r2-allocation-chart", style={"height": "100%"}),
                     style={"border": "1px solid #ddd", "borderRadius": "4px"}),
        ],
    )


def register_callbacks(app):
    @app.callback(
        [Output("r2-label-a", "children"),
         Output("r2-label-b", "children"),
         Output("r2-dist-description", "children")],
        [Input("r2-dist-type", "value")],
    )
    def update_param_labels(dist_type):
        a, b = _param_label(dist_type)
        return a, b, _dist_description(dist_type)

    @app.callback(
        [Output("r2-pnl-chart", "figure"),
         Output("r2-allocation-chart", "figure")],
        [Input("r2-dist-type", "value"),
         Input("r2-dist-a", "value"),
         Input("r2-dist-b", "value"),
         Input("r2-num-players", "value"),
         Input("r2-seed", "value"),
         Input("r2-budget-cap", "value")],
    )
    def update_pnl_and_allocation(dist_type, a, b, n_players, seed, budget_cap):
        sp_grid, pnl, r_grid, s_grid, speed_mults, _ = _pnl_curve(
            dist_type, a, b, n_players, seed, budget_cap
        )

        # PnL chart with argmax annotation
        idx_best = int(np.argmax(pnl))
        sp_best = sp_grid[idx_best]
        r_best, s_best = r_grid[idx_best], s_grid[idx_best]
        pnl_best = pnl[idx_best]
        mult_best = speed_mults[idx_best]

        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(
            x=sp_grid, y=pnl, mode="lines", name="PnL",
            line={"color": "#1f77b4", "width": 2.5},
            customdata=np.stack([r_grid, s_grid, speed_mults], axis=-1),
            hovertemplate=(
                "speed=%{x:.0f}%<br>PnL=%{y:,.0f}<br>"
                "research=%{customdata[0]:.1f}%<br>"
                "scale=%{customdata[1]:.1f}%<br>"
                "speed_mult=%{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        ))
        pnl_fig.add_vline(x=sp_best, line={"color": "red", "dash": "dash"})
        pnl_fig.add_annotation(
            x=sp_best, y=pnl_best,
            text=(f"<b>optimum</b><br>sp={sp_best:.0f}%, r={r_best:.1f}%, s={s_best:.1f}%"
                  f"<br>mult={mult_best:.3f}<br>PnL={pnl_best:,.0f}"),
            showarrow=True, arrowhead=2, bgcolor="white", bordercolor="red",
        )
        pnl_fig.update_layout(
            title=f"PnL vs Your Speed Investment (distribution: {dist_type})",
            xaxis_title="Your Speed Investment (%)",
            yaxis_title="PnL (XIRECs)",
            hovermode="x unified",
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            template="plotly_white",
        )

        # Allocation chart
        alloc_fig = go.Figure()
        alloc_fig.add_trace(go.Scatter(
            x=sp_grid, y=r_grid, mode="lines", name="Research %",
            line={"color": "#2ca02c", "width": 2},
        ))
        alloc_fig.add_trace(go.Scatter(
            x=sp_grid, y=s_grid, mode="lines", name="Scale %",
            line={"color": "#ff7f0e", "width": 2},
        ))
        alloc_fig.add_trace(go.Scatter(
            x=sp_grid, y=sp_grid, mode="lines", name="Speed %",
            line={"color": "#d62728", "width": 2, "dash": "dot"},
        ))
        alloc_fig.update_layout(
            title="Optimal Research / Scale Split vs Your Speed Investment",
            xaxis_title="Your Speed Investment (%)",
            yaxis_title="Allocation (%)",
            hovermode="x unified",
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            template="plotly_white",
            legend={"orientation": "h", "y": -0.2},
        )

        return pnl_fig, alloc_fig

    @app.callback(
        Output("r2-dist-chart", "figure"),
        [Input("r2-dist-type", "value"),
         Input("r2-dist-a", "value"),
         Input("r2-dist-b", "value"),
         Input("r2-num-players", "value"),
         Input("r2-seed", "value")],
    )
    def update_dist_chart(dist_type, a, b, n_players, seed):
        samples = _sample_distribution(dist_type, a, b, int(n_players), seed)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples, nbinsx=min(50, max(10, int(n_players) // 4)),
            marker={"color": "#9467bd"},
            name="Other players' speed %",
        ))
        fig.update_layout(
            title=f"Other Players' Speed Distribution ({dist_type}, n={int(n_players)})",
            xaxis_title="Speed Investment (%)",
            yaxis_title="Count",
            xaxis={"range": [0, 100]},
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            template="plotly_white",
            showlegend=False,
        )
        return fig
