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


def _sample_clusters(clusters, n, rng):
    if not clusters:
        return rng.uniform(0, 100, size=n)
    centers = np.array([float(c[0]) for c in clusters])
    stds = np.array([max(float(c[1]), 1e-3) for c in clusters])
    weights = np.array([max(float(c[2]), 0.0) for c in clusters])
    total = weights.sum()
    if total <= 0:
        return rng.uniform(0, 100, size=n)
    weights = weights / total
    idx = rng.choice(len(centers), size=n, p=weights)
    samples = rng.normal(centers[idx], stds[idx])
    return np.clip(samples, 0, 100)


def _sample_distribution(dist_type, a, b, n, seed, clusters=None):
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
    if dist_type == "Clusters":
        return _sample_clusters(clusters or [], n, rng)
    return rng.uniform(0, 100, size=n)


def _speed_mult(sp_self, others):
    n_total = len(others) + 1
    if n_total < 2:
        return 0.9
    rank = 1 + int(np.sum(others > sp_self))
    return 0.9 - 0.8 * (rank - 1) / (n_total - 1)


def _pnl_curve(dist_type, a, b, n_players, seed, budget_cap, clusters=None):
    others = _sample_distribution(dist_type, a, b, int(n_players), seed, clusters)
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
    if dist_type == "Clusters":
        return html.Div([
            html.B("Clusters (mixture) "),
            "models multi-modal fields — e.g. profit-max + focal-point + speed-first players.",
            html.Ul(style={"margin": "4px 0", "paddingLeft": "16px"}, children=[
                html.Li("Each cluster is a truncated normal clipped to [0, 100]"),
                html.Li("Weight = relative share (normalized across clusters)"),
                html.Li("Set a cluster's weight to 0 to disable it"),
            ]),
        ])
    return ""


CLUSTER_DEFAULTS = [
    {"center": 5,  "std": 5,  "weight": 15, "color": "#2ca02c", "label": "Low (profit-max)"},
    {"center": 50, "std": 8,  "weight": 35, "color": "#1f77b4", "label": "Mid (focal / level-1)"},
    {"center": 70, "std": 8,  "weight": 40, "color": "#d62728", "label": "High (level-2 / speed)"},
]


def _cluster_block(idx, cfg):
    return html.Div(
        style={
            "borderLeft": f"3px solid {cfg['color']}",
            "paddingLeft": "8px",
            "marginBottom": "10px",
        },
        children=[
            html.Div(
                f"Cluster {idx} — {cfg['label']}",
                style={"fontWeight": "bold", "fontSize": "12px", "color": cfg["color"]},
            ),
            html.Label("Center", style={"fontSize": "11px"}),
            dcc.Slider(
                id=f"r2-c{idx}-center", min=0, max=100, step=1, value=cfg["center"],
                marks={0: "0", 50: "50", 100: "100"},
                tooltip={"placement": "bottom"},
            ),
            html.Label("Spread", style={"fontSize": "11px"}),
            dcc.Slider(
                id=f"r2-c{idx}-std", min=1, max=30, step=1, value=cfg["std"],
                marks={1: "1", 15: "15", 30: "30"},
                tooltip={"placement": "bottom"},
            ),
            html.Label("Weight", style={"fontSize": "11px"}),
            dcc.Slider(
                id=f"r2-c{idx}-weight", min=0, max=100, step=1, value=cfg["weight"],
                marks={0: "0", 50: "50", 100: "100"},
                tooltip={"placement": "bottom"},
            ),
        ],
    )


CLUSTER_INPUT_IDS = [
    "r2-c1-center", "r2-c1-std", "r2-c1-weight",
    "r2-c2-center", "r2-c2-std", "r2-c2-weight",
    "r2-c3-center", "r2-c3-std", "r2-c3-weight",
]
CLUSTER_INPUTS = [Input(cid, "value") for cid in CLUSTER_INPUT_IDS]


def _pack_clusters(c1c, c1s, c1w, c2c, c2s, c2w, c3c, c3s, c3w):
    return [(c1c, c1s, c1w), (c2c, c2s, c2w), (c3c, c3s, c3w)]


def controls_layout():
    return html.Div(id="manual-round2-controls", children=[
        html.Label("Distribution Type", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="r2-dist-type",
            options=[
                {"label": "Uniform", "value": "Uniform"},
                {"label": "Normal", "value": "Normal"},
                {"label": "Beta", "value": "Beta"},
                {"label": "Clusters (mixture)", "value": "Clusters"},
            ],
            value="Uniform",
            clearable=False,
        ),
        html.Div(id="r2-dist-description", style={
            "fontSize": "11px", "color": "#555", "padding": "6px 4px",
            "lineHeight": "1.4",
        }),
        html.Div(id="r2-single-params", children=[
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
        ]),
        html.Div(
            id="r2-cluster-params",
            style={"display": "none"},
            children=[_cluster_block(i + 1, cfg) for i, cfg in enumerate(CLUSTER_DEFAULTS)],
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
         Output("r2-dist-description", "children"),
         Output("r2-single-params", "style"),
         Output("r2-cluster-params", "style")],
        [Input("r2-dist-type", "value")],
    )
    def update_param_labels(dist_type):
        a, b = _param_label(dist_type)
        if dist_type == "Clusters":
            single_style = {"display": "none"}
            cluster_style = {"display": "block"}
        else:
            single_style = {"display": "block"}
            cluster_style = {"display": "none"}
        return a, b, _dist_description(dist_type), single_style, cluster_style

    @app.callback(
        [Output("r2-pnl-chart", "figure"),
         Output("r2-allocation-chart", "figure")],
        [Input("r2-dist-type", "value"),
         Input("r2-dist-a", "value"),
         Input("r2-dist-b", "value"),
         Input("r2-num-players", "value"),
         Input("r2-seed", "value"),
         Input("r2-budget-cap", "value"),
         *CLUSTER_INPUTS],
    )
    def update_pnl_and_allocation(dist_type, a, b, n_players, seed, budget_cap,
                                  c1c, c1s, c1w, c2c, c2s, c2w, c3c, c3s, c3w):
        clusters = _pack_clusters(c1c, c1s, c1w, c2c, c2s, c2w, c3c, c3s, c3w)
        sp_grid, pnl, r_grid, s_grid, speed_mults, _ = _pnl_curve(
            dist_type, a, b, n_players, seed, budget_cap, clusters=clusters
        )

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
         Input("r2-seed", "value"),
         *CLUSTER_INPUTS],
    )
    def update_dist_chart(dist_type, a, b, n_players, seed,
                          c1c, c1s, c1w, c2c, c2s, c2w, c3c, c3s, c3w):
        clusters = _pack_clusters(c1c, c1s, c1w, c2c, c2s, c2w, c3c, c3s, c3w)
        samples = _sample_distribution(dist_type, a, b, int(n_players), seed, clusters)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples, nbinsx=min(50, max(10, int(n_players) // 4)),
            marker={"color": "#9467bd"},
            name="Other players' speed %",
        ))
        if dist_type == "Clusters":
            total_w = sum(max(float(c[2]), 0.0) for c in clusters) or 1.0
            for i, (cfg, c) in enumerate(zip(CLUSTER_DEFAULTS, clusters)):
                center, _std, weight = c
                if float(weight) <= 0:
                    continue
                share = 100 * float(weight) / total_w
                fig.add_vline(
                    x=float(center),
                    line={"color": cfg["color"], "dash": "dash", "width": 1.5},
                    annotation_text=f"C{i+1} ({share:.0f}%)",
                    annotation_position="top",
                    annotation_font={"color": cfg["color"], "size": 10},
                )
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
