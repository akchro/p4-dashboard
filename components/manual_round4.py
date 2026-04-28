"""Round 4 manual: Vanilla Just Isn't Exotic Enough.

GBM-pricing sandbox for AETHER_CRYSTAL plus 11 derivatives. Each contract has
a 3000 PnL multiplier. The server marks at expiry to the *average payoff
across only 100 GBM sims* (per the manual). Positions are entered at t=0
and held to expiry — no intraday roll.

Pricing convention: zero drift GBM, sigma=251% annualized, 4 steps/day,
252 trading days/year. dt = 1/(252*4) ≈ 0.000992 years.

The dashboard runs a much larger MC (default 10k paths) for stable fair
values, then resamples 100-sim "trials" to surface the realised PnL
distribution under the server's 100-sim scoring noise.
"""

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, no_update

SIGMA_ANN_DEFAULT = 2.51
S0_DEFAULT = 50.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY
DT = 1.0 / STEPS_PER_YEAR
CONTRACT_SIZE = 3000
SCORING_N_SIMS = 100  # server scoring uses 100 sims per the manual

WEEKS_2_STEPS = 2 * 5 * STEPS_PER_DAY  # 40
WEEKS_3_STEPS = 3 * 5 * STEPS_PER_DAY  # 60

INSTRUMENTS = [
    {"name": "AC", "kind": "underlying", "bid": 49.975, "ask": 50.025, "limit": 200,
     "label": "AC (spot)",
     "params": {"expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_50_P", "kind": "vanilla", "bid": 12.00, "ask": 12.05, "limit": 50,
     "label": "3w Put K=50",
     "params": {"K": 50, "is_call": False, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_50_C", "kind": "vanilla", "bid": 12.00, "ask": 12.05, "limit": 50,
     "label": "3w Call K=50",
     "params": {"K": 50, "is_call": True, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_35_P", "kind": "vanilla", "bid": 4.33, "ask": 4.35, "limit": 50,
     "label": "3w Put K=35",
     "params": {"K": 35, "is_call": False, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_40_P", "kind": "vanilla", "bid": 6.50, "ask": 6.55, "limit": 50,
     "label": "3w Put K=40",
     "params": {"K": 40, "is_call": False, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_45_P", "kind": "vanilla", "bid": 9.05, "ask": 9.10, "limit": 50,
     "label": "3w Put K=45",
     "params": {"K": 45, "is_call": False, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_60_C", "kind": "vanilla", "bid": 8.80, "ask": 8.85, "limit": 50,
     "label": "3w Call K=60",
     "params": {"K": 60, "is_call": True, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_50_P_2", "kind": "vanilla", "bid": 9.70, "ask": 9.75, "limit": 50,
     "label": "2w Put K=50",
     "params": {"K": 50, "is_call": False, "expiry_steps": WEEKS_2_STEPS}},
    {"name": "AC_50_C_2", "kind": "vanilla", "bid": 9.70, "ask": 9.75, "limit": 50,
     "label": "2w Call K=50",
     "params": {"K": 50, "is_call": True, "expiry_steps": WEEKS_2_STEPS}},
    {"name": "AC_50_CO", "kind": "chooser", "bid": 22.20, "ask": 22.30, "limit": 50,
     "label": "Chooser K=50 (decide@2w)",
     "params": {"K": 50, "decision_steps": WEEKS_2_STEPS, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_40_BP", "kind": "binary_put", "bid": 5.00, "ask": 5.10, "limit": 50,
     "label": "Binary Put K=40 pay=10",
     "params": {"K": 40, "payoff": 10, "expiry_steps": WEEKS_3_STEPS}},
    {"name": "AC_45_KO", "kind": "knockout_put", "bid": 0.15, "ask": 0.175, "limit": 500,
     "label": "KO Put K=45 (barrier=35)",
     "params": {"K": 45, "barrier": 35, "expiry_steps": WEEKS_3_STEPS}},
]

CONTAINER_STYLE = {
    "display": "grid",
    "gridTemplateRows": "26vh 32vh 22vh 22vh 32vh",
    "gap": "2px",
    "height": "100%",
}


def _z_for_seed(seed, n_sims, n_steps):
    return np.random.default_rng(int(seed)).standard_normal((int(n_sims), int(n_steps)))


def _log_paths(z, sigma_ann):
    """Cumulative log-returns under GBM zero-drift. S0-free → Δ-bumps reuse this."""
    drift = -0.5 * float(sigma_ann) * float(sigma_ann) * DT
    diffusion = float(sigma_ann) * np.sqrt(DT)
    cum = np.cumsum(drift + diffusion * z, axis=1)
    return np.concatenate([np.zeros((z.shape[0], 1)), cum], axis=1)


def _paths(s0, log_paths):
    return float(s0) * np.exp(log_paths)


def _payoff(inst, paths):
    """Vectorized payoff per simulation."""
    p = inst["params"]
    kind = inst["kind"]
    if kind == "underlying":
        return paths[:, p["expiry_steps"]]
    if kind == "vanilla":
        S_T = paths[:, p["expiry_steps"]]
        if p["is_call"]:
            return np.maximum(S_T - p["K"], 0.0)
        return np.maximum(p["K"] - S_T, 0.0)
    if kind == "chooser":
        S_d = paths[:, p["decision_steps"]]
        S_T = paths[:, p["expiry_steps"]]
        choose_call = S_d > p["K"]
        return np.where(
            choose_call,
            np.maximum(S_T - p["K"], 0.0),
            np.maximum(p["K"] - S_T, 0.0),
        )
    if kind == "binary_put":
        S_T = paths[:, p["expiry_steps"]]
        return np.where(S_T < p["K"], float(p["payoff"]), 0.0)
    if kind == "knockout_put":
        knocked = np.any(paths[:, : p["expiry_steps"] + 1] < p["barrier"], axis=1)
        S_T = paths[:, p["expiry_steps"]]
        return np.where(knocked, 0.0, np.maximum(p["K"] - S_T, 0.0))
    return np.zeros(paths.shape[0])


def _payoffs_from_paths(paths):
    out = np.empty((len(INSTRUMENTS), paths.shape[0]))
    for i, inst in enumerate(INSTRUMENTS):
        out[i] = _payoff(inst, paths)
    return out


def _all_payoffs(s0, sigma_ann, n_sims, seed):
    """Returns (payoffs (n_inst, n_sims), log_paths, paths)."""
    z = _z_for_seed(seed, n_sims, WEEKS_3_STEPS)
    lp = _log_paths(z, sigma_ann)
    paths = _paths(s0, lp)
    return _payoffs_from_paths(paths), lp, paths


def _exec_price(qty, bid, ask):
    if qty > 0:
        return float(ask)
    if qty < 0:
        return float(bid)
    return 0.0


def _signed_edge(fair, bid, ask):
    """Per-share edge: + means buy edge, - means sell edge, 0 means within spread."""
    if fair > ask:
        return float(fair - ask)
    if fair < bid:
        return float(fair - bid)
    return 0.0


def _build_fresh_history_chart(history):
    """Time-series of empirical means across Fresh 100× runs."""
    fig = go.Figure()
    if not history:
        fig.add_annotation(
            text="Click 'Fresh 100×' to plot empirical-mean history.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 12, "color": "#666"},
        )
        fig.update_layout(
            title="Empirical mean across Fresh 100× runs",
            xaxis_title="Run #",
            yaxis_title="Mean PnL ($)",
            template="plotly_white",
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
        )
        return fig

    x = list(range(1, len(history) + 1))
    means = [float(h.get("mean", 0.0)) for h in history]
    p5s = [float(h.get("p5", 0.0)) for h in history]
    p95s = [float(h.get("p95", 0.0)) for h in history]
    sds = [float(h.get("sd", 0.0)) for h in history]

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=p95s + p5s[::-1],
        fill="toself",
        fillcolor="rgba(31,119,180,0.15)",
        line={"color": "rgba(0,0,0,0)"},
        hoverinfo="skip",
        showlegend=False,
        name="p5-p95",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=means,
        mode="lines+markers",
        line={"color": "#1f77b4", "width": 2},
        marker={"size": 7},
        name="Mean",
        customdata=list(zip(sds, p5s, p95s)),
        hovertemplate=("Run %{x}<br>Mean=$%{y:,.0f}<br>"
                       "SD=$%{customdata[0]:,.0f}<br>"
                       "p5=$%{customdata[1]:,.0f}<br>"
                       "p95=$%{customdata[2]:,.0f}<extra></extra>"),
    ))
    fig.add_hline(y=0, line={"color": "#333", "width": 1, "dash": "dot"})

    overall = float(np.mean(means))
    fig.add_hline(y=overall, line={"color": "#d62728", "width": 1, "dash": "dash"},
                  annotation_text=f"avg of means=${overall:,.0f}",
                  annotation_position="bottom right",
                  annotation_font={"color": "#d62728", "size": 10})

    fig.update_layout(
        title=f"Empirical mean across Fresh 100× runs (n={len(history)})",
        xaxis_title="Run #",
        yaxis_title="Mean PnL ($)",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        showlegend=False,
    )
    return fig


def _solve_qp(mu, Sigma, limits, lam):
    """Box-constrained QP: max μᵀq - λ qᵀΣq, -limits ≤ q ≤ limits.

    L-BFGS-B from max-edge corner so λ→0 ≡ sign(μ)×limit. Tiny positions
    (|q|<0.5) zeroed; spread otherwise dominates edge.
    """
    from scipy.optimize import minimize, Bounds
    lim_arr = np.asarray(limits, dtype=float)
    lam = max(float(lam), 0.0)
    x0 = np.where(mu > 0, lim_arr, np.where(mu < 0, -lim_arr, 0.0))
    if lam < 1e-15:
        q = x0
    else:
        bounds = Bounds(-lim_arr, lim_arr)

        def neg_obj(q):
            return -(mu @ q - lam * q @ Sigma @ q)

        def neg_grad(q):
            return -(mu - 2.0 * lam * (Sigma @ q))

        result = minimize(neg_obj, x0=x0, jac=neg_grad, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
        q = result.x
    q = np.where(np.abs(q) < 0.5, 0.0, q)
    return np.round(q).astype(int)


def _mu_sigma_from_payoffs(payoffs, bids, asks):
    bids_arr = np.asarray(bids, dtype=float)
    asks_arr = np.asarray(asks, dtype=float)
    fair = payoffs.mean(axis=1)
    mu = (fair - 0.5 * (bids_arr + asks_arr)) * CONTRACT_SIZE
    Sigma = np.cov(payoffs) * (CONTRACT_SIZE ** 2) / SCORING_N_SIMS
    return mu, Sigma


def _portfolio_optimize(payoffs, bids, asks, limits, lambda_risk):
    """Mean-variance portfolio. Maximizes μᵀq − λ qᵀΣq with box constraints.

    μ = (fair − mid) × 3000           per-unit expected $-edge
    Σ = Cov(payoffs) / 100 × 3000²    covariance of 100-sim marks
    """
    mu, Sigma = _mu_sigma_from_payoffs(payoffs, bids, asks)
    return _solve_qp(mu, Sigma, limits, lambda_risk).tolist()


def _run_mvo_analysis(s0, sigma, n_sims_calib, seed_calib,
                      lambda_min, lambda_max, lambda_step,
                      n_batches, scoring_seed):
    """Sweep λ, optimize at each, score with fresh batches (common RNs across λ).

    Calibration MC drives optimization (μ, Σ). Scoring pool is N independent
    100-sim batches drawn once and shared across all λ values — the "common
    random numbers" trick eliminates MC noise from λ-vs-λ comparisons, so
    the SD across batches reflects strategy stability rather than path luck.
    """
    payoffs_calib, _, _ = _all_payoffs(s0, sigma, n_sims_calib, seed_calib)
    bids = [i["bid"] for i in INSTRUMENTS]
    asks = [i["ask"] for i in INSTRUMENTS]
    limits = [i["limit"] for i in INSTRUMENTS]
    bids_arr = np.array(bids, dtype=float)
    asks_arr = np.array(asks, dtype=float)
    mu, Sigma = _mu_sigma_from_payoffs(payoffs_calib, bids_arr, asks_arr)

    n_inst = len(INSTRUMENTS)
    n_batches = max(int(n_batches), 2)
    n_scoring = n_batches * SCORING_N_SIMS
    z = _z_for_seed(scoring_seed, n_scoring, WEEKS_3_STEPS)
    log_paths_score = _log_paths(z, sigma)
    paths_score = _paths(s0, log_paths_score)
    payoffs_score = _payoffs_from_paths(paths_score)
    payoffs_batched = payoffs_score.reshape(n_inst, n_batches, SCORING_N_SIMS)
    batch_marks = payoffs_batched.mean(axis=2)  # (n_inst, n_batches)

    lambdas = np.round(np.arange(lambda_min, lambda_max + 1e-9, lambda_step), 4)
    n_lam = len(lambdas)
    avg_pnl = np.zeros(n_lam)
    std_pnl = np.zeros(n_lam)
    qty_count = np.zeros(n_lam, dtype=int)
    qty_log = []

    for i, log_lam in enumerate(lambdas):
        lam = 10.0 ** float(log_lam)
        q = _solve_qp(mu, Sigma, limits, lam).astype(float)
        exec_prices = np.where(q > 0, asks_arr, np.where(q < 0, bids_arr, 0.0))
        per_inst_per_batch = (batch_marks.T - exec_prices) * q * CONTRACT_SIZE
        per_batch_pnl = per_inst_per_batch.sum(axis=1)
        avg_pnl[i] = float(per_batch_pnl.mean())
        std_pnl[i] = float(per_batch_pnl.std(ddof=1))
        qty_count[i] = int(np.sum(np.abs(q) > 0))
        qty_log.append(q.astype(int).tolist())
    return lambdas, avg_pnl, std_pnl, qty_count, qty_log


def _build_mvo_analysis_fig(lambdas, avg_pnl, std_pnl, qty_count, n_batches):
    fig = go.Figure()
    if len(lambdas) == 0:
        fig.add_annotation(
            text="Click 'Run full λ analysis' to sweep risk aversion.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 13, "color": "#666"},
        )
        fig.update_layout(
            title="Full λ sweep (MVO)",
            xaxis_title="log10(λ)",
            yaxis_title="PnL ($)",
            template="plotly_white",
            margin={"l": 60, "r": 60, "t": 40, "b": 60},
        )
        return fig

    avg_pnl = np.asarray(avg_pnl, dtype=float)
    std_pnl = np.asarray(std_pnl, dtype=float)
    qty_count = np.asarray(qty_count, dtype=int)
    upper = avg_pnl + std_pnl
    lower = avg_pnl - std_pnl

    fig.add_trace(go.Scatter(
        x=list(lambdas) + list(lambdas[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(31,119,180,0.18)",
        line={"color": "rgba(0,0,0,0)"},
        name="±1 SD across batches",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=lambdas, y=avg_pnl,
        mode="lines+markers",
        line={"color": "#1f77b4", "width": 2.2},
        marker={"size": 4},
        name="Avg PnL across batches",
        customdata=np.column_stack([std_pnl, qty_count]),
        hovertemplate=("log10(λ)=%{x:.1f}<br>"
                       "Avg PnL=$%{y:,.0f}<br>"
                       "SD=$%{customdata[0]:,.0f}<br>"
                       "n positions=%{customdata[1]:.0f}"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=lambdas, y=std_pnl,
        mode="lines",
        line={"color": "#d62728", "width": 1.5, "dash": "dot"},
        yaxis="y2",
        name="SD across batches (right axis)",
        hovertemplate="log10(λ)=%{x:.1f}<br>SD=$%{y:,.0f}<extra></extra>",
    ))

    best_mean_i = int(np.argmax(avg_pnl))
    sharpe = np.where(std_pnl > 1e-6, avg_pnl / std_pnl, 0.0)
    best_sharpe_i = int(np.argmax(sharpe))

    fig.add_vline(x=float(lambdas[best_mean_i]),
                  line={"color": "#2ca02c", "dash": "dot"},
                  annotation_text=f"max avg @ {lambdas[best_mean_i]:.1f}",
                  annotation_position="top left",
                  annotation_font={"color": "#2ca02c", "size": 10})
    if best_sharpe_i != best_mean_i:
        fig.add_vline(x=float(lambdas[best_sharpe_i]),
                      line={"color": "#9467bd", "dash": "dash"},
                      annotation_text=f"max Sharpe @ {lambdas[best_sharpe_i]:.1f}",
                      annotation_position="bottom right",
                      annotation_font={"color": "#9467bd", "size": 10})
    fig.add_hline(y=0, line={"color": "#333", "width": 1})

    fig.update_layout(
        title=(f"λ sweep: avg PnL ± SD across {n_batches} fresh 100-sim batches "
               "(common random numbers)"),
        xaxis_title="log10(λ) — left = pure return, right = max hedge",
        yaxis_title="Avg PnL ($)",
        yaxis2={"title": "SD across batches ($)",
                "overlaying": "y", "side": "right",
                "showgrid": False, "color": "#d62728"},
        template="plotly_white",
        margin={"l": 60, "r": 60, "t": 40, "b": 60},
        legend={"orientation": "h", "y": -0.22},
    )
    return fig


def _format_score_display(history, latest):
    """Render the realized-score sidebar block from accumulated history."""
    n = len(history)
    if n == 0:
        return html.Div(
            "(no scoring runs yet)",
            style={"color": "#888", "fontStyle": "italic"},
        )
    arr = np.array(history, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    p5 = float(np.percentile(arr, 5)) if n >= 5 else float(arr.min())
    p95 = float(np.percentile(arr, 95)) if n >= 5 else float(arr.max())
    win = float((arr > 0).mean())

    children = []
    if latest is not None:
        color = "#2ca02c" if latest > 0 else "#d62728" if latest < 0 else "#333"
        children.append(html.Div([
            html.B("Last run: "),
            html.Span(f"${latest:,.0f}",
                      style={"color": color, "fontWeight": "bold"}),
        ]))
    children.extend([
        html.Div([html.B(f"Runs: "), f"{n}"]),
        html.Div([html.B("Empirical mean: "), f"${mean:,.0f}"]),
        html.Div([html.B("Empirical SD: "), f"${sd:,.0f}"]),
        html.Div([html.B("p5 / p95: "), f"${p5:,.0f} / ${p95:,.0f}"]),
        html.Div([html.B("Win rate: "), f"{win:.1%}"]),
    ])
    return html.Div(children)


def _position_table_layout():
    rows = []
    cell = {"padding": "3px 6px", "fontSize": "11px"}
    cell_r = {**cell, "textAlign": "right"}
    for inst in INSTRUMENTS:
        rows.append(html.Tr(style={"borderBottom": "1px solid #eee"}, children=[
            html.Td(inst["label"], style=cell),
            html.Td(f"{inst['bid']:.3f}", style=cell_r),
            html.Td(f"{inst['ask']:.3f}", style=cell_r),
            html.Td(id=f"r4-fair-{inst['name']}", style=cell_r),
            html.Td(id=f"r4-edge-{inst['name']}", style=cell_r),
            html.Td(dcc.Input(
                id=f"r4-qty-{inst['name']}",
                type="number",
                min=-inst["limit"], max=inst["limit"], step=1, value=0,
                debounce=True,
                style={"width": "64px", "fontSize": "11px", "textAlign": "right"},
            ), style={**cell, "textAlign": "center"}),
            html.Td(f"±{inst['limit']}", style={**cell_r, "color": "#888"}),
            html.Td(id=f"r4-pnl-{inst['name']}", style=cell_r),
        ]))
    header = html.Thead(html.Tr(
        style={"borderBottom": "2px solid #333", "fontSize": "11px",
               "background": "#f7f7f7", "position": "sticky", "top": 0},
        children=[
            html.Th("Instrument", style={"textAlign": "left", "padding": "4px 8px"}),
            html.Th("Bid", style={"textAlign": "right", "padding": "4px 8px"}),
            html.Th("Ask", style={"textAlign": "right", "padding": "4px 8px"}),
            html.Th("MC fair", style={"textAlign": "right", "padding": "4px 8px"}),
            html.Th("Edge/share", style={"textAlign": "right", "padding": "4px 8px"}),
            html.Th("Qty (±limit)", style={"textAlign": "center", "padding": "4px 8px"}),
            html.Th("Limit", style={"textAlign": "right", "padding": "4px 8px"}),
            html.Th("E[PnL] $", style={"textAlign": "right", "padding": "4px 8px"}),
        ],
    ))
    footer = html.Tfoot(html.Tr([
        html.Td("PORTFOLIO E[PnL]", colSpan=7,
                style={"padding": "6px 8px", "fontWeight": "bold",
                       "borderTop": "2px solid #333", "fontSize": "12px"}),
        html.Td(id="r4-total-pnl", style={**cell_r, "fontWeight": "bold",
                                          "borderTop": "2px solid #333",
                                          "fontSize": "12px"}),
    ]))
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[header, html.Tbody(rows), footer],
    )


def controls_layout():
    return html.Div(id="manual-round4-controls", style={"display": "none"}, children=[
        html.B("MC parameters"),
        html.Div(style={"marginTop": "6px"}, children=[
            html.Label("Initial spot S0", style={"fontSize": "11px"}),
            dcc.Input(id="r4-s0", type="number", value=S0_DEFAULT, step=0.01,
                      debounce=True, style={"width": "100%", "fontSize": "11px"}),
        ]),
        html.Div(style={"marginTop": "6px"}, children=[
            html.Label(f"Annualized vol σ (manual = {SIGMA_ANN_DEFAULT})",
                       style={"fontSize": "11px"}),
            dcc.Input(id="r4-sigma", type="number", value=SIGMA_ANN_DEFAULT, step=0.01,
                      debounce=True, style={"width": "100%", "fontSize": "11px"}),
        ]),
        html.Label("MC sims (fair value precision)",
                   style={"fontSize": "11px", "marginTop": "10px"}),
        dcc.Slider(id="r4-n-sims", min=2000, max=50000, step=1000, value=10000,
                   marks={2000: "2k", 10000: "10k", 25000: "25k", 50000: "50k"},
                   tooltip={"placement": "bottom"}),
        html.Label("Seed", style={"fontSize": "11px"}),
        dcc.Slider(id="r4-seed", min=0, max=99, step=1, value=0,
                   marks={0: "0", 50: "50", 99: "99"},
                   tooltip={"placement": "bottom"}),
        html.Label("Score-noise trials (each = 100 sims)",
                   style={"fontSize": "11px"}),
        dcc.Slider(id="r4-n-trials", min=200, max=5000, step=100, value=1000,
                   marks={200: "200", 1000: "1k", 5000: "5k"},
                   tooltip={"placement": "bottom"}),

        html.Hr(),
        html.B("Quick presets"),
        html.Div(style={"display": "flex", "gap": "4px", "flexWrap": "wrap",
                        "marginTop": "6px"}, children=[
            html.Button("Zero all", id="r4-preset-zero", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Max-edge", id="r4-preset-edge", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Δ-hedge AC", id="r4-preset-hedge", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
        ]),
        html.Div(
            "Max-edge: take ±limit per instrument toward MC edge. "
            "Δ-hedge AC: set AC qty so total portfolio Δ = 0 against current options.",
            style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
        ),

        html.Hr(),
        html.B("Mean-variance optimizer"),
        html.Div(
            "Solves max(E[PnL] − λ × Var[PnL]) under ±limit. Higher λ hedges "
            "against the same covariance that drives seed-to-seed variation, "
            "giving a strategy that's stable across both scoring noise and "
            "different MC seeds.",
            style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
        ),
        html.Label("Risk aversion λ (log10 scale)",
                   style={"fontSize": "11px", "marginTop": "8px"}),
        dcc.Slider(id="r4-risk-lambda", min=-12, max=-3, step=0.01, value=-7,
                   marks={-12: "0", -10: "-10", -8: "-8", -6: "-6", -4: "-4",
                          -3: "high"},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(style={"display": "flex", "gap": "4px", "alignItems": "center",
                        "marginTop": "4px"}, children=[
            html.Label("λ exact:", style={"fontSize": "11px"}),
            dcc.Input(id="r4-risk-lambda-input", type="number",
                      min=-12, max=-3, step=0.001, value=-7, debounce=True,
                      style={"width": "70px", "fontSize": "11px"}),
        ]),
        html.Button("Optimize portfolio", id="r4-preset-mvo", n_clicks=0,
                    style={"fontSize": "11px", "padding": "4px 8px",
                           "marginTop": "6px", "width": "100%"}),

        html.Div(style={"marginTop": "10px", "padding": "6px",
                        "background": "#f9f9f9", "borderRadius": "4px"},
                 children=[
            html.B("Full λ sweep", style={"fontSize": "11px"}),
            html.Div(
                "For each λ in [-12,-3] step 0.1: optimizes portfolio, "
                "scores N fresh 100-sim batches (common random numbers across λ). "
                "Plots avg PnL ± SD across those N batches.",
                style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
            ),
            html.Label("Batches per λ", style={"fontSize": "11px", "marginTop": "6px"}),
            dcc.Slider(id="r4-mvo-analysis-batches", min=5, max=30, step=1, value=10,
                       marks={5: "5", 10: "10", 20: "20", 30: "30"},
                       tooltip={"placement": "bottom"}),
            html.Button("Run full λ analysis", id="r4-mvo-analysis-run", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px",
                               "marginTop": "6px", "width": "100%"}),
            html.Div(id="r4-mvo-analysis-status", style={
                "fontSize": "10px", "color": "#666", "marginTop": "4px",
            }),
        ]),
        dcc.Store(id="r4-mvo-analysis-data", data=None),

        html.Hr(),
        html.B("Realized score (fresh 100-sim)"),
        html.Div(
            "Each click runs 100 fresh GBM paths (independent of MC pool) and "
            "averages payoffs — exactly how the server scores. Click repeatedly "
            "to build the empirical distribution.",
            style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
        ),
        html.Div(style={"display": "flex", "gap": "4px", "flexWrap": "wrap",
                        "marginTop": "6px"}, children=[
            html.Button("Score 1×", id="r4-score-once", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Score 100×", id="r4-score-batch", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Fresh 100×", id="r4-score-fresh", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Reset", id="r4-score-reset", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
            html.Button("Reset graph", id="r4-fresh-history-reset", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 8px"}),
        ]),
        html.Div(id="r4-score-display", style={
            "marginTop": "6px", "fontSize": "11px", "lineHeight": "1.5",
        }),
        dcc.Store(id="r4-score-history", data=[]),
        dcc.Store(id="r4-fresh-history", data=[]),

        html.Hr(),
        html.B("Portfolio"),
        html.Div(id="r4-portfolio-stats", style={
            "fontSize": "11px", "lineHeight": "1.6", "marginTop": "6px",
        }),
    ])


def charts_layout():
    return html.Div(
        id="manual-round4-container",
        style={**CONTAINER_STYLE, "display": "none"},
        children=[
            html.Div(dcc.Graph(id="r4-edge-chart", style={"height": "100%"}),
                     style={"border": "1px solid #ddd", "borderRadius": "4px"}),
            html.Div(_position_table_layout(),
                     style={"border": "1px solid #ddd", "borderRadius": "4px",
                            "padding": "8px", "overflowY": "auto"}),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                            "gap": "2px", "height": "100%"}, children=[
                html.Div(dcc.Graph(id="r4-pnl-hist", style={"height": "100%"}),
                         style={"border": "1px solid #ddd", "borderRadius": "4px"}),
                html.Div(dcc.Graph(id="r4-payoff-diagram", style={"height": "100%"}),
                         style={"border": "1px solid #ddd", "borderRadius": "4px"}),
            ]),
            html.Div(dcc.Graph(id="r4-fresh-history-chart", style={"height": "100%"}),
                     style={"border": "1px solid #ddd", "borderRadius": "4px"}),
            html.Div(dcc.Loading(
                dcc.Graph(id="r4-mvo-analysis-chart", style={"height": "100%"}),
                type="default",
            ), style={"border": "1px solid #ddd", "borderRadius": "4px"}),
        ],
    )


def _build_edge_fig(fair, bids, asks, edge_signed):
    names = [i["name"] for i in INSTRUMENTS]
    labels = [i["label"] for i in INSTRUMENTS]
    limits = np.array([i["limit"] for i in INSTRUMENTS], dtype=float)
    max_edge_signed_usd = edge_signed * limits * CONTRACT_SIZE
    colors = [
        "#2ca02c" if e > 0 else "#d62728" if e < 0 else "#bbb"
        for e in edge_signed
    ]
    hover = [
        (f"<b>{lab}</b><br>Bid: {b:.3f}<br>Ask: {a:.3f}<br>MC fair: {f:.3f}<br>"
         f"Edge: {e:+.4f}/share<br>Limit: ±{int(lim)}<br>$ at full size: {full:+,.0f}")
        for lab, b, a, f, e, lim, full in zip(
            labels, bids, asks, fair, edge_signed, limits, max_edge_signed_usd
        )
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=max_edge_signed_usd,
        marker_color=colors,
        text=[f"{e:+,.0f}" if abs(e) >= 1 else "" for e in max_edge_signed_usd],
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
        name="$ edge at full size",
    ))
    fig.add_hline(y=0, line={"color": "#333", "width": 1})
    fig.update_layout(
        title="Tradeable edge per instrument (green=buy, red=sell, ×limit×3000)",
        yaxis_title="$ edge at max position",
        xaxis={"tickangle": -25},
        margin={"l": 60, "r": 20, "t": 40, "b": 60},
        template="plotly_white",
        showlegend=False,
    )
    return fig


def _build_payoff_diagram(per_path_pnl, S_T, s0):
    """PnL conditional on terminal underlying — the hedge view."""
    fig = go.Figure()
    if per_path_pnl.size == 0 or np.allclose(per_path_pnl, 0.0):
        fig.add_annotation(
            text="Set positions to see payoff diagram.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 13, "color": "#666"},
        )
        fig.update_layout(
            title="Payoff diagram (PnL vs S at 3w expiry)",
            template="plotly_white",
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
        )
        return fig

    s_lo = float(np.percentile(S_T, 1))
    s_hi = float(np.percentile(S_T, 99))
    n_bins = 40
    edges = np.linspace(s_lo, s_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(S_T, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        if m.sum() >= 5:
            means[k] = float(per_path_pnl[m].mean())

    sub = max(1, len(S_T) // 1500)
    fig.add_trace(go.Scatter(
        x=S_T[::sub], y=per_path_pnl[::sub],
        mode="markers",
        marker={"size": 2, "color": "#aaa", "opacity": 0.35},
        name="per-path PnL",
        hovertemplate="S_T=%{x:.2f}<br>path PnL=%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=means,
        mode="lines+markers",
        line={"color": "#d62728", "width": 2.5},
        marker={"size": 4},
        name="E[PnL | S_T]",
        hovertemplate="S_T=%{x:.2f}<br>E[PnL]=%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": "#333", "width": 1})
    fig.add_vline(x=s0, line={"color": "#888", "dash": "dot"},
                  annotation_text=f"S0={s0:.1f}", annotation_position="top")
    fig.update_layout(
        title="Payoff diagram: portfolio PnL vs terminal AC at 3w (flat = hedged)",
        xaxis_title="S_T (terminal AC)",
        yaxis_title="PnL ($) per path",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def _build_pnl_hist(trial_pnl, mean, p5, p95, n_trials):
    fig = go.Figure()
    if trial_pnl.size == 0 or np.allclose(trial_pnl, 0.0):
        fig.add_annotation(
            text="Set positions to see the PnL distribution.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": 13, "color": "#666"},
        )
    else:
        fig.add_trace(go.Histogram(
            x=trial_pnl, nbinsx=60,
            marker={"color": "#1f77b4", "line": {"color": "white", "width": 0.5}},
        ))
        fig.add_vline(x=0, line={"color": "#333", "width": 1})
        fig.add_vline(x=mean, line={"color": "red", "dash": "dash"},
                      annotation_text=f"mean=${mean:,.0f}",
                      annotation_position="top left")
        fig.add_vline(x=p5, line={"color": "orange", "dash": "dot"},
                      annotation_text=f"p5=${p5:,.0f}",
                      annotation_position="bottom left")
        fig.add_vline(x=p95, line={"color": "orange", "dash": "dot"},
                      annotation_text=f"p95=${p95:,.0f}",
                      annotation_position="bottom right")
    fig.update_layout(
        title=f"Portfolio PnL distribution under 100-sim scoring noise (n_trials={n_trials})",
        xaxis_title="Trial PnL ($)",
        yaxis_title="count",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
    )
    return fig


def register_callbacks(app):
    fair_outputs = [Output(f"r4-fair-{i['name']}", "children") for i in INSTRUMENTS]
    edge_outputs = [Output(f"r4-edge-{i['name']}", "children") for i in INSTRUMENTS]
    pnl_outputs = [Output(f"r4-pnl-{i['name']}", "children") for i in INSTRUMENTS]
    qty_inputs = [Input(f"r4-qty-{i['name']}", "value") for i in INSTRUMENTS]
    qty_outputs = [Output(f"r4-qty-{i['name']}", "value") for i in INSTRUMENTS]

    @app.callback(
        [Output("r4-edge-chart", "figure"),
         Output("r4-pnl-hist", "figure"),
         Output("r4-payoff-diagram", "figure"),
         Output("r4-total-pnl", "children"),
         Output("r4-portfolio-stats", "children"),
         *fair_outputs, *edge_outputs, *pnl_outputs],
        [Input("r4-s0", "value"),
         Input("r4-sigma", "value"),
         Input("r4-n-sims", "value"),
         Input("r4-seed", "value"),
         Input("r4-n-trials", "value"),
         *qty_inputs],
    )
    def update_all(s0, sigma, n_sims, seed, n_trials, *qtys):
        s0 = float(s0) if s0 not in (None, "") else S0_DEFAULT
        sigma = float(sigma) if sigma not in (None, "") and float(sigma) > 0 else SIGMA_ANN_DEFAULT
        n_sims = max(int(n_sims or 10000), 1000)
        seed = int(seed or 0)
        n_trials = max(int(n_trials or 1000), 100)
        qtys = [int(q) if q not in (None, "") else 0 for q in qtys]

        payoffs, log_paths_arr, paths = _all_payoffs(s0, sigma, n_sims, seed)
        fair = payoffs.mean(axis=1)
        bids = np.array([i["bid"] for i in INSTRUMENTS])
        asks = np.array([i["ask"] for i in INSTRUMENTS])

        edge_signed = np.array([
            _signed_edge(f, b, a) for f, b, a in zip(fair, bids, asks)
        ])

        qty_arr = np.array(qtys, dtype=float)
        exec_prices = np.array([_exec_price(q, b, a) for q, b, a in zip(qtys, bids, asks)])
        per_inst_expected_pnl = qty_arr * (fair - exec_prices) * CONTRACT_SIZE
        total_expected_pnl = float(per_inst_expected_pnl.sum())

        rng_resample = np.random.default_rng(int(seed) + 7919)
        n_sims_actual = payoffs.shape[1]
        all_idx = rng_resample.integers(0, n_sims_actual, size=(n_trials, SCORING_N_SIMS))
        trial_marks = payoffs[:, all_idx].mean(axis=2)  # (n_inst, n_trials)
        per_inst_trial_pnl = (trial_marks.T - exec_prices) * qty_arr * CONTRACT_SIZE
        trial_pnl = per_inst_trial_pnl.sum(axis=1)

        # Per-path PnL conditional on path realization (the "what if S_T ends here" view)
        S_T_3w = paths[:, WEEKS_3_STEPS]
        per_path_pnl = ((payoffs.T - exec_prices) * qty_arr * CONTRACT_SIZE).sum(axis=1)

        # Δ via S0 bump — log_paths is S0-free, so bumps just remultiply (cheap).
        eps_s = 0.5
        payoffs_up = _payoffs_from_paths(_paths(s0 + eps_s, log_paths_arr))
        payoffs_dn = _payoffs_from_paths(_paths(s0 - eps_s, log_paths_arr))
        delta_per_inst = (payoffs_up.mean(axis=1) - payoffs_dn.mean(axis=1)) / (2 * eps_s)
        portfolio_delta = float((qty_arr * delta_per_inst).sum() * CONTRACT_SIZE)

        # SE of E[PnL]: how much the fair-value estimate would drift on a fresh MC pool.
        if np.any(qty_arr != 0):
            cov_payoff = np.cov(payoffs)
            var_e_pnl = float(qty_arr @ cov_payoff @ qty_arr) * (CONTRACT_SIZE ** 2) / n_sims
            se_e_pnl = float(np.sqrt(max(var_e_pnl, 0.0)))
        else:
            se_e_pnl = 0.0

        # Δ-neutral AC qty (what AC must equal so non-AC option Δ cancels).
        non_ac_delta_sum = float(sum(
            qty_arr[i] * delta_per_inst[i] for i in range(1, len(INSTRUMENTS))
        ))
        ac_neutral_qty = -int(round(non_ac_delta_sum))
        ac_neutral_qty = max(min(ac_neutral_qty, INSTRUMENTS[0]["limit"]),
                             -INSTRUMENTS[0]["limit"])

        if np.any(qty_arr != 0):
            pnl_mean = float(np.mean(trial_pnl))
            pnl_std = float(np.std(trial_pnl, ddof=1))
            pnl_p5 = float(np.percentile(trial_pnl, 5))
            pnl_p50 = float(np.percentile(trial_pnl, 50))
            pnl_p95 = float(np.percentile(trial_pnl, 95))
            win_rate = float(np.mean(trial_pnl > 0))
        else:
            pnl_mean = pnl_std = pnl_p5 = pnl_p50 = pnl_p95 = 0.0
            win_rate = 0.0

        edge_fig = _build_edge_fig(fair, bids, asks, edge_signed)
        pnl_fig = _build_pnl_hist(trial_pnl, pnl_mean, pnl_p5, pnl_p95, n_trials)
        payoff_fig = _build_payoff_diagram(per_path_pnl, S_T_3w, s0)

        fair_cells = [f"{f:.3f}" for f in fair]
        edge_cells = []
        for f, b, a in zip(fair, bids, asks):
            if f > a:
                edge_cells.append(html.Span(
                    f"+{f - a:.3f} BUY",
                    style={"color": "#2ca02c", "fontWeight": "bold"},
                ))
            elif f < b:
                edge_cells.append(html.Span(
                    f"+{b - f:.3f} SELL",
                    style={"color": "#d62728", "fontWeight": "bold"},
                ))
            else:
                edge_cells.append(html.Span("flat", style={"color": "#888"}))

        pnl_cells = []
        for p in per_inst_expected_pnl:
            if abs(p) < 1:
                pnl_cells.append(html.Span("—", style={"color": "#bbb"}))
            else:
                color = "#2ca02c" if p > 0 else "#d62728"
                pnl_cells.append(html.Span(f"{p:,.0f}", style={"color": color}))

        if abs(total_expected_pnl) < 1:
            total_color = "#333"
        else:
            total_color = "#2ca02c" if total_expected_pnl > 0 else "#d62728"
        total_cell = html.Span(f"{total_expected_pnl:,.0f}", style={"color": total_color})

        sharpe = pnl_mean / pnl_std if pnl_std > 1e-9 else 0.0
        delta_in_units_S = portfolio_delta / CONTRACT_SIZE
        ac_change = ac_neutral_qty - qtys[0]
        portfolio_stats = html.Div([
            html.Div([html.B("E[PnL] (true): "), f"${total_expected_pnl:,.0f}"]),
            html.Div([html.B("Stability SE: "), f"±${se_e_pnl:,.0f} ",
                      html.Span("(seed-to-seed E[PnL] noise)",
                                style={"color": "#888"})], style={"color": "#555"}),
            html.Hr(style={"margin": "8px 0"}),
            html.B("Score-noise distribution"),
            html.Div([html.B("Trial mean: "), f"${pnl_mean:,.0f}"]),
            html.Div([html.B("Trial SD: "), f"${pnl_std:,.0f}"]),
            html.Div([html.B("p5 / p50 / p95: "),
                      f"${pnl_p5:,.0f} / ${pnl_p50:,.0f} / ${pnl_p95:,.0f}"]),
            html.Div([html.B("P(profit): "), f"{win_rate:.1%}"]),
            html.Div([html.B("Sharpe-ish: "), f"{sharpe:.2f}"],
                     style={"color": "#555"}),
            html.Hr(style={"margin": "8px 0"}),
            html.B("Hedging"),
            html.Div([html.B("Portfolio Δ: "),
                      f"${portfolio_delta:+,.0f}/unit-S ",
                      html.Span(f"(≈ {delta_in_units_S:+.1f} S-units)",
                                style={"color": "#888"})]),
            html.Div([html.B("Δ-neutral AC qty: "),
                      f"{ac_neutral_qty:+d} ",
                      html.Span(f"(currently {qtys[0]:+d}, change {ac_change:+d})",
                                style={"color": "#888"})]),
            html.Div(
                "Use the payoff diagram to see your S-exposure curve. Flatten with the "
                "Δ-hedge AC button or set qty manually. Stability SE drops when your "
                "options + AC offsets net out across MC noise.",
                style={"fontSize": "10px", "color": "#666", "marginTop": "6px"},
            ),
        ])

        return [edge_fig, pnl_fig, payoff_fig, total_cell, portfolio_stats] \
            + fair_cells + edge_cells + pnl_cells

    qty_states = [State(f"r4-qty-{i['name']}", "value") for i in INSTRUMENTS]

    @app.callback(
        qty_outputs,
        [Input("r4-preset-zero", "n_clicks"),
         Input("r4-preset-edge", "n_clicks"),
         Input("r4-preset-hedge", "n_clicks"),
         Input("r4-preset-mvo", "n_clicks")],
        [State("r4-s0", "value"),
         State("r4-sigma", "value"),
         State("r4-n-sims", "value"),
         State("r4-seed", "value"),
         State("r4-risk-lambda", "value"),
         *qty_states],
        prevent_initial_call=True,
    )
    def apply_preset(_zero_clicks, _edge_clicks, _hedge_clicks, _mvo_clicks,
                     s0, sigma, n_sims, seed, risk_log_lambda,
                     *qty_state_vals):
        ctx = callback_context
        if not ctx.triggered:
            return [no_update] * len(INSTRUMENTS)
        button = ctx.triggered[0]["prop_id"].split(".")[0]
        if button == "r4-preset-zero":
            return [0] * len(INSTRUMENTS)

        s0 = float(s0) if s0 not in (None, "") else S0_DEFAULT
        sigma = (float(sigma) if sigma not in (None, "") and float(sigma) > 0
                 else SIGMA_ANN_DEFAULT)
        n_sims = max(int(n_sims or 10000), 1000)
        seed = int(seed or 0)

        if button == "r4-preset-edge":
            payoffs, _, _ = _all_payoffs(s0, sigma, n_sims, seed)
            fair = payoffs.mean(axis=1)
            qtys = []
            for i, inst in enumerate(INSTRUMENTS):
                if fair[i] > inst["ask"]:
                    qtys.append(int(inst["limit"]))
                elif fair[i] < inst["bid"]:
                    qtys.append(-int(inst["limit"]))
                else:
                    qtys.append(0)
            return qtys

        if button == "r4-preset-hedge":
            current_qtys = [int(q) if q not in (None, "") else 0 for q in qty_state_vals]
            _, log_paths_arr, _ = _all_payoffs(s0, sigma, n_sims, seed)
            eps_s = 0.5
            payoffs_up = _payoffs_from_paths(_paths(s0 + eps_s, log_paths_arr))
            payoffs_dn = _payoffs_from_paths(_paths(s0 - eps_s, log_paths_arr))
            delta_per_inst = (payoffs_up.mean(axis=1) - payoffs_dn.mean(axis=1)) / (2 * eps_s)
            non_ac_delta_sum = float(sum(
                current_qtys[i] * delta_per_inst[i] for i in range(1, len(INSTRUMENTS))
            ))
            ac_neutral = -int(round(non_ac_delta_sum))
            ac_neutral = max(min(ac_neutral, INSTRUMENTS[0]["limit"]),
                             -INSTRUMENTS[0]["limit"])
            return [ac_neutral] + [no_update] * (len(INSTRUMENTS) - 1)

        if button == "r4-preset-mvo":
            payoffs, _, _ = _all_payoffs(s0, sigma, n_sims, seed)
            bids = [i["bid"] for i in INSTRUMENTS]
            asks = [i["ask"] for i in INSTRUMENTS]
            limits = [i["limit"] for i in INSTRUMENTS]
            lam = 10.0 ** float(risk_log_lambda if risk_log_lambda is not None else -7)
            return _portfolio_optimize(payoffs, bids, asks, limits, lam)

        return [no_update] * len(INSTRUMENTS)

    @app.callback(
        [Output("r4-score-display", "children"),
         Output("r4-score-history", "data"),
         Output("r4-fresh-history", "data")],
        [Input("r4-score-once", "n_clicks"),
         Input("r4-score-batch", "n_clicks"),
         Input("r4-score-fresh", "n_clicks"),
         Input("r4-score-reset", "n_clicks"),
         Input("r4-fresh-history-reset", "n_clicks")],
        [State("r4-s0", "value"),
         State("r4-sigma", "value"),
         State("r4-score-history", "data"),
         State("r4-fresh-history", "data"),
         *qty_states],
        prevent_initial_call=True,
    )
    def run_realized_score(_once_clicks, _batch_clicks, _fresh_clicks,
                           _reset_clicks, _fresh_reset_clicks,
                           s0, sigma, history, fresh_hist, *qty_state_vals):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        button = ctx.triggered[0]["prop_id"].split(".")[0]

        if button == "r4-fresh-history-reset":
            return no_update, no_update, []

        if button == "r4-score-reset":
            return _format_score_display([], None), [], no_update

        s0 = float(s0) if s0 not in (None, "") else S0_DEFAULT
        sigma = (float(sigma) if sigma not in (None, "") and float(sigma) > 0
                 else SIGMA_ANN_DEFAULT)
        qtys = [int(q) if q not in (None, "") else 0 for q in qty_state_vals]
        bids = np.array([i["bid"] for i in INSTRUMENTS])
        asks = np.array([i["ask"] for i in INSTRUMENTS])
        qty_arr = np.array(qtys, dtype=float)
        exec_prices = np.array([_exec_price(q, b, a) for q, b, a in zip(qtys, bids, asks)])

        if button == "r4-score-fresh":
            history = []
        else:
            history = list(history or [])
        fresh_hist = list(fresh_hist or [])

        n_batches = 1 if button == "r4-score-once" else 100
        # Mix click counts + history length into seed so each click is unique.
        base_seed = (len(history) * 1009 + int(_once_clicks or 0) * 31337
                     + int(_batch_clicks or 0) * 99991
                     + int(_fresh_clicks or 0) * 65537) % 1_000_000

        new_pnls = []
        for k in range(n_batches):
            seed_k = (base_seed + k * 7919) % 1_000_000
            z = _z_for_seed(seed_k, SCORING_N_SIMS, WEEKS_3_STEPS)
            log_paths_fresh = _log_paths(z, sigma)
            paths_fresh = _paths(s0, log_paths_fresh)
            payoffs_fresh = _payoffs_from_paths(paths_fresh)
            marks = payoffs_fresh.mean(axis=1)
            realized_pnl = float(((marks - exec_prices) * qty_arr * CONTRACT_SIZE).sum())
            new_pnls.append(realized_pnl)
            history.append(realized_pnl)

        if len(history) > 5000:
            history = history[-5000:]

        if button == "r4-score-fresh" and new_pnls:
            arr = np.array(new_pnls, dtype=float)
            fresh_hist.append({
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "n": int(len(arr)),
            })
            if len(fresh_hist) > 200:
                fresh_hist = fresh_hist[-200:]

        latest = new_pnls[-1] if new_pnls else None
        return _format_score_display(history, latest), history, fresh_hist

    @app.callback(
        Output("r4-fresh-history-chart", "figure"),
        Input("r4-fresh-history", "data"),
    )
    def render_fresh_history_chart(fresh_hist):
        return _build_fresh_history_chart(fresh_hist or [])

    @app.callback(
        [Output("r4-risk-lambda", "value"),
         Output("r4-risk-lambda-input", "value")],
        [Input("r4-risk-lambda", "value"),
         Input("r4-risk-lambda-input", "value")],
        prevent_initial_call=True,
    )
    def sync_risk_lambda(slider_val, input_val):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        src = ctx.triggered[0]["prop_id"].split(".")[0]
        if src == "r4-risk-lambda":
            if slider_val is None:
                return no_update, no_update
            return no_update, float(slider_val)
        if src == "r4-risk-lambda-input":
            if input_val is None:
                return no_update, no_update
            v = max(min(float(input_val), -3.0), -12.0)
            return v, no_update
        return no_update, no_update

    @app.callback(
        [Output("r4-mvo-analysis-data", "data"),
         Output("r4-mvo-analysis-status", "children")],
        Input("r4-mvo-analysis-run", "n_clicks"),
        [State("r4-s0", "value"),
         State("r4-sigma", "value"),
         State("r4-n-sims", "value"),
         State("r4-seed", "value"),
         State("r4-mvo-analysis-batches", "value")],
        prevent_initial_call=True,
    )
    def run_mvo_analysis(n_clicks, s0, sigma, n_sims, seed_calib, n_batches):
        s0 = float(s0) if s0 not in (None, "") else S0_DEFAULT
        sigma = (float(sigma) if sigma not in (None, "") and float(sigma) > 0
                 else SIGMA_ANN_DEFAULT)
        n_sims = max(int(n_sims or 10000), 1000)
        seed_calib = int(seed_calib or 0)
        n_batches = max(int(n_batches or 10), 2)

        scoring_seed = (seed_calib * 7919 + int(n_clicks or 0) * 31337
                        + 13) % 1_000_000

        lambdas, avg_pnl, std_pnl, qty_count, _qty_log = _run_mvo_analysis(
            s0, sigma, n_sims, seed_calib,
            -12.0, -3.0, 0.1, n_batches, scoring_seed,
        )
        data = {
            "lambdas": lambdas.tolist(),
            "avg_pnl": avg_pnl.tolist(),
            "std_pnl": std_pnl.tolist(),
            "qty_count": qty_count.tolist(),
            "n_batches": int(n_batches),
        }
        best_i = int(np.argmax(avg_pnl))
        sharpe = np.where(np.array(std_pnl) > 1e-6,
                          np.array(avg_pnl) / np.array(std_pnl), 0.0)
        sharpe_i = int(np.argmax(sharpe))
        status = html.Div([
            html.Div(f"Done: {len(lambdas)} λ × {n_batches} batches."),
            html.Div([
                html.Span(f"Max avg @ log10(λ)={lambdas[best_i]:.1f}: "),
                html.B(f"${avg_pnl[best_i]:,.0f}"),
                html.Span(f" ± ${std_pnl[best_i]:,.0f}"),
            ]),
            html.Div([
                html.Span(f"Max Sharpe @ log10(λ)={lambdas[sharpe_i]:.1f}: "),
                html.B(f"${avg_pnl[sharpe_i]:,.0f}"),
                html.Span(f" ± ${std_pnl[sharpe_i]:,.0f}"),
            ]),
        ])
        return data, status

    @app.callback(
        Output("r4-mvo-analysis-chart", "figure"),
        Input("r4-mvo-analysis-data", "data"),
    )
    def render_mvo_analysis_chart(data):
        if not data:
            return _build_mvo_analysis_fig([], [], [], [], 0)
        return _build_mvo_analysis_fig(
            np.array(data["lambdas"]),
            np.array(data["avg_pnl"]),
            np.array(data["std_pnl"]),
            np.array(data["qty_count"]),
            data["n_batches"],
        )
