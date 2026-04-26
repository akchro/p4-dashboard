"""Round 3 manual: The Celestial Gardeners' Guild — bid against NPC reserves.

Reserves R ~ Uniform{670, 675, ..., 920}; we sell at 920. Submit (b1, b2) with
b1 < b2 (else b2 is dominated). Trade resolution per counterparty:

    R < b1                         -> trade at b1, profit = 920 - b1
    b1 <= R < b2 and b2 >  avg_b2  -> trade at b2, profit = 920 - b2
    b1 <= R < b2 and b2 <= avg_b2  -> trade at b2, profit = (920-avg_b2)^3 / (920-b2)^2

avg_b2 is the mean of every (human) player's second bid.

Continuous Nash: split [670, 920] into thirds. b1 = 753.33, b2 = 836.67,
profit = 250/3 ~= 83.33. Discrete optimum (bid just above a 5-multiple):
b1 = 751, b2 = 836, profit ~= 84.33.
"""

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

R_MIN, R_MAX, R_STEP = 670, 920, 5
SELL_PRICE = 920
RESERVE_VALUES = np.arange(R_MIN, R_MAX + 1, R_STEP)
N_RESERVES = len(RESERVE_VALUES)  # 51

NASH_B1_C = R_MIN + (SELL_PRICE - R_MIN) / 3        # 753.33
NASH_B2_C = R_MIN + 2 * (SELL_PRICE - R_MIN) / 3    # 836.67
NASH_PROFIT_C = (SELL_PRICE - R_MIN) / 3            # 83.33

NASH_B1_D = 751
NASH_B2_D = 836

CONTAINER_STYLE = {
    "display": "grid",
    "gridTemplateRows": "55vh 30vh auto",
    "gap": "2px",
    "height": "100%",
}


def _sample_population(weights, knobs, n, seed):
    """Sample n second-bid values from a 5-component mixture population."""
    rng = np.random.default_rng(int(seed))
    w = np.maximum(np.array(weights, dtype=float), 0.0)
    if w.sum() <= 0 or n <= 0:
        return np.array([])
    w = w / w.sum()
    counts = np.floor(w * n).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        frac = w * n - counts
        idx = np.argsort(-frac)[:remainder]
        counts[idx] += 1

    parts = []
    if counts[0] > 0:
        parts.append(np.full(counts[0], float(knobs["nash_b2"])))
    if counts[1] > 0:
        s = rng.normal(knobs["conc_mid"], max(knobs["conc_std"], 1e-3), counts[1])
        parts.append(np.clip(s, R_MIN, SELL_PRICE))
    if counts[2] > 0:
        s = rng.normal(
            knobs["nash_b2"] + knobs["higher_offset"],
            max(knobs["higher_std"], 1e-3),
            counts[2],
        )
        parts.append(np.clip(s, R_MIN, SELL_PRICE))
    if counts[3] > 0:
        lo, hi = sorted([float(knobs["rand_lo"]), float(knobs["rand_hi"])])
        if hi <= lo:
            parts.append(np.full(counts[3], lo))
        else:
            parts.append(rng.uniform(lo, hi, counts[3]))
    if counts[4] > 0:
        nice = np.array(knobs["nice"], dtype=float)
        if len(nice) == 0:
            nice = np.array([700, 750, 800, 850, 900], dtype=float)
        parts.append(rng.choice(nice, counts[4]))
    return np.concatenate(parts) if parts else np.array([])


def _b2_profit(b2, avg_b2):
    if b2 > avg_b2:
        return SELL_PRICE - b2
    if b2 < SELL_PRICE:
        return (SELL_PRICE - avg_b2) ** 3 / (SELL_PRICE - b2) ** 2
    return 0.0


def _profit(b1, b2, avg_b2):
    """Expected profit per counterparty for strategy (b1, b2) given pop avg_b2."""
    if b1 >= b2:
        n1 = int(np.sum(RESERVE_VALUES < b1))
        return n1 / N_RESERVES * (SELL_PRICE - b1)
    n1 = int(np.sum(RESERVE_VALUES < b1))
    n2 = int(np.sum((RESERVE_VALUES >= b1) & (RESERVE_VALUES < b2)))
    profit_b1 = SELL_PRICE - b1
    profit_b2 = _b2_profit(b2, avg_b2)
    return (n1 * profit_b1 + n2 * profit_b2) / N_RESERVES


def _profit_grid(b1_vals, b2_vals, avg_b2):
    """Profit over (b1 x b2) grid. NaN where b1 >= b2 (single-bid region)."""
    n1_counts = np.array([(RESERVE_VALUES < b1).sum() for b1 in b1_vals], dtype=float)
    profit_b1 = SELL_PRICE - b1_vals
    grid = np.full((len(b2_vals), len(b1_vals)), np.nan)
    for i, b2 in enumerate(b2_vals):
        mask = b1_vals < b2
        if not mask.any():
            continue
        n_below_b2 = (RESERVE_VALUES < b2).sum()
        n2 = np.maximum(n_below_b2 - n1_counts, 0)
        pb2 = _b2_profit(b2, avg_b2)
        prof = (n1_counts * profit_b1 + n2 * pb2) / N_RESERVES
        row = grid[i, :]
        row[mask] = prof[mask]
    return grid


def _mc_avg_b2(weights, knobs, n_players, base_seed, n_mc):
    """Run n_mc independent population draws, return their avg_b2 values."""
    out = np.empty(int(n_mc), dtype=float)
    for i in range(int(n_mc)):
        s = _sample_population(weights, knobs, n_players, int(base_seed) + i * 9973)
        out[i] = float(s.mean()) if len(s) > 0 else float(NASH_B2_D)
    return out


def _expected_b2_profit_mc(b2_vals, avg_samples):
    """E_avg[profit_b2(b2, avg)] for each b2, averaged over MC samples of avg."""
    avg = avg_samples[None, :]            # (1, M)
    b2 = np.asarray(b2_vals)[:, None]     # (B, 1)
    above = b2 > avg
    safe_denom = np.maximum((SELL_PRICE - b2) ** 2, 1e-9)
    penalised = (SELL_PRICE - avg) ** 3 / safe_denom
    full = SELL_PRICE - b2
    return np.where(above, full, penalised).mean(axis=1)  # (B,)


def _profit_grid_mc(b1_vals, b2_vals, avg_samples):
    """Profit grid where each (b1, b2) entry is averaged over MC avg_b2 draws."""
    n1_counts = np.array([(RESERVE_VALUES < b1).sum() for b1 in b1_vals], dtype=float)
    profit_b1 = SELL_PRICE - b1_vals
    e_pb2 = _expected_b2_profit_mc(b2_vals, avg_samples)  # (B,)
    grid = np.full((len(b2_vals), len(b1_vals)), np.nan)
    for i, b2 in enumerate(b2_vals):
        mask = b1_vals < b2
        if not mask.any():
            continue
        n_below_b2 = (RESERVE_VALUES < b2).sum()
        n2 = np.maximum(n_below_b2 - n1_counts, 0)
        prof = (n1_counts * profit_b1 + n2 * e_pb2[i]) / N_RESERVES
        row = grid[i, :]
        row[mask] = prof[mask]
    return grid


def _profit_mc(b1, b2, avg_samples):
    """Per-counterparty profit averaged over MC avg_b2 samples."""
    if b1 >= b2:
        n1 = int(np.sum(RESERVE_VALUES < b1))
        return n1 / N_RESERVES * (SELL_PRICE - b1)
    n1 = int(np.sum(RESERVE_VALUES < b1))
    n2 = int(np.sum((RESERVE_VALUES >= b1) & (RESERVE_VALUES < b2)))
    pb2_arr = np.array([_b2_profit(b2, a) for a in avg_samples])
    return (n1 * (SELL_PRICE - b1) + n2 * pb2_arr.mean()) / N_RESERVES


def _slider(slider_id, lo, hi, step, value, marks):
    return dcc.Slider(
        id=slider_id, min=lo, max=hi, step=step, value=value,
        marks=marks, tooltip={"placement": "bottom"},
    )


def controls_layout():
    return html.Div(id="manual-round3-controls", style={"display": "none"}, children=[
        html.Div(html.B("Player Population (%)"), style={"marginBottom": "4px"}),
        html.Div("Mixture weights — will be normalized.", style={
            "fontSize": "11px", "color": "#555", "marginBottom": "8px",
        }),

        html.Label("% Perfect Nash", style={"fontSize": "11px"}),
        _slider("r3-w-nash", 0, 100, 1, 15, {0: "0", 50: "50", 100: "100"}),
        html.Label("% Concentrated around mid", style={"fontSize": "11px"}),
        _slider("r3-w-conc", 0, 100, 1, 25, {0: "0", 50: "50", 100: "100"}),
        html.Label("% Slightly higher than Nash", style={"fontSize": "11px"}),
        _slider("r3-w-high", 0, 100, 1, 50, {0: "0", 50: "50", 100: "100"}),
        html.Label("% Random", style={"fontSize": "11px"}),
        _slider("r3-w-rand", 0, 100, 1, 5, {0: "0", 50: "50", 100: "100"}),
        html.Label("% Nice numbers", style={"fontSize": "11px"}),
        _slider("r3-w-nice", 0, 100, 1, 5, {0: "0", 50: "50", 100: "100"}),

        html.Hr(),
        html.B("Group knobs"),

        html.Label(f"Nash b2 (theory = {NASH_B2_D})", style={"fontSize": "11px"}),
        _slider("r3-nash-b2", R_MIN, SELL_PRICE - 1, 1, NASH_B2_D,
                {R_MIN: str(R_MIN), 795: "795", NASH_B2_D: str(NASH_B2_D),
                 SELL_PRICE - 1: str(SELL_PRICE - 1)}),

        html.Label("Concentrated mid", style={"fontSize": "11px"}),
        _slider("r3-conc-mid", R_MIN, SELL_PRICE, 1, 795,
                {R_MIN: str(R_MIN), 795: "795", 836: "836", SELL_PRICE: str(SELL_PRICE)}),
        html.Label("Concentrated std", style={"fontSize": "11px"}),
        _slider("r3-conc-std", 1, 60, 1, 25, {1: "1", 25: "25", 60: "60"}),

        html.Label("Slightly-higher offset (above Nash b2)", style={"fontSize": "11px"}),
        _slider("r3-high-off", 0, 50, 1, 10, {0: "0", 25: "25", 50: "50"}),
        html.Label("Slightly-higher std", style={"fontSize": "11px"}),
        _slider("r3-high-std", 1, 30, 1, 5, {1: "1", 15: "15", 30: "30"}),

        html.Label("Random low", style={"fontSize": "11px"}),
        _slider("r3-rand-lo", R_MIN, SELL_PRICE, 5, R_MIN,
                {R_MIN: str(R_MIN), 795: "795", SELL_PRICE: str(SELL_PRICE)}),
        html.Label("Random high", style={"fontSize": "11px"}),
        _slider("r3-rand-hi", R_MIN, SELL_PRICE, 5, SELL_PRICE,
                {R_MIN: str(R_MIN), 795: "795", SELL_PRICE: str(SELL_PRICE)}),

        html.Label("Nice numbers (comma-separated)", style={"fontSize": "11px"}),
        dcc.Input(id="r3-nice-list", value="700,750,800,850,900", type="text",
                  style={"width": "100%", "fontSize": "11px"}),

        html.Hr(),
        html.Label("Number of other players", style={"fontWeight": "bold"}),
        _slider("r3-n-players", 10, 500, 10, 100,
                {10: "10", 100: "100", 500: "500"}),
        html.Label("Seed", style={"fontWeight": "bold"}),
        _slider("r3-seed", 0, 99, 1, 42, {0: "0", 50: "50", 99: "99"}),
        html.Label("Monte Carlo runs (1 = single sample)",
                   style={"fontWeight": "bold"}),
        _slider("r3-mc-runs", 1, 1000, 1, 1,
                {1: "1", 50: "50", 200: "200", 1000: "1000"}),
        html.Div(
            "MC averages the profit surface across many independent population draws — "
            "use it to find the bid that's robustly best across the field's sampling noise.",
            style={"fontSize": "11px", "color": "#555", "marginTop": "4px"},
        ),

        html.Hr(),
        html.B("Your strategy (overlay on heatmap)"),
        html.Label("Your b1", style={"fontSize": "11px"}),
        _slider("r3-my-b1", R_MIN, SELL_PRICE - 1, 1, NASH_B1_D,
                {R_MIN: str(R_MIN), NASH_B1_D: str(NASH_B1_D),
                 SELL_PRICE - 1: str(SELL_PRICE - 1)}),
        html.Label("Your b2", style={"fontSize": "11px"}),
        _slider("r3-my-b2", R_MIN, SELL_PRICE - 1, 1, NASH_B2_D,
                {R_MIN: str(R_MIN), NASH_B2_D: str(NASH_B2_D),
                 SELL_PRICE - 1: str(SELL_PRICE - 1)}),
    ])


def charts_layout():
    return html.Div(id="manual-round3-container", style={**CONTAINER_STYLE, "display": "none"}, children=[
        html.Div(dcc.Graph(id="r3-heatmap", style={"height": "100%"}),
                 style={"border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Div(dcc.Graph(id="r3-pop-dist", style={"height": "100%"}),
                 style={"border": "1px solid #ddd", "borderRadius": "4px"}),
        html.Div(id="r3-summary", style={
            "border": "1px solid #ddd", "borderRadius": "4px",
            "padding": "10px", "fontSize": "12px", "overflowY": "auto",
        }),
    ])


def _summary_row(label, bids, profit, *, highlight=False):
    style = {"borderTop": "1px solid #eee"}
    if highlight:
        style["background"] = "#fffbe6"
        style["fontWeight"] = "bold"
    return html.Tr(style=style, children=[
        html.Td(label, style={"padding": "4px 8px"}),
        html.Td(bids, style={"padding": "4px 8px", "color": "#666"}),
        html.Td(f"{profit:.2f}", style={"padding": "4px 8px", "textAlign": "right"}),
    ])


def register_callbacks(app):
    @app.callback(
        [Output("r3-heatmap", "figure"),
         Output("r3-pop-dist", "figure"),
         Output("r3-summary", "children")],
        [Input("r3-w-nash", "value"),
         Input("r3-w-conc", "value"),
         Input("r3-w-high", "value"),
         Input("r3-w-rand", "value"),
         Input("r3-w-nice", "value"),
         Input("r3-nash-b2", "value"),
         Input("r3-conc-mid", "value"),
         Input("r3-conc-std", "value"),
         Input("r3-high-off", "value"),
         Input("r3-high-std", "value"),
         Input("r3-rand-lo", "value"),
         Input("r3-rand-hi", "value"),
         Input("r3-nice-list", "value"),
         Input("r3-n-players", "value"),
         Input("r3-seed", "value"),
         Input("r3-mc-runs", "value"),
         Input("r3-my-b1", "value"),
         Input("r3-my-b2", "value")],
    )
    def update(w_nash, w_conc, w_high, w_rand, w_nice,
               nash_b2, conc_mid, conc_std, high_off, high_std,
               rand_lo, rand_hi, nice_str, n_players, seed, mc_runs,
               my_b1, my_b2):
        try:
            nice = [float(x.strip()) for x in (nice_str or "").split(",") if x.strip()]
        except ValueError:
            nice = []
        if not nice:
            nice = [700, 750, 800, 850, 900]
        knobs = {
            "nash_b2": float(nash_b2),
            "conc_mid": float(conc_mid),
            "conc_std": float(conc_std),
            "higher_offset": float(high_off),
            "higher_std": float(high_std),
            "rand_lo": float(rand_lo),
            "rand_hi": float(rand_hi),
            "nice": nice,
        }
        weights = [w_nash, w_conc, w_high, w_rand, w_nice]
        n_mc = max(1, int(mc_runs))
        mc_mode = n_mc > 1

        b1_vals = np.arange(R_MIN, SELL_PRICE, 1, dtype=float)
        b2_vals = np.arange(R_MIN, SELL_PRICE, 1, dtype=float)

        if mc_mode:
            avg_samples = _mc_avg_b2(weights, knobs, int(n_players), int(seed), n_mc)
            avg_b2 = float(avg_samples.mean())
            avg_std = float(avg_samples.std())
            avg_p5 = float(np.percentile(avg_samples, 5))
            avg_p95 = float(np.percentile(avg_samples, 95))
            grid = _profit_grid_mc(b1_vals, b2_vals, avg_samples)
            # Last sample for the (now secondary) bid distribution view
            last_pop = _sample_population(weights, knobs, int(n_players),
                                          int(seed) + (n_mc - 1) * 9973)
        else:
            last_pop = _sample_population(weights, knobs, int(n_players), int(seed))
            avg_b2 = float(last_pop.mean()) if len(last_pop) > 0 else float(NASH_B2_D)
            avg_samples = np.array([avg_b2])
            avg_std = 0.0
            avg_p5 = avg_p95 = avg_b2
            grid = _profit_grid(b1_vals, b2_vals, avg_b2)

        flat_idx = int(np.nanargmax(grid))
        opt_i, opt_j = np.unravel_index(flat_idx, grid.shape)
        opt_b1, opt_b2 = float(b1_vals[opt_j]), float(b2_vals[opt_i])
        opt_profit = float(grid[opt_i, opt_j])

        title_suffix = (
            f"averaged over {n_mc} MC runs · avg b2 = {avg_b2:.2f} ± {avg_std:.2f}"
            if mc_mode else
            f"single sample · avg b2 = {avg_b2:.2f}"
        )

        heat = go.Figure()
        heat.add_trace(go.Heatmap(
            x=b1_vals, y=b2_vals, z=grid,
            colorscale="Viridis",
            colorbar={"title": "E[π]"},
            hovertemplate="b1=%{x:.0f}<br>b2=%{y:.0f}<br>E[π]=%{z:.2f}<extra></extra>",
        ))
        heat.add_trace(go.Scatter(
            x=[NASH_B1_D], y=[NASH_B2_D], mode="markers+text",
            marker={"symbol": "x", "size": 14, "color": "white",
                    "line": {"color": "black", "width": 2}},
            text=["Nash"], textposition="top right",
            name="Nash (theory)",
        ))
        heat.add_trace(go.Scatter(
            x=[opt_b1], y=[opt_b2], mode="markers+text",
            marker={"symbol": "star", "size": 18, "color": "yellow",
                    "line": {"color": "black", "width": 1.5}},
            text=[f"Exploit  E[π]={opt_profit:.1f}"], textposition="bottom right",
            name="Robust exploit" if mc_mode else "Exploit (given pop)",
        ))
        heat.add_trace(go.Scatter(
            x=[my_b1], y=[my_b2], mode="markers+text",
            marker={"symbol": "circle", "size": 12, "color": "red",
                    "line": {"color": "white", "width": 2}},
            text=["You"], textposition="top left",
            name="Your strategy",
        ))
        heat.update_layout(
            title=f"E[profit] surface — {title_suffix}",
            xaxis_title="b1 (first / lower bid)",
            yaxis_title="b2 (second / higher bid)",
            template="plotly_white",
            margin={"l": 60, "r": 80, "t": 40, "b": 40},
            xaxis={"range": [R_MIN, SELL_PRICE]},
            yaxis={"range": [R_MIN, SELL_PRICE]},
            legend={"orientation": "h", "y": -0.18},
        )

        dist = go.Figure()
        if mc_mode:
            dist.add_trace(go.Histogram(
                x=avg_samples, nbinsx=min(60, max(10, n_mc // 4)),
                marker={"color": "#1f77b4", "line": {"color": "white", "width": 0.5}},
                name="avg_b2 across runs",
            ))
            dist.add_vline(x=avg_b2, line={"color": "red", "dash": "dash", "width": 2},
                           annotation_text=f"mean={avg_b2:.2f}",
                           annotation_position="top right")
            dist.add_vline(x=NASH_B2_D, line={"color": "blue", "dash": "dot"},
                           annotation_text=f"Nash b2={NASH_B2_D}",
                           annotation_position="top left")
            dist_title = (
                f"Distribution of population avg_b2 across {n_mc} MC runs "
                f"(p5={avg_p5:.1f} · p95={avg_p95:.1f})"
            )
        elif len(last_pop) > 0:
            dist.add_trace(go.Histogram(
                x=last_pop, nbinsx=60,
                marker={"color": "#9467bd", "line": {"color": "white", "width": 0.5}},
                name="Population b2",
            ))
            dist.add_vline(x=avg_b2, line={"color": "red", "dash": "dash", "width": 2},
                           annotation_text=f"avg={avg_b2:.1f}",
                           annotation_position="top right")
            dist.add_vline(x=NASH_B2_D, line={"color": "blue", "dash": "dot"},
                           annotation_text=f"Nash b2={NASH_B2_D}",
                           annotation_position="top left")
            dist_title = f"Population b2 distribution (n={int(n_players)})"
        else:
            dist_title = "Population b2 distribution (empty)"

        dist.update_layout(
            title=dist_title,
            xaxis_title="avg_b2" if mc_mode else "b2",
            yaxis_title="count",
            xaxis={"range": [R_MIN - 10, SELL_PRICE + 10]},
            template="plotly_white",
            margin={"l": 60, "r": 20, "t": 40, "b": 40},
            showlegend=False,
        )

        if mc_mode:
            my_profit = _profit_mc(my_b1, my_b2, avg_samples)
            nash_profit = _profit_mc(NASH_B1_D, NASH_B2_D, avg_samples)
        else:
            my_profit = _profit(my_b1, my_b2, avg_b2)
            nash_profit = _profit(NASH_B1_D, NASH_B2_D, avg_b2)
        b1_only = [_profit(b, b, avg_b2) for b in range(R_MIN, SELL_PRICE)]
        single_best = R_MIN + int(np.argmax(b1_only))
        single_best_profit = float(np.max(b1_only))

        gain_vs_nash = my_profit - nash_profit
        opt_vs_nash = opt_profit - nash_profit

        exploit_label = "Robust exploit (MC mean)" if mc_mode else "Optimal exploit (given pop)"
        summary = html.Div([
            html.Table(style={"width": "100%", "fontSize": "12px",
                              "borderCollapse": "collapse"}, children=[
                html.Thead(html.Tr(style={"borderBottom": "2px solid #333"}, children=[
                    html.Th("Strategy", style={"padding": "4px 8px", "textAlign": "left"}),
                    html.Th("Bids", style={"padding": "4px 8px", "textAlign": "left"}),
                    html.Th("E[π] / counterparty",
                            style={"padding": "4px 8px", "textAlign": "right"}),
                ])),
                html.Tbody([
                    _summary_row("Nash (continuous)",
                                 f"b1={NASH_B1_C:.2f}, b2={NASH_B2_C:.2f}",
                                 NASH_PROFIT_C),
                    _summary_row("Nash (discrete)",
                                 f"b1={NASH_B1_D}, b2={NASH_B2_D}", nash_profit),
                    _summary_row("Best single bid",
                                 f"b={single_best}", single_best_profit),
                    _summary_row("Your strategy",
                                 f"b1={my_b1}, b2={my_b2}", my_profit),
                    _summary_row(exploit_label,
                                 f"b1={int(opt_b1)}, b2={int(opt_b2)}",
                                 opt_profit, highlight=True),
                ]),
            ]),
            html.Div(style={"marginTop": "8px", "fontSize": "11px", "color": "#555"}, children=[
                html.Div([
                    html.B("avg b2: "),
                    f"{avg_b2:.2f}" + (f" ± {avg_std:.2f}  " if mc_mode else "  "),
                    html.B("·  Nash b2: "), f"{NASH_B2_D}  ",
                    html.B("·  MC runs: "), f"{n_mc}  ",
                    html.B("·  Your Δ vs Nash: "), f"{gain_vs_nash:+.2f}  ",
                    html.B("·  Exploit Δ vs Nash: "), f"{opt_vs_nash:+.2f}",
                ]),
                html.Div([
                    "When b2 ≤ avg, profit per b2-trade collapses as ",
                    "(920-avg)³ / (920-b2)². The exploit is to bid b2 just above ",
                    "avg (or stay at Nash if pop is below it), then set b1 = (670+b2)/2.",
                    html.Br(),
                    html.B("MC mode: ") if mc_mode else "",
                    ("the heatmap, summary profits, and exploit are averaged over "
                     f"{n_mc} independent population draws — this finds the bid that's "
                     "robust to sampling noise rather than tuned to one realised seed.")
                    if mc_mode else "",
                ], style={"marginTop": "4px"}),
            ]),
        ])

        return heat, dist, summary
