from dash import html, dcc

# Defaults mirror velvet_zscore.py — kept in sync by hand because that file
# imports `datamodel` (only available in the Prosperity sandbox), so we can't
# import it from the dashboard process.
DEFAULT_MU = 5250.0
DEFAULT_SIGMA = 15.6
DEFAULT_Z_ENTRY = 1.5
DEFAULT_Z_EXIT = 0.25

UNDERLYING = 'VELVETFRUIT_EXTRACT'


def layout(prefix: str):
    """Sidebar block for the VFE z-score overlay.

    `prefix` namespaces all IDs (e.g. "live", "hist") so the same component
    can be mounted in multiple sidebars without ID collisions.
    """
    p = prefix
    return html.Div(id=f"{p}-zscore-panel", children=[
        html.Label("VFE Z-Score Overlay", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id=f"{p}-zscore-toggle",
            options=[{"label": " Show on VELVETFRUIT_EXTRACT", "value": "show"}],
            value=[],
            inputStyle={"marginRight": "4px"},
            style={"fontSize": "12px"},
        ),
        html.Label(f"Mean (default {DEFAULT_MU:.0f})",
                   style={"fontSize": "11px", "color": "#444"}),
        dcc.Slider(
            id=f"{p}-zscore-mean",
            min=5200, max=5300, step=0.5, value=DEFAULT_MU,
            marks={5200: "5200", 5250: "5250", 5300: "5300"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label(f"Z entry (default {DEFAULT_Z_ENTRY})",
                   style={"fontSize": "11px", "color": "#444"}),
        dcc.Slider(
            id=f"{p}-zscore-entry",
            min=0.5, max=3.0, step=0.1, value=DEFAULT_Z_ENTRY,
            marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0", 3.0: "3.0"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Label(f"Z exit (default {DEFAULT_Z_EXIT})",
                   style={"fontSize": "11px", "color": "#444"}),
        dcc.Slider(
            id=f"{p}-zscore-exit",
            min=0.0, max=1.5, step=0.05, value=DEFAULT_Z_EXIT,
            marks={0: "0", 0.25: "0.25", 0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ])


def add_lines(fig, product: str, toggle, mu: float, z_entry: float, z_exit: float,
              sigma: float = DEFAULT_SIGMA):
    """Draw mean / ±Z_ENTRY*σ / ±Z_EXIT*σ horizontal lines on `fig`.

    No-op unless `product == UNDERLYING` and `toggle` contains "show".
    """
    if product != UNDERLYING:
        return
    if not toggle or "show" not in toggle:
        return

    fig.add_hline(y=mu, line_dash="dash", line_color="#666",
                  annotation_text=f"μ={mu:.1f}", annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#666")

    upper_entry = mu + z_entry * sigma
    lower_entry = mu - z_entry * sigma
    fig.add_hline(y=upper_entry, line_dash="dot", line_color="#D32F2F",
                  annotation_text=f"+{z_entry}σ short",
                  annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#D32F2F")
    fig.add_hline(y=lower_entry, line_dash="dot", line_color="#388E3C",
                  annotation_text=f"-{z_entry}σ long",
                  annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#388E3C")

    if z_exit > 0:
        upper_exit = mu + z_exit * sigma
        lower_exit = mu - z_exit * sigma
        fig.add_hline(y=upper_exit, line_dash="dashdot", line_color="#F57C00",
                      annotation_text=f"+{z_exit}σ exit",
                      annotation_position="right",
                      annotation_font_size=10, annotation_font_color="#F57C00")
        fig.add_hline(y=lower_exit, line_dash="dashdot", line_color="#F57C00",
                      annotation_text=f"-{z_exit}σ exit",
                      annotation_position="right",
                      annotation_font_size=10, annotation_font_color="#F57C00")


def register_callbacks(app):
    # Pure presentation — no callbacks owned here. The chart modules read the
    # slider/toggle values directly via Input() and call add_lines().
    pass
