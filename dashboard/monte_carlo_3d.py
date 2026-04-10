import numpy as np
import plotly.graph_objects as go
import webbrowser
import os





def plot_3d_surface(
    simulated_paths: np.ndarray,
    tickers: list[str] | None = None,
    n_paths: int = 80,
    t_stride: int = 5,
):
    """
    Animated fullscreen 3D surface with:
      • Multi-sector dropdown  (switch between assets)
      • Growing time animation with Play / Pause
      • Auto-rotating camera orbit
      • Slow, flicker-free animation (fixed z-range, 250 ms frames)

    Parameters
    ----------
    simulated_paths : np.ndarray
        Shape: (N_paths, N_steps+1, N_assets)
    tickers : list[str] | None
        Ticker labels for each asset.  Falls back to "Asset 0", "Asset 1", …
    n_paths : int
        Paths to render (default 150).
    t_stride : int
        Frame interval (default 8 → ~30 frames for 252 steps).
    """
    n_total_paths, n_t, n_assets = simulated_paths.shape
    n_p = min(n_paths, n_total_paths)

    if tickers is None:
        tickers = [f"Asset {i}" for i in range(n_assets)]

    # ── Per-asset z-ranges for proper scaling ──────────────────────────────────
    z_ranges = {}
    for k in range(n_assets):
        d = simulated_paths[:n_p, :, k]
        z_ranges[k] = (float(d.min()), float(d.max()))
    z_min, z_max = z_ranges[0]

    # Frame timesteps — always include last step
    timesteps = list(range(0, n_t, t_stride))
    if timesteps[-1] != n_t - 1:
        timesteps.append(n_t - 1)

    # ── Helper: surface for one asset up to time t_end ────────────────────────
    def make_surface(asset_idx: int, t_end: int) -> go.Surface:
        z = simulated_paths[:n_p, : t_end + 1, asset_idx]
        X, Y = np.meshgrid(np.arange(t_end + 1), np.arange(n_p))
        zlo, zhi = z_ranges[asset_idx]
        return go.Surface(
            x=X, y=Y, z=z,
            colorscale="Viridis",
            cmin=zlo, cmax=zhi,
            colorbar=dict(title="Price (₹)", tickfont=dict(color="white")),
            showscale=True,
        )

    # ── Initial figure — asset 0, full timeline ──────────────────────────────
    fig = go.Figure(data=[make_surface(0, n_t - 1)])

    # ── Frames: grow surface over time (default asset 0) ─────────────────────
    fig.frames = [
        go.Frame(data=[make_surface(0, t)], name=str(t))
        for t in timesteps
    ]

    # ── Slider steps ─────────────────────────────────────────────────────────
    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(t)],
                {
                    "frame": {"duration": 250, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 100},
                },
            ],
            label=f"D{t}",
        )
        for t in timesteps
    ]

    # ── Multi-sector dropdown buttons ─────────────────────────────────────────
    sector_buttons = []
    for k, ticker in enumerate(tickers):
        surface = make_surface(k, n_t - 1)
        zlo, zhi = z_ranges[k]
        sector_buttons.append(
            dict(
                label=ticker,
                method="update",
                args=[
                    {"x": [surface.x], "y": [surface.y], "z": [surface.z],
                     "cmin": [zlo], "cmax": [zhi]},
                    {"scene.zaxis.range": [zlo, zhi]},
                ],
            )
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Monte Carlo 3D Surface | {n_p} paths × {n_t - 1} steps",
            font=dict(size=16, color="white"),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        scene=dict(
            xaxis=dict(title="Time Step (days)", color="white", gridcolor="#333"),
            yaxis=dict(title="Path Index",       color="white", gridcolor="#333"),
            zaxis=dict(
                title="Price (₹)", color="white", gridcolor="#333",
                range=[z_min, z_max],
            ),
            bgcolor="#0f1117",
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        autosize=True,
        width=None,
        height=800,
        margin=dict(l=0, r=0, t=60, b=0),
        # ── Controls ──────────────────────────────────────────────────────────
        updatemenus=[
            # Play / Pause
            dict(
                type="buttons",
                showactive=False,
                y=0.98, x=0.01, xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
            # Sector dropdown
            dict(
                type="dropdown",
                showactive=True,
                y=0.98, x=0.99, xanchor="right", yanchor="top",
                bgcolor="#1a1a2e",
                font=dict(color="white"),
                buttons=sector_buttons,
            ),
        ],
        # ── Time slider ───────────────────────────────────────────────────────
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Day: ", font=dict(color="white")),
                pad=dict(t=50),
                steps=slider_steps,
                font=dict(color="white"),
            )
        ],
    )

    # ── Write fullscreen HTML with auto-rotate JS ─────────────────────────────
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "monte_carlo_fullscreen.html",
    )

    raw_html = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )

    # Inject fullscreen CSS
    raw_html = raw_html.replace(
        "<body>",
        '<body style="margin:0; padding:0; overflow:hidden; background:#0f1117;">',
    )
    raw_html = raw_html.replace(
        'class="plotly-graph-div"',
        'class="plotly-graph-div" style="width:100vw; height:100vh;"',
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(raw_html)

    webbrowser.open("file:///" + out_path.replace("\\", "/"))
    print(f"  3D fullscreen plot → {out_path}")
