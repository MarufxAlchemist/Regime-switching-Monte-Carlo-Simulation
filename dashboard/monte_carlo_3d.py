import numpy as np
import plotly.graph_objects as go


def plot_3d_surface(simulated_paths: np.ndarray, asset_idx: int = 0, n_paths: int = 200):
    """
    Plot a 3D surface of Monte Carlo simulation paths.

    Parameters
    ----------
    simulated_paths : np.ndarray
        Shape: (N_paths, N_steps+1, N_assets)
        Full 3D output from run_simulation().
    asset_idx : int
        Which asset to visualise (default 0 = first asset).
    n_paths : int
        Number of paths to show (default 200 for performance).
    """
    # Slice: (N_paths, N_steps+1, N_assets) → (n_paths, N_steps+1)  for one asset
    z = simulated_paths[:n_paths, :, asset_idx]          # shape: (n_paths, N_steps+1)

    n_paths_actual, n_steps_plus1 = z.shape
    x = np.arange(n_steps_plus1)                         # time axis  (0 … N_steps)
    y = np.arange(n_paths_actual)                         # path index axis

    X, Y = np.meshgrid(x, y)                             # both shape: (n_paths, N_steps+1)

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=z,
        colorscale="Viridis",
        opacity=0.85,
        colorbar=dict(title="Price (₹)", tickfont=dict(color="white")),
    )])

    fig.update_layout(
        title=dict(
            text=f"Monte Carlo Simulation — Asset {asset_idx}  "
                 f"({n_paths_actual} paths × {n_steps_plus1 - 1} steps)",
            font=dict(size=16, color="white"),
        ),
        scene=dict(
            xaxis=dict(title="Time Step (days)", color="white", gridcolor="#333"),
            yaxis=dict(title="Path Index",       color="white", gridcolor="#333"),
            zaxis=dict(title="Price (₹)",        color="white", gridcolor="#333"),
            bgcolor="#0f1117",
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        width=1000,
        height=750,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig.show()
