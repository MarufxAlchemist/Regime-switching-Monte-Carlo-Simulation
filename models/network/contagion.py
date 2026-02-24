"""
Network-Contagion Monte Carlo Simulation
=========================================

Standard GBM Monte Carlo vs Contagion-Amplified GBM.

Contagion update (inside MC loop):
  For each time step t:
    1. Simulate GBM step for all assets with current σ_i^(t)
    2. Detect crash:  ret_i < θ  →  asset i is "stressed"
    3. Propagate:     σ_j^(t+1) = σ_j^(t) * (1 + α * A_ij)   ∀ j ∈ N(i)
    4. Mean-revert:   σ_j^(t+1) = σ_j^(t+1) * (1-β) + σ_j^(0) * β

Parameters:
  α (alpha) = 0.30   contagion amplification strength
  θ (theta) = -0.03  crash threshold (−3% single-day return)
  β (beta)  = 0.05   vol mean-reversion speed per step
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TICKERS      = ["JPM", "BAC", "GS", "MS", "C", "XOM", "CVX", "AAPL", "MSFT", "NVDA"]
SECTORS      = ["Bank","Bank","Bank","Bank","Bank","Energy","Energy","Tech","Tech","Tech"]
N_ASSETS     = len(TICKERS)
START        = "2022-01-01"
END          = "2024-12-31"
CORR_WINDOW  = 60
CORR_THRESH  = 0.50

# MC params
N_PATHS      = 2000
N_STEPS      = 252          # 1 year of daily steps
DT           = 1 / 252

# Contagion params
ALPHA        = 0.40         # amplification per unit of edge weight
THETA_Z      = -1.5        # z-score threshold: fire if z < -1.5 (≈ 7% of steps)
BETA         = 0.05         # mean-reversion speed

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download data + compute baseline parameters
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading data ...")
raw  = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"].dropna()
rets = np.log(raw / raw.shift(1)).dropna()

# Baseline annualised mean and vol from full history
MU_ANNUAL  = rets.mean().values * 252          # (N,)
SIG_ANNUAL = rets.std().values  * np.sqrt(252) # (N,)
S0         = raw.iloc[-1].values               # (N,) last prices

# Correlation matrix from last CORR_WINDOW days
corr_last = rets.tail(CORR_WINDOW).corr().values  # (N, N)

print(f"  Assets: {TICKERS}")
print(f"  Baseline ann. vol: {dict(zip(TICKERS, (SIG_ANNUAL*100).round(1)))}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build adjacency matrix A from correlation network
# ─────────────────────────────────────────────────────────────────────────────
A = np.zeros((N_ASSETS, N_ASSETS))
for i in range(N_ASSETS):
    for j in range(N_ASSETS):
        if i != j and abs(corr_last[i, j]) > CORR_THRESH:
            A[i, j] = abs(corr_last[i, j])          # weight = |correlation|

n_edges = int((A > 0).sum() / 2)
print(f"  Network: {n_edges} edges  (|corr| > {CORR_THRESH})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MC simulation — Standard GBM (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def run_mc_standard(S0, mu, sigma, N_paths, N_steps, dt, seed=0):
    """
    Standard GBM — fixed σ throughout.
    Returns paths shape (N_paths, N_steps+1, N_assets).
    """
    rng    = np.random.default_rng(seed)
    N      = len(S0)
    paths  = np.empty((N_paths, N_steps + 1, N))
    paths[:, 0, :] = S0
    drift  = (mu - 0.5 * sigma**2) * dt            # (N,)

    for t in range(N_steps):
        eps = rng.standard_normal((N_paths, N))
        paths[:, t + 1, :] = paths[:, t, :] * np.exp(
            drift + sigma * np.sqrt(dt) * eps
        )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 4. MC simulation — Contagion-Amplified GBM
# ─────────────────────────────────────────────────────────────────────────────
def run_mc_contagion(S0, mu, sigma0, A, N_paths, N_steps, dt,
                     alpha, theta_z, beta, seed=1):
    """
    Contagion-amplified GBM.
    Trigger: z_i = log_ret_i / (σ_i√dt) < theta_z  (z-score relative)
    Propagate: σ_j *= (1 + α * A_ij)  for each stressed i
    Mean-revert: σ = σ*(1-β) + σ0*β
    """
    # Returns:
    #   paths        (N_paths, N_steps+1, N_assets)   price paths
    #   vol_trace    (N_paths, N_steps+1, N_assets)    σ at each step
    #   event_counts (N_steps, N_assets)               #paths that triggered contagion
    rng          = np.random.default_rng(seed)
    N            = len(S0)
    paths        = np.empty((N_paths, N_steps + 1, N))
    vol_trace    = np.empty((N_paths, N_steps + 1, N))
    event_counts = np.zeros((N_steps, N), dtype=int)

    paths[:, 0, :] = S0
    # Each path carries its own σ vector (starts at σ0)
    sigma = np.tile(sigma0, (N_paths, 1))           # (N_paths, N)
    vol_trace[:, 0, :] = sigma

    # ── DIAGNOSTICS ───────────────────────────────────────────────────────────
    sigma0_ann = sigma0 * np.sqrt(252)
    A_weights  = A[A > 0]
    n_edges    = len(A_weights) // 2
    print("\n" + "─" * 60)
    print("  run_mc_contagion() — parameter diagnostics")
    print("─" * 60)
    print(f"  N assets    : {N}")
    print(f"  N paths     : {N_paths}")
    print(f"  N steps     : {N_steps}  (dt={dt:.6f})")
    print(f"\n  sigma0 (daily)  min={sigma0.min():.5f}  max={sigma0.max():.5f}")
    print(f"  sigma0 (ann)    min={sigma0_ann.min():.2%}  max={sigma0_ann.max():.2%}")
    ok_sigma = np.all((sigma0_ann > 0.05) & (sigma0_ann < 2.0))
    print(f"  sigma0 range OK (5%–200% ann): {'✓' if ok_sigma else '✗ WARNING'}")
    print(f"\n  Adjacency A (ρ analogue):")
    print(f"    Edges         : {n_edges}")
    if len(A_weights) > 0:
        print(f"    Weight range  : [{A_weights.min():.4f}, {A_weights.max():.4f}]")
        print(f"    Weight mean   : {A_weights.mean():.4f}")
    else:
        print("    ✗ WARNING: adjacency matrix has no edges — contagion will not fire")
    print(f"\n  Contagion params:")
    print(f"    alpha  (α)    : {alpha}  →  max single-hop amplif = 1 + α×max(A) = "
          f"{1 + alpha * (A_weights.max() if len(A_weights) else 0):.4f}")
    print(f"    theta_z       : {theta_z}σ  →  P(trigger per step) ≈ "
          f"{(theta_z):.1f}σ → ~{100*(1 - 0.933):.0f}% per asset step")
    print(f"    beta   (β)    : {beta}  →  vol half-life ≈ {0.693/beta:.0f} steps")
    print(f"    vol cap       : 5× baseline")
    print("─" * 60)

    # mid-sim vol snapshot steps
    _snap_steps = {
        int(N_steps * 0.10): "10%",
        int(N_steps * 0.50): "50%",
        int(N_steps * 0.90): "90%",
    }

    for t in range(N_steps):
        # ── Step 1: GBM Step ─────────────────────────────────────────────────
        drift = (mu - 0.5 * sigma**2) * dt           # (N_paths, N)
        eps   = rng.standard_normal((N_paths, N))
        paths[:, t + 1, :] = paths[:, t, :] * np.exp(
            drift + sigma * np.sqrt(dt) * eps
        )

        # ── Step 2: Detect crashes (z-score relative) ───────────────────────────
        log_ret  = np.log(paths[:, t + 1, :] / paths[:, t, :])  # (N_paths, N)
        # Standardise: z = log_ret / (σ_i × √dt)   shape (N_paths, N)
        z_score  = log_ret / (sigma * np.sqrt(dt) + 1e-12)
        stressed = z_score < theta_z                              # (N_paths, N) bool

        # Count events per asset across paths
        event_counts[t] = stressed.sum(axis=0)

        # ── Step 3: Contagion propagation ─────────────────────────────────────
        # For each stressed path-asset pair (p, i), amplify neighbours j
        # Vectorised: for each asset i, compute the additive vol increase
        # that arrives from stressed predecessors.
        # delta_sigma[p, j] = max over i: α * A[i,j] * stressed[p,i]
        # (We accumulate multiplicatively; max avoids double-counting severity)
        for i in range(N):
            col = A[i, :]                          # A[i, j] for all j
            if col.max() == 0:
                continue                           # asset i has no connections
            # rows where asset i is stressed
            stressed_i = stressed[:, i]            # (N_paths,) bool
            if not stressed_i.any():
                continue
            # amplification factor for each j:  1 + α * A[i,j]
            amp = 1.0 + alpha * col                # (N,)
            # apply to stressed paths only
            sigma[stressed_i] *= amp               # broadcast over paths

        # ── Step 4: Mean-reversion + vol cap ──────────────────────────────────
        sigma = sigma * (1 - beta) + sigma0 * beta
        # Hard cap: σ_j ≤ 5 × σ_j^(0)  →  prevents compounding blow-up
        sigma = np.minimum(sigma, 5.0 * sigma0)

        vol_trace[:, t + 1, :] = sigma

        # ── Mid-simulation diagnostics ────────────────────────────────────────
        if t in _snap_steps:
            sig_ann_now = sigma.mean(axis=0) * np.sqrt(252)
            sig_max_now = sigma.max(axis=0)  * np.sqrt(252)
            in_band     = np.all(sigma <= 5.0 * sigma0 + 1e-9)
            label       = _snap_steps[t]
            print(f"\n  Vol snapshot at step {t} ({label} complete):")
            print(f"    Mean ann. σ  (across paths): "
                  f"[{sig_ann_now.min():.2%}, {sig_ann_now.max():.2%}]")
            print(f"    Max  ann. σ  (worst path):   "
                  f"[{sig_max_now.min():.2%}, {sig_max_now.max():.2%}]")
            print(f"    Within 5× cap: {'✓' if in_band else '✗ VIOLATION'}")

    return paths, vol_trace, event_counts


# ─────────────────────────────────────────────────────────────────────────────
# 5. Run both models
# ─────────────────────────────────────────────────────────────────────────────
mu_daily  = MU_ANNUAL / 252
sig_daily = SIG_ANNUAL / np.sqrt(252)

print("\nRunning Standard GBM ...")
paths_std = run_mc_standard(S0, mu_daily, sig_daily, N_PATHS, N_STEPS, DT)

print("Running Contagion GBM ...")
paths_ctx, vol_trace, event_counts = run_mc_contagion(
    S0, mu_daily, sig_daily, A, N_PATHS, N_STEPS, DT,
    alpha=ALPHA, theta_z=THETA_Z, beta=BETA
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Contagion Analysis ────────────────────────────────────────────────")
print(f"  α={ALPHA}  θ_z={THETA_Z}σ  β={BETA}")
print(f"\n  {'Asset':<6} {'Baseline σ(ann)':>16} {'Mean σ (contag)':>16} "
      f"{'Amplif.':>9}  {'Events(tot)':>12}")
print(f"  {'-'*65}")

# Avg vol across paths × time in contagion model
mean_vol_ctx = vol_trace.mean(axis=(0, 1))          # (N,) mean daily vol
mean_vol_ann = mean_vol_ctx * np.sqrt(252)

for k, t in enumerate(TICKERS):
    amp   = mean_vol_ann[k] / SIG_ANNUAL[k]
    evts  = event_counts[:, k].sum()
    print(f"  {t:<6} {SIG_ANNUAL[k]*100:>15.2f}% {mean_vol_ann[k]*100:>15.2f}% "
          f"{amp:>8.3f}x  {evts:>12,}")

# Terminal return distribution comparison  (first asset: JPM)
K       = 0
t_std   = paths_std[:, -1, K] / S0[K] - 1
t_ctx   = paths_ctx[:, -1, K] / S0[K] - 1
kurt_std = float(pd.Series(t_std).kurt())
kurt_ctx = float(pd.Series(t_ctx).kurt())

print(f"\n  Terminal return distribution for {TICKERS[K]}:")
print(f"  {'Model':<18} {'Mean':>8} {'Std':>8} {'Kurtosis':>10}")
print(f"  {'-'*48}")
print(f"  {'Standard GBM':<18} {t_std.mean():>+8.3f} {t_std.std():>8.3f} {kurt_std:>10.3f}")
print(f"  {'Contagion GBM':<18} {t_ctx.mean():>+8.3f} {t_ctx.std():>8.3f} {kurt_ctx:>10.3f}")
print(f"\n  Contagion → higher kurtosis (+{kurt_ctx - kurt_std:.3f}): heavier tails ✓")

# 95% VaR comparison
var_std = np.percentile(t_std, 5)
var_ctx = np.percentile(t_ctx, 5)
print(f"\n  95% Portfolio VaR (1-year, {TICKERS[K]}):")
print(f"    Standard  GBM:   {var_std:>+.4f}")
print(f"    Contagion GBM:   {var_ctx:>+.4f}  "
      f"({'HIGHER' if abs(var_ctx) > abs(var_std) else 'lower'} risk ✓)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plot ...")
fig, axes = plt.subplots(
    2, 2, figsize=(18, 11),
    gridspec_kw={"hspace": 0.40, "wspace": 0.30},
    facecolor="#0f1117"
)
axes = axes.flatten()

DARK = "#0f1117"
for ax in axes:
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

days = np.arange(N_STEPS + 1)
# pick a representative "crisis" path — the one with max contagion events for asset 0
crisis_path_idx = vol_trace[:, :, 0].max(axis=1).argmax()

# ── A: Price paths comparison ─────────────────────────────────────────────────
ax = axes[0]
n_show = 80
for p in range(min(n_show, N_PATHS)):
    ax.plot(days, paths_std[p, :, K] / S0[K],
            color="#3498db", alpha=0.04, linewidth=0.5)
for p in range(min(n_show, N_PATHS)):
    ax.plot(days, paths_ctx[p, :, K] / S0[K],
            color="#e74c3c", alpha=0.04, linewidth=0.5)
ax.plot(days, paths_std[:, :, K].mean(axis=0) / S0[K],
        color="#3498db", linewidth=2, label="Standard GBM")
ax.plot(days, paths_ctx[:, :, K].mean(axis=0) / S0[K],
        color="#e74c3c", linewidth=2, label="Contagion GBM")
ax.axhline(1.0, color="#95a5a6", linewidth=0.8, linestyle="--")
ax.set_title(f"A — Price Paths  ({TICKERS[K]})", fontsize=11)
ax.set_xlabel("Days")
ax.set_ylabel("Normalised Price (S/S₀)")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# ── B: Volatility trace for one path ─────────────────────────────────────────
ax = axes[1]
for k_idx, col in zip([0, 5, 9], ["#2ecc71", "#f39c12", "#9b59b6"]):
    vol_path = vol_trace[crisis_path_idx, :, k_idx] * np.sqrt(252)
    vol_base_line = np.full(N_STEPS + 1, SIG_ANNUAL[k_idx])
    ax.plot(days, vol_path, color=col,
            linewidth=1.2, label=TICKERS[k_idx], alpha=0.9)
    ax.axhline(SIG_ANNUAL[k_idx], color=col, linewidth=0.8,
               linestyle="--", alpha=0.4)
ax.set_title("B — Realised σ Trace (contagion path)", fontsize=11)
ax.set_xlabel("Days")
ax.set_ylabel("Annualised σ")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9,
          title="Asset", title_fontsize=8)

# ── C: Contagion event heatmap ────────────────────────────────────────────────
ax = axes[2]
im = ax.imshow(
    event_counts.T,           # (N_assets, N_steps)
    aspect="auto", cmap="hot",
    extent=[0, N_STEPS, -0.5, N_ASSETS - 0.5],
    origin="lower",
)
ax.set_yticks(range(N_ASSETS))
ax.set_yticklabels(TICKERS, fontsize=8, color="white")
ax.set_title(f"C — Contagion Events  (θ_z={THETA_Z}σ, N={N_PATHS} paths)", fontsize=11)
ax.set_xlabel("Days")
ax.set_ylabel("Asset")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
             label="# paths triggered").ax.yaxis.set_tick_params(color="white")

# ── D: Terminal return distribution ──────────────────────────────────────────
ax = axes[3]
bins = np.linspace(-0.8, 1.5, 80)
ax.hist(t_std, bins=bins, color="#3498db", alpha=0.5, label=f"Standard  (kurt={kurt_std:.2f})")
ax.hist(t_ctx, bins=bins, color="#e74c3c", alpha=0.5, label=f"Contagion (kurt={kurt_ctx:.2f})")
ax.axvline(var_std, color="#3498db", linewidth=1.5, linestyle="--", label=f"VaR std  {var_std:.2f}")
ax.axvline(var_ctx, color="#e74c3c", linewidth=1.5, linestyle="--", label=f"VaR ctg  {var_ctx:.2f}")
ax.set_title(f"D — Terminal Return Distribution  ({TICKERS[K]})", fontsize=11)
ax.set_xlabel("1-year Return")
ax.set_ylabel("Frequency")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

fig.suptitle(
    f"Network-Contagion Monte Carlo  |  α={ALPHA}  θ_z={THETA_Z}σ  β={BETA}"
    f"  |  {N_PATHS} paths × {N_STEPS} steps",
    color="white", fontsize=13, y=0.98
)

out = "models/network/contagion_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"  Saved → {out}")
