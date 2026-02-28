"""
Financial Network Engine
========================
Step 1 — Rolling 60-day pairwise return correlation
Step 2 — Build NetworkX graph: edge weight = |corr|, edge exists if |corr| > threshold
Step 3 — Compute eigenvector centrality & network density (at each rolling window)

Final snapshot: full report + network visualisation saved to network_plot.png
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "JPM", "BAC", "GS", "MS", "C",        # Banks
    "XOM", "CVX",                           # Energy
    "AAPL", "MSFT", "NVDA",                # Tech
]
START      = "2019-01-01"
END        = "2024-12-31"
WINDOW     = 60           # rolling window in trading days
CORR_THRESH = 0.60        # minimum |correlation| to draw an edge

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Download data
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading price data ...")
raw   = yf.download(TICKERS, start=START, end=END,
                    auto_adjust=True, progress=False)["Close"]
raw   = raw.dropna()
rets  = np.log(raw / raw.shift(1)).dropna()
print(f"  {len(raw)} days × {len(TICKERS)} assets")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Rolling 60-day correlation
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1 — Computing rolling 60-day correlations ...")

roll_corr = {}      # date → (N×N) correlation DataFrame.
dates_roll = rets.index[WINDOW - 1:]

for i, date in enumerate(dates_roll):
    window_rets = rets.iloc[i : i + WINDOW]
    roll_corr[date] = window_rets.corr()

print(f"  Computed {len(roll_corr)} rolling correlation matrices")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build NetworkX graph (at each date); track metrics over time
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2 — Building NetworkX graphs ...")

def build_graph(corr_df: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Nodes = tickers (always present).
    Edge (i,j) exists iff |corr[i,j]| > threshold (i ≠ j).
    Edge weight = |corr[i,j]|.
    """
    G = nx.Graph()
    G.add_nodes_from(corr_df.columns)
    n = len(corr_df)
    for a in range(n):
        for b in range(a + 1, n):
            c = corr_df.iloc[a, b]
            if abs(c) > threshold:
                G.add_edge(
                    corr_df.columns[a],
                    corr_df.columns[b],
                    weight=abs(c),
                    sign=np.sign(c),
                )
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Robust eigenvector centrality (handles disconnected graphs)
# ─────────────────────────────────────────────────────────────────────────────
def eigenvector_centrality_robust(G: nx.Graph) -> dict:
    """
    Eigenvector centrality for possibly-disconnected graphs.
    Strategy:
      - For each connected component, compute eigenvector centrality
        scaled by the component's fractional size (node_count / total_nodes).
      - Isolated nodes get centrality 0.
      - Falls back to degree centrality if numpy solver fails.
    """
    result = {n: 0.0 for n in G.nodes()}
    N = G.number_of_nodes()
    if N == 0:
        return result
    for component in nx.connected_components(G):
        sub = G.subgraph(component).copy()
        n_sub = len(sub)
        if n_sub < 2:
            continue                   # single isolated node  → 0
        scale = n_sub / N             # weight by component size
        try:
            ec = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except Exception:
            # fallback: weighted degree centrality
            deg = dict(sub.degree(weight="weight"))
            total = sum(deg.values()) or 1
            ec = {n: v / total for n, v in deg.items()}
        for node, val in ec.items():
            result[node] = val * scale
    return result


print("\nStep 3 — Computing eigenvector centrality & density over time ...")

density_ts    = {}
centrality_ts = {}          # date → {ticker: centrality}

for date, corr_df in roll_corr.items():
    G = build_graph(corr_df, CORR_THRESH)

    # Network density: fraction of possible edges that are present.
    density_ts[date] = nx.density(G)

    # Eigenvector centrality (may not converge if graph is disconnected/sparse)
    try:
        ec = eigenvector_centrality_robust(G)
    except Exception:
        ec = {t: 0.0 for t in TICKERS}
    centrality_ts[date] = ec

density_s     = pd.Series(density_ts)
centrality_df = pd.DataFrame(centrality_ts).T     # rows=dates, cols=tickers

print(f"  Done.  Avg density: {density_s.mean():.4f}  "
      f"  Peak density: {density_s.max():.4f} on {density_s.idxmax().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# Final Snapshot — full report on the most recent window
# ─────────────────────────────────────────────────────────────────────────────
last_date  = dates_roll[-1]
G_final    = build_graph(roll_corr[last_date], CORR_THRESH)
density_f  = nx.density(G_final)
try:
    ec_final   = eigenvector_centrality_robust(G_final)
except Exception:
    ec_final   = {t: 0.0 for t in TICKERS}

SEP = "=" * 60
print(f"\n{SEP}")
print(f"  NETWORK SNAPSHOT  —  {last_date.date()}  (last {WINDOW}-day window)")
print(SEP)
print(f"\n  Nodes : {G_final.number_of_nodes()}")
print(f"  Edges : {G_final.number_of_edges()}")
print(f"  Density : {density_f:.4f}  "
      f"({G_final.number_of_edges()} / {G_final.number_of_nodes()*(G_final.number_of_nodes()-1)//2} possible)")

print(f"\n  Eigenvector Centrality (sorted):")
print(f"  {'Ticker':<8} {'Centrality':>12}  {'Sector'}")
print(f"  {'-'*40}")
sectors = {
    "JPM":"Bank","BAC":"Bank","GS":"Bank","MS":"Bank","C":"Bank",
    "XOM":"Energy","CVX":"Energy",
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech",
}
for ticker, val in sorted(ec_final.items(), key=lambda x: -x[1]):
    print(f"  {ticker:<8} {val:>12.6f}  {sectors.get(ticker,'')}")

# Peak density period
peak_date = density_s.idxmax()
G_peak    = build_graph(roll_corr[peak_date], CORR_THRESH)
try:
    ec_peak = eigenvector_centrality_robust(G_peak)
except Exception:
    ec_peak = {t: 0.0 for t in TICKERS}

print(f"\n  Peak density window: {peak_date.date()}  density={density_s[peak_date]:.4f}")
print(f"  Most central node at peak: "
      f"{max(ec_peak, key=ec_peak.get)} "
      f"(centrality={max(ec_peak.values()):.4f})")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# Plot — 3-panel: density, centrality heatmap, final network graph
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots ...")

fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

ax_dens  = fig.add_subplot(gs[0, 0])
ax_cent  = fig.add_subplot(gs[0, 1])
ax_net   = fig.add_subplot(gs[1, :])

for ax in [ax_dens, ax_cent, ax_net]:
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# ── Panel A: Network density over time ───────────────────────────────────────
ax_dens.plot(density_s.index, density_s.values, color="#3498db", linewidth=1.2)
ax_dens.fill_between(density_s.index, density_s.values, alpha=0.15, color="#3498db")
ax_dens.axvline(pd.Timestamp("2020-03-16"), color="#e74c3c",
                linewidth=1.5, linestyle="--", label="COVID crash")
ax_dens.set_title("A — Rolling 60d Network Density", fontsize=11)
ax_dens.set_ylabel("Density", color="white")
ax_dens.set_xlabel("Date", color="white")
ax_dens.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── Panel B: Eigenvector centrality heatmap ───────────────────────────────────
cent_plot = centrality_df[TICKERS]
im = ax_cent.imshow(cent_plot.T.values, aspect="auto", cmap="plasma",
                    extent=[0, len(cent_plot), 0, len(TICKERS)])
ax_cent.set_yticks(np.arange(len(TICKERS)) + 0.5)
ax_cent.set_yticklabels(TICKERS[::-1], fontsize=8, color="white")
step   = max(1, len(cent_plot) // 5)
xticks = range(0, len(cent_plot), step)
xlbls  = [cent_plot.index[i].strftime("%Y-%m") for i in xticks]
ax_cent.set_xticks(xticks)
ax_cent.set_xticklabels(xlbls, rotation=30, fontsize=7, color="white")
ax_cent.set_title("B — Eigenvector Centrality Over Time", fontsize=11)
plt.colorbar(im, ax=ax_cent, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color="white")

# ── Panel C: Final network graph ─────────────────────────────────────────────
pos          = nx.spring_layout(G_final, seed=42, weight="weight")
edge_weights = [G_final[u][v]["weight"] for u, v in G_final.edges()]
node_ec_vals = [ec_final.get(t, 0) for t in G_final.nodes()]
max_ec       = max(node_ec_vals) if node_ec_vals else 1.0
node_colors  = [cm.plasma(v / (max_ec + 1e-9)) for v in node_ec_vals]
node_sizes   = [300 + 2500 * (ec_final.get(t, 0) / (max_ec + 1e-9))
                for t in G_final.nodes()]

nx.draw_networkx_edges(G_final, pos, ax=ax_net,
                       width=[w * 3 for w in edge_weights],
                       edge_color=edge_weights, edge_cmap=plt.cm.Blues,
                       alpha=0.6)
nx.draw_networkx_nodes(G_final, pos, ax=ax_net,
                       node_color=node_colors, node_size=node_sizes, alpha=0.95)
nx.draw_networkx_labels(G_final, pos, ax=ax_net,
                        font_color="white", font_size=9, font_weight="bold")
ax_net.set_title(
    f"C — Final Network Snapshot  ({last_date.date()})  "
    f"Density={density_f:.3f}   |corr| threshold={CORR_THRESH}",
    fontsize=11
)
ax_net.axis("off")

fig.suptitle("Financial Asset Network — Rolling 60-day Correlation",
             color="white", fontsize=14, y=0.97)

out_path = "models/network/network_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"  Saved → {out_path}")
