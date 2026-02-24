"""
Density Validation — Network density vs time with crisis annotations.
Checks that density spikes during known crisis windows.
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
from hmmlearn import hmm

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["JPM","BAC","GS","MS","C","XOM","CVX","AAPL","MSFT","NVDA"]
START, END   = "2015-01-01", "2024-12-31"   # extended to capture GFC aftermath & COVID
WINDOW       = 60
CORR_THRESH  = 0.60

CRISIS_WINDOWS = [
    ("2015-08-18", "2015-09-10",  "China\nSelloff"),
    ("2018-10-01", "2018-12-28",  "2018 Q4\nSelloff"),
    ("2020-02-20", "2020-05-01",  "COVID\nCrash"),
    ("2022-01-01", "2022-10-15",  "Rate Hike\nBear"),
]

# ── Download & returns ────────────────────────────────────────────────────────
print("Downloading data ...")
raw  = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"].dropna()
rets = np.log(raw / raw.shift(1)).dropna()
T    = len(rets)
dates_all = rets.index
print(f"  {T} days")

# ── Rolling 60d correlation → graph → density ─────────────────────────────────
print("Computing rolling density ...")
density_ts = {}
for i, date in enumerate(dates_all[WINDOW - 1:]):
    w_rets = rets.iloc[i : i + WINDOW]
    corr   = w_rets.corr()
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    n = len(corr)
    for a in range(n):
        for b in range(a + 1, n):
            if abs(corr.iloc[a, b]) > CORR_THRESH:
                G.add_edge(corr.columns[a], corr.columns[b], weight=abs(corr.iloc[a, b]))
    density_ts[date] = nx.density(G)

density = pd.Series(density_ts)
print(f"  Done. Mean={density.mean():.4f}  Max={density.max():.4f}")

# ── Quantitative crisis spike check ──────────────────────────────────────────
print("\n── Crisis spike validation ──────────────────────────────────────────")
baseline = density.mean()
print(f"  Baseline mean density (full period): {baseline:.4f}\n")
print(f"  {'Crisis':<22} {'Avg Density':>12}  {'vs Baseline':>12}  {'Spike?':>8}")
print(f"  {'-'*60}")
for s, e, label in CRISIS_WINDOWS:
    mask   = (density.index >= s) & (density.index <= e)
    if mask.sum() == 0:
        continue
    avg    = density[mask].mean()
    pct    = (avg / baseline - 1) * 100
    spike  = "✓ YES" if avg > baseline * 1.05 else "✗ NO"
    name   = label.replace("\n", " ")
    print(f"  {name:<22} {avg:>12.4f}  {pct:>+11.1f}%  {spike:>8}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6), facecolor="#0f1117")
ax.set_facecolor("#0f1117")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

# Density area + line
ax.fill_between(density.index, density.values, alpha=0.20, color="#3498db")
ax.plot(density.index, density.values, color="#3498db", linewidth=1.2, label="Network Density")

# Baseline
ax.axhline(baseline, color="#95a5a6", linewidth=1.0, linestyle="--", label=f"Mean ({baseline:.3f})")

# Crisis windows
colors_c = ["#e74c3c", "#e67e22", "#e74c3c", "#9b59b6"]
for (s, e, label), col in zip(CRISIS_WINDOWS, colors_c):
    s_ts = pd.Timestamp(s)
    e_ts = pd.Timestamp(e)
    if s_ts < density.index[0]:
        continue
    ax.axvspan(s_ts, e_ts, alpha=0.18, color=col)
    ax.text(
        s_ts + (e_ts - s_ts) / 2,
        density.max() * 0.97,
        label, color=col, fontsize=8,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f1117", alpha=0.7),
    )

# Rolling 30d mean overlay
smooth = density.rolling(30).mean()
ax.plot(density.index, smooth.values, color="#f39c12", linewidth=1.5,
        linestyle="-", alpha=0.8, label="30d rolling avg")

ax.set_title("Network Density vs Time  |  Rolling 60-day Correlation  (|corr| > 0.60 threshold)",
             color="white", fontsize=13, pad=10)
ax.set_ylabel("Network Density", color="white", fontsize=11)
ax.set_xlabel("Date", color="white", fontsize=11)
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9, loc="upper left")
ax.set_ylim(0, density.max() * 1.12)
ax.set_xlim(density.index[0], density.index[-1])

plt.tight_layout()
out = "models/network/density_validation.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"\nPlot saved → {out}")
