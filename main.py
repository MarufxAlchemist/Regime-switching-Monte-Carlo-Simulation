"""
Systemic Risk Engine — Main Orchestrator
=========================================
Research-grade pipeline for Indian equity sector risk analysis

Pipeline order:
  1. Load sector return data
  2. Detect current market regime (HMM)
  3. Compute latest correlation network (density, centrality)
  4. Score news sentiment (FinBERT)
  5. Adjust GBM drift and volatility with sentiment
  6. Run contagion-adjusted Monte Carlo simulation
  7. Compute VaR, Expected Shortfall, systemic crash probability

Usage:
    python main.py

Output:
    Structured dict printed to stdout + log file.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import networkx as nx
import yfinance as yf
from hmmlearn import hmm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("systemic_risk.log", mode="w"),
    ],
)
log = logging.getLogger("systemic_risk")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (single source of truth — no hardcoded strings elsewhere)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG: dict[str, Any] = {
    # --- Data ---
    "tickers": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
        "AXISBANK.NS",  "SBIN.NS",     "ONGC.NS",      "LT.NS",  "BAJFINANCE.NS",
    ],
    "sectors": [
        "Energy", "Bank", "Bank", "Tech",   "Tech",
        "Bank",   "Bank", "Energy","Infra", "Finance",
    ],
    "start_date":   "2021-01-01",
    "end_date":     "2024-12-31",

    # --- HMM ---
    "hmm_n_states":  3,
    "hmm_n_iter":    300,
    "hmm_corr_win":  60,     # days for HMM initialisation correlation

    # --- Network ---
    "corr_window":   60,
    "corr_threshold": 0.50,

    # --- Sentiment ---
    "finbert_model":  "ProsusAI/finbert",
    "sentiment_alpha": 0.05,   # drift sensitivity
    "sentiment_beta":  0.10,   # vol sensitivity

    # --- Monte Carlo ---
    "mc_n_paths":  5_000,
    "mc_n_steps":  252,        # 1 trading year
    "mc_seed":     42,

    # --- Contagion ---
    "contagion_alpha":   0.40,
    "contagion_theta_z": -1.50,
    "contagion_beta":    0.05,
    "contagion_vol_cap": 5.0,

    # --- Risk ---
    "var_confidence":     0.95,
    "crash_threshold":   -0.10,   # sector drops >10% in simulation horizon
    "crash_n_sectors":    3,      # systemic = N sectors crash simultaneously

    # --- GPU ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Download OHLCV data and compute log returns.

    Returns
    -------
    prices  : DataFrame  (T × N)  daily adjusted close
    returns : DataFrame  (T-1 × N)  daily log returns
    S0      : ndarray    (N,)  latest prices  (simulation starting point)
    """
    log.info("Stage 1 — Loading sector return data (%s → %s)",
             cfg["start_date"], cfg["end_date"])
    raw = yf.download(
        cfg["tickers"],
        start=cfg["start_date"],
        end=cfg["end_date"],
        auto_adjust=True,
        progress=False,
    )["Close"].dropna()

    if raw.empty:
        raise RuntimeError("No price data returned. Check tickers and date range.")

    returns = np.log(raw / raw.shift(1)).dropna()
    S0      = raw.iloc[-1].values

    log.info("  Loaded %d assets × %d trading days", raw.shape[1], raw.shape[0])
    return raw, returns, S0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — HMM Regime Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_regime(returns: pd.DataFrame, cfg: dict) -> dict:
    """
    Fit 3-state Gaussian HMM on daily returns and decode the current regime.
    Uses quantile-based mean initialisation and a persistent transition prior.

    Returns
    -------
    {
      "current_regime_id"   : int   (0/1/2)
      "current_regime_name" : str   (Bull/Bear/Crisis)
      "regime_sequence"     : ndarray  state per day
      "transition_matrix"   : ndarray  (3×3)
      "regime_means"        : ndarray  (3×N)  ann. mean returns per regime
      "regime_vols"         : ndarray  (3×N)  ann. volatilities per regime
    }
    """
    log.info("Stage 2 — Detecting current regime via 3-state Gaussian HMM")
    obs = returns.values                     # (T, N)
    N   = cfg["hmm_n_states"]
    n   = obs.shape[1]

    # ── Quantile-based mean seed (Bull = top 20%, Crisis = bottom 20%) ────────
    scalar    = obs.mean(axis=1)
    q20, q80  = np.percentile(scalar, 20), np.percentile(scalar, 80)
    mu_bull   = obs[scalar >= q80].mean(axis=0)
    mu_crisis = obs[scalar <= q20].mean(axis=0)
    mu_bear   = obs[(scalar > q20) & (scalar < q80)].mean(axis=0)

    model = hmm.GaussianHMM(
        n_components=N,
        covariance_type="full",
        n_iter=cfg["hmm_n_iter"],
        tol=1e-6,
        random_state=42,
        init_params="c",
        params="stmc",
    )
    model.means_    = np.stack([mu_bull, mu_bear, mu_crisis])
    model.transmat_ = np.array([[0.92, 0.06, 0.02],
                                 [0.10, 0.85, 0.05],
                                 [0.05, 0.20, 0.75]])
    model.startprob_ = np.array([0.60, 0.30, 0.10])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(obs)

    _, raw_states = model.decode(obs, algorithm="viterbi")

    # Align state labels: highest avg mean → Bull, lowest → Crisis
    mu_avg  = model.means_.mean(axis=1)
    perm    = np.argsort(-mu_avg)           # perm[0]=Bull, perm[2]=Crisis
    inv_map = np.empty(N, dtype=int)
    for k in range(N):
        inv_map[perm[k]] = k
    states = inv_map[raw_states]

    current = int(states[-1])
    A_aligned = model.transmat_[np.ix_(perm, perm)]
    mu_aligned = model.means_[perm] * 252          # annualised
    cov_aligned = model.covars_[perm]
    vol_aligned = np.sqrt(np.array([np.diag(cov_aligned[k]) for k in range(N)])) * np.sqrt(252)

    log.info("  Current regime : %s  (state %d)", REGIME_NAMES[current], current)
    log.info("  Regime distribution: Bull=%.1f%%  Bear=%.1f%%  Crisis=%.1f%%",
             100 * (states == 0).mean(),
             100 * (states == 1).mean(),
             100 * (states == 2).mean())

    return {
        "current_regime_id"   : current,
        "current_regime_name" : REGIME_NAMES[current],
        "regime_sequence"     : states,
        "transition_matrix"   : A_aligned,
        "regime_means"        : mu_aligned,
        "regime_vols"         : vol_aligned,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Correlation Network
# ─────────────────────────────────────────────────────────────────────────────
def build_network(returns: pd.DataFrame, cfg: dict) -> dict:
    """
    Compute rolling correlation network on the last `corr_window` days.

    Returns
    -------
    {
      "adjacency_matrix"     : ndarray  (N×N)
      "network_density"      : float
      "eigenvector_centrality": dict {ticker: centrality}
      "graph"                : nx.Graph
    }
    """
    log.info("Stage 3 — Building correlation network  (window=%dd, threshold=%.2f)",
             cfg["corr_window"], cfg["corr_threshold"])

    window_rets = returns.tail(cfg["corr_window"])
    corr        = window_rets.corr()
    tickers     = list(corr.columns)
    N           = len(tickers)

    G = nx.Graph()
    G.add_nodes_from(tickers)
    A = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            w = abs(corr.iloc[i, j])
            if w > cfg["corr_threshold"]:
                G.add_edge(tickers[i], tickers[j], weight=w)
                A[i, j] = A[j, i] = w

    density = nx.density(G)

    # Eigenvector centrality — per-component to handle disconnected graph
    ec = {t: 0.0 for t in tickers}
    for comp in nx.connected_components(G):
        sub   = G.subgraph(comp).copy()
        n_sub = len(sub)
        if n_sub < 2:
            continue
        scale = n_sub / N
        try:
            sub_ec = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except Exception:
            deg    = dict(sub.degree(weight="weight"))
            total  = sum(deg.values()) or 1.0
            sub_ec = {nd: v / total for nd, v in deg.items()}
        for node, val in sub_ec.items():
            ec[node] = val * scale

    log.info("  Network: %d edges, density=%.4f", G.number_of_edges(), density)
    log.info("  Top centrality: %s",
             sorted(ec.items(), key=lambda x: -x[1])[:3])

    return {
        "adjacency_matrix"      : A,
        "network_density"       : density,
        "eigenvector_centrality": ec,
        "graph"                 : G,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Sentiment Scoring (FinBERT)
# ─────────────────────────────────────────────────────────────────────────────
def compute_sentiment(headlines: list[str], cfg: dict) -> dict:
    """
    Score a list of financial headlines with FinBERT.
    Returns aggregate sentiment score S = mean(P_pos − P_neg).

    If no headlines provided or model unavailable, returns S = 0 (neutral).

    Returns
    -------
    {
      "aggregate_score" : float  ∈ (−1, +1)
      "per_headline"    : list[dict]
      "tone"            : str   "bullish" | "bearish" | "neutral"
    }
    """
    log.info("Stage 4 — Computing FinBERT sentiment  (%d headlines)", len(headlines))

    if not headlines:
        log.warning("  No headlines provided — sentiment defaulting to 0.0 (neutral)")
        return {"aggregate_score": 0.0, "per_headline": [], "tone": "neutral"}

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(cfg["finbert_model"])
        fb_model  = AutoModelForSequenceClassification.from_pretrained(
            cfg["finbert_model"]
        ).to(cfg["device"]).eval()

        IDX_POS, IDX_NEG = 0, 1
        results = []
        for text in headlines:
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(cfg["device"]) for k, v in inputs.items()}
            with torch.no_grad():
                logits = fb_model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1)[0]
            P_pos  = probs[IDX_POS].item()
            P_neg  = probs[IDX_NEG].item()
            score  = float(np.clip(P_pos - P_neg, -1.0, 1.0))
            label  = ["positive", "negative", "neutral"][probs.argmax().item()]
            results.append({"text": text, "score": score, "label": label,
                            "P_pos": P_pos, "P_neg": P_neg})

        agg   = float(np.mean([r["score"] for r in results]))
        tone  = ("bullish" if agg > 0.1 else "bearish" if agg < -0.1 else "neutral")
        log.info("  Aggregate sentiment score : %.4f  (%s)", agg, tone)
        return {"aggregate_score": agg, "per_headline": results, "tone": tone}

    except Exception as exc:
        log.warning("  FinBERT error (%s) — defaulting sentiment to 0.0", exc)
        return {"aggregate_score": 0.0, "per_headline": [], "tone": "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Sentiment-Adjusted GBM Parameters
# ─────────────────────────────────────────────────────────────────────────────
def adjust_parameters(
    returns: pd.DataFrame,
    sentiment_score: float,
    regime_result: dict,
    cfg: dict,
) -> dict:
    """
    Merge baseline GBM params with HMM regime and sentiment adjustments.

      μ_adj = μ_baseline + α_sentiment × S
      σ_adj = σ_baseline × (1 + β_sentiment × |S|)

    If current regime is Crisis, additionally scale vol by 1.5 as a stress buffer.

    Returns dict with daily-scale mu and sigma arrays (N,).
    """
    log.info("Stage 5 — Adjusting GBM parameters  (sentiment %.4f, regime '%s')",
             sentiment_score, regime_result["current_regime_name"])

    mu_ann  = returns.mean().values * 252
    sig_ann = returns.std().values  * np.sqrt(252)
    S       = float(np.clip(sentiment_score, -1.0, 1.0))

    # Sentiment adjustment
    alpha = cfg["sentiment_alpha"]
    beta  = cfg["sentiment_beta"]
    mu_adj  = mu_ann  + alpha * S
    sig_adj = sig_ann * (1.0 + beta * abs(S))

    # Regime stress buffer (Crisis → extra vol)
    regime = regime_result["current_regime_id"]
    regime_vol_scale = {0: 1.0, 1: 1.20, 2: 1.50}[regime]
    sig_adj *= regime_vol_scale
    if regime_vol_scale > 1.0:
        log.info("  Regime stress buffer applied: vol scaled ×%.2f", regime_vol_scale)

    log.info("  μ range  (ann): [%.2f%%, %.2f%%]",
             mu_adj.min() * 100, mu_adj.max() * 100)
    log.info("  σ range  (ann): [%.2f%%, %.2f%%]",
             sig_adj.min() * 100, sig_adj.max() * 100)

    return {
        "mu_daily"    : mu_adj  / 252,
        "sigma_daily" : sig_adj / np.sqrt(252),
        "mu_annual"   : mu_adj,
        "sigma_annual": sig_adj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Contagion-Adjusted Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(
    S0: np.ndarray,
    params: dict,
    network_result: dict,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run contagion-amplified GBM Monte Carlo.
    Inline implementation (does NOT call contagion.py as a subprocess).

    Returns
    -------
    paths       : (N_paths, N_steps+1, N_assets)  price paths
    vol_trace   : (N_paths, N_steps+1, N_assets)  realised vol per step
    event_counts: (N_steps, N_assets)             contagion events per step
    """
    log.info("Stage 6 — Running contagion Monte Carlo  (%d paths × %d steps)",
             cfg["mc_n_paths"], cfg["mc_n_steps"])
    t0 = time.time()

    mu    = params["mu_daily"]
    sig0  = params["sigma_daily"]
    A     = network_result["adjacency_matrix"]
    N_P   = cfg["mc_n_paths"]
    N_S   = cfg["mc_n_steps"]
    dt    = 1.0 / 252
    alpha = cfg["contagion_alpha"]
    theta = cfg["contagion_theta_z"]
    beta  = cfg["contagion_beta"]
    cap   = cfg["contagion_vol_cap"]
    N     = len(S0)

    rng          = np.random.default_rng(cfg["mc_seed"])
    paths        = np.empty((N_P, N_S + 1, N))
    vol_trace    = np.empty((N_P, N_S + 1, N))
    event_counts = np.zeros((N_S, N), dtype=int)

    paths[:, 0, :]  = S0
    sigma           = np.tile(sig0, (N_P, 1))
    vol_trace[:, 0, :] = sigma

    for t in range(N_S):
        # GBM step
        drift = (mu - 0.5 * sigma**2) * dt
        eps   = rng.standard_normal((N_P, N))
        paths[:, t + 1, :] = paths[:, t, :] * np.exp(
            drift + sigma * np.sqrt(dt) * eps
        )
        # Detect z-score crashes
        log_ret  = np.log(paths[:, t + 1, :] / (paths[:, t, :] + 1e-12))
        z_score  = log_ret / (sigma * np.sqrt(dt) + 1e-12)
        stressed = z_score < theta
        event_counts[t] = stressed.sum(axis=0)

        # Contagion propagation  σ_j *= (1 + α·A_ij)
        for i in range(N):
            col       = A[i, :]
            if col.max() == 0:
                continue
            stressed_i = stressed[:, i]
            if not stressed_i.any():
                continue
            amp = 1.0 + alpha * col
            sigma[stressed_i] *= amp

        # Mean-reversion + hard cap
        sigma = sigma * (1 - beta) + sig0 * beta
        sigma = np.minimum(sigma, cap * sig0)
        vol_trace[:, t + 1, :] = sigma

    elapsed = time.time() - t0
    log.info("  Simulation complete in %.2f s", elapsed)
    log.info("  Total contagion events: %d", event_counts.sum())

    return paths, vol_trace, event_counts


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk_metrics(
    paths: np.ndarray,
    S0: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Compute VaR, Expected Shortfall, and systemic crash probability.

    Parameters
    ----------
    paths : (N_paths, N_steps+1, N_assets)  — price paths from simulation
    S0    : (N_assets,)                      — initial prices

    Returns
    -------
    {
      "var_95"                    : float  (portfolio-level, annualised)
      "expected_shortfall"        : float
      "systemic_crash_probability": float
      "per_asset_var"             : ndarray  (N,)
      "per_asset_es"              : ndarray  (N,)
    }
    """
    log.info("Stage 7 — Computing risk metrics  (VaR %.0f%%, crash_n=%d, crash_thresh=%.0f%%)",
             cfg["var_confidence"] * 100,
             cfg["crash_n_sectors"],
             cfg["crash_threshold"] * 100)

    q = 1.0 - cfg["var_confidence"]

    # Terminal log returns per path × asset
    terminal_ret = np.log(paths[:, -1, :] / S0)          # (N_paths, N_assets)

    # ── Portfolio VaR and ES  (equal-weight portfolio) ────────────────────────
    port_ret = terminal_ret.mean(axis=1)                  # (N_paths,)
    var_95   = float(np.percentile(port_ret, q * 100))
    es_mask  = port_ret <= var_95
    es       = float(port_ret[es_mask].mean()) if es_mask.any() else var_95

    # ── Per-asset VaR and ES ─────────────────────────────────────────────────
    per_var = np.percentile(terminal_ret, q * 100, axis=0)   # (N,)
    per_es  = np.array([
        terminal_ret[terminal_ret[:, k] <= per_var[k], k].mean()
        for k in range(terminal_ret.shape[1])
    ])

    # ── Systemic crash probability ────────────────────────────────────────────
    # P(>= crash_n_sectors drop > crash_threshold in terminal period)
    crashed      = terminal_ret < cfg["crash_threshold"]          # (N_paths, N)
    n_crashed    = crashed.sum(axis=1)                             # (N_paths,)
    systemic     = (n_crashed >= cfg["crash_n_sectors"]).mean()

    log.info("  Portfolio VaR  (95%%): %.4f  (%.2f%%)", var_95, var_95 * 100)
    log.info("  Expected Shortfall  : %.4f  (%.2f%%)", es,     es     * 100)
    log.info("  Systemic crash prob : %.4f  (%.2f%%)", systemic, systemic * 100)

    return {
        "var_95"                     : var_95,
        "expected_shortfall"         : es,
        "systemic_crash_probability" : systemic,
        "per_asset_var"              : per_var,
        "per_asset_es"               : per_es,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main(
    headlines: list[str] | None = None,
    cfg: dict | None = None,
) -> dict:
    """
    Full systemic risk pipeline.

    Parameters
    ----------
    headlines : list of financial news headlines for sentiment scoring.
                Pass [] or None to skip sentiment (defaults to neutral S=0).
    cfg       : override any keys in CONFIG.

    Returns
    -------
    Structured output dict (see module docstring).
    """
    run_cfg = {**CONFIG, **(cfg or {})}
    if headlines is None:
        headlines = []

    log.info("=" * 70)
    log.info("  SYSTEMIC RISK ENGINE  —  device: %s", run_cfg["device"])
    log.info("=" * 70)
    t_start = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    prices, returns, S0 = load_data(run_cfg)

    # ── 2. Regime ─────────────────────────────────────────────────────────────
    regime_result = detect_regime(returns, run_cfg)

    # ── 3. Network ────────────────────────────────────────────────────────────
    network_result = build_network(returns, run_cfg)

    # ── 4. Sentiment ──────────────────────────────────────────────────────────
    sentiment_result = compute_sentiment(headlines, run_cfg)

    # ── 5. Parameter adjustment ───────────────────────────────────────────────
    params = adjust_parameters(
        returns,
        sentiment_result["aggregate_score"],
        regime_result,
        run_cfg,
    )

    # ── 6. Monte Carlo ────────────────────────────────────────────────────────
    paths, vol_trace, event_counts = run_simulation(S0, params, network_result, run_cfg)

    # ── 7. Risk metrics ───────────────────────────────────────────────────────
    risk = compute_risk_metrics(paths, S0, run_cfg)

    t_end = time.time()
    log.info("=" * 70)
    log.info("  Pipeline complete in %.2f s", t_end - t_start)

    # ── Structured output ─────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "regime"                    : regime_result["current_regime_name"],
        "sentiment_score"           : round(sentiment_result["aggregate_score"], 6),
        "sentiment_tone"            : sentiment_result["tone"],
        "var_95"                    : round(risk["var_95"],                    6),
        "expected_shortfall"        : round(risk["expected_shortfall"],        6),
        "systemic_crash_probability": round(risk["systemic_crash_probability"],6),
        "network_density"           : round(network_result["network_density"], 6),
        # Detail payloads (for downstream use / serialisation)
        "_detail": {
            "regime_transition_matrix"  : regime_result["transition_matrix"].tolist(),
            "regime_means_annual"       : regime_result["regime_means"].tolist(),
            "eigenvector_centrality"    : network_result["eigenvector_centrality"],
            "per_asset_var_95"          : risk["per_asset_var"].tolist(),
            "per_asset_es"              : risk["per_asset_es"].tolist(),
            "contagion_events_total"    : int(event_counts.sum()),
            "mu_annual"                 : params["mu_annual"].tolist(),
            "sigma_annual"              : params["sigma_annual"].tolist(),
        },
    }

    # ── Print structured summary ──────────────────────────────────────────────
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  SYSTEMIC RISK ENGINE — OUTPUT SUMMARY")
    print(SEP)
    print(f"  Regime                    : {output['regime']}")
    print(f"  Sentiment score           : {output['sentiment_score']:>+.4f}  "
          f"({output['sentiment_tone']})")
    print(f"  Network density           : {output['network_density']:.4f}")
    print(f"  VaR 95%  (portfolio)      : {output['var_95']:>+.4f}  "
          f"({output['var_95']*100:.2f}%)")
    print(f"  Expected shortfall        : {output['expected_shortfall']:>+.4f}  "
          f"({output['expected_shortfall']*100:.2f}%)")
    print(f"  Systemic crash prob       : {output['systemic_crash_probability']:.4f}  "
          f"({output['systemic_crash_probability']*100:.2f}%)")
    print(f"  Contagion events (total)  : "
          f"{output['_detail']['contagion_events_total']:,}")
    print(f"\n  Per-asset VaR 95%:")
    for i, t in enumerate(run_cfg["tickers"]):
        print(f"    {t:<18} {risk['per_asset_var'][i]:>+.4f}  "
              f"({risk['per_asset_var'][i]*100:.2f}%)")
    print(SEP)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_HEADLINES = [
        "RBI holds repo rate steady amid persistent inflation concerns.",
        "Nifty 50 falls 2% as FII outflows accelerate on global recession fears.",
        "HDFC Bank reports strong Q3 profit, beats Street estimates.",
        "India GDP growth slows to 6.1% in Q2, below consensus forecast.",
        "Reliance Industries announces $10B green energy capex expansion.",
    ]

    try:
        result = main(headlines=SAMPLE_HEADLINES)
    except Exception as err:
        log.exception("Pipeline failed: %s", err)
        sys.exit(1)
