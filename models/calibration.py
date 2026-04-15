"""
Sensitivity Calibration — Real Pipeline Integration
=====================================================
Runs a parameter-grid sweep over (alpha, beta, window) using the
actual systemic-risk pipeline stages from main.py.

For each grid point a lightweight expanding-window backtest is performed
and the following metrics are recorded per trial:
  - VaR violation rate, average VaR, average realised loss
  - Mean systemic crash probability
  - Kupiec POF test and Christoffersen conditional-coverage test

Output:  data/calibration_results.csv
"""

from __future__ import annotations

import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    CONFIG,
    load_data,
    detect_regime,
    build_network,
    adjust_parameters,
    run_simulation,
    compute_risk_metrics,
)
import validation_tests as vt

log = logging.getLogger("calibration")

# Configuration

# Parameter grid
ALPHAS  = [0.10, 0.40, 0.70]           # contagion strength
BETAS   = [0.00, 0.10, 0.30]           # sentiment amplification
WINDOWS = [30, 60, 90]                 # correlation window (days)

# Backtest settings (reduced for feasibility)
MIN_TRAIN_DAYS     = 252               # ~1 year burn-in
STEP_SIZE          = 20                # advance 20 days between origins
FORECAST_HORIZON   = 30               # 30-day forward evaluation
MC_N_PATHS         = 2000             # paths per MC simulation
MC_N_STEPS         = 30               # steps per path
HMM_N_ITER         = 100              # reduced HMM iterations for speed


# Single-origin pipeline evaluation

def evaluate_origin(
    returns: pd.DataFrame,
    origin_idx: int,
    horizon: int,
    run_cfg: dict,
) -> dict:
    """
    Run the full pipeline on returns[:origin_idx] and compare the
    predicted VaR against the realised forward return.

    Returns a dict with per-origin metrics.
    """
    train = returns.iloc[:origin_idx]
    forward = returns.iloc[origin_idx : origin_idx + horizon]

    regime_result  = detect_regime(train, run_cfg)
    network_result = build_network(train, run_cfg)

    # Sentiment fixed to neutral in backtest context
    params = adjust_parameters(train, 0.0, regime_result, run_cfg)

    # Price-level proxy for MC starting point
    S0_proxy = np.exp(train.cumsum().iloc[-1].values)

    paths, _, _ = run_simulation(S0_proxy, params, network_result, run_cfg)
    risk = compute_risk_metrics(paths, S0_proxy, run_cfg)

    daily_portfolio = forward.mean(axis=1)
    realized_loss   = float(daily_portfolio.sum())

    predicted_var = risk["var_95"]
    violation     = realized_loss < predicted_var

    return {
        "origin_date":       train.index[-1],
        "predicted_var_95":  predicted_var,
        "realized_loss":     realized_loss,
        "violation":         violation,
        "crash_prob":        risk["systemic_crash_probability"],
        "expected_shortfall": risk["expected_shortfall"],
        "regime":            regime_result["current_regime_name"],
    }


# Rolling backtest for one parameter combination

def run_trial(
    returns: pd.DataFrame,
    alpha: float,
    beta: float,
    window: int,
) -> dict:
    """
    Run expanding-window backtest for a single (alpha, beta, window)
    combination.  Returns aggregated trial-level metrics.
    """
    T = len(returns)

    run_cfg = {
        **CONFIG,
        "contagion_alpha": alpha,
        "sentiment_beta":  beta,
        "corr_window":     window,
        "hmm_corr_win":    window,
        "mc_n_paths":      MC_N_PATHS,
        "mc_n_steps":      MC_N_STEPS,
        "hmm_n_iter":      HMM_N_ITER,
        "crash_threshold": -0.05,       # relaxed from -0.10 for detectability
    }

    first_origin = MIN_TRAIN_DAYS
    last_origin  = T - FORECAST_HORIZON
    if first_origin > last_origin:
        raise ValueError(
            f"Not enough data: T={T}, min_train={MIN_TRAIN_DAYS}, "
            f"horizon={FORECAST_HORIZON}"
        )
    origins = list(range(first_origin, last_origin + 1, STEP_SIZE))

    window_results = []
    for i, t in enumerate(origins):
        try:
            result = evaluate_origin(returns, t, FORECAST_HORIZON, run_cfg)
            window_results.append(result)
        except Exception as exc:
            log.warning("  Origin %d failed: %s", t, exc)
            continue

        if (i + 1) % max(1, len(origins) // 5) == 0:
            vr = sum(r["violation"] for r in window_results) / len(window_results)
            log.info(
                "    [%d/%d] origin=%s  VaR=%.4f  loss=%.4f  cum_viol=%.1f%%",
                i + 1, len(origins),
                result["origin_date"].strftime("%Y-%m-%d"),
                result["predicted_var_95"],
                result["realized_loss"],
                vr * 100,
            )

    if not window_results:
        return {
            "alpha": alpha, "beta": beta, "window_size": window,
            "error": "All origins failed",
        }

    violations     = np.array([int(r["violation"]) for r in window_results])
    vars_          = np.array([r["predicted_var_95"] for r in window_results])
    losses         = np.array([r["realized_loss"] for r in window_results])
    crash_probs    = np.array([r["crash_prob"] for r in window_results])
    n_origins      = len(window_results)
    n_violations   = int(violations.sum())
    violation_rate = n_violations / n_origins

    kupiec = vt.kupiec_pof_test(violations, alpha=0.05)
    cc     = vt.christoffersen_conditional_coverage_test(violations, alpha=0.05)

    return {
        "alpha":              alpha,
        "beta":               beta,
        "window_size":        window,
        "total_origins":      n_origins,
        "violation_count":    n_violations,
        "violation_rate":     round(violation_rate, 6),
        "avg_var":            round(float(vars_.mean()), 6),
        "avg_loss":           round(float(losses.mean()), 6),
        "mean_crash_prob":    round(float(crash_probs.mean()), 6),
        "kupiec_statistic":   kupiec["test_statistic"],
        "kupiec_pvalue":      kupiec["p_value"],
        "kupiec_reject_null": kupiec["reject_null"],
        "cc_statistic":       cc["test_statistic"],
        "cc_pvalue":          cc["p_value"],
        "cc_reject_null":     cc["reject_null"],
    }


# Main entry point

def main():
    t_start = time.time()

    print("=" * 70)
    print("  SENSITIVITY CALIBRATION — Parameter Grid Sweep")
    print("=" * 70)

    print("\n[1/3] Loading market data...")
    prices, returns, S0 = load_data(CONFIG)
    print(f"  Loaded: {returns.shape[0]} days × {returns.shape[1]} assets")

    grid  = list(product(ALPHAS, BETAS, WINDOWS))
    total = len(grid)
    print(f"\n[2/3] Running {total} trials  "
          f"({len(ALPHAS)} α × {len(BETAS)} β × {len(WINDOWS)} W)")
    print(f"  Settings: min_train={MIN_TRAIN_DAYS}, step={STEP_SIZE}, "
          f"horizon={FORECAST_HORIZON}, mc_paths={MC_N_PATHS}\n")

    all_rows = []
    for idx, (alpha, beta, window) in enumerate(grid, 1):
        print(f"── Trial {idx:>2}/{total}  "
              f"α={alpha:.2f}  β={beta:.2f}  W={window:>3d}  ", end="", flush=True)

        trial_start = time.time()
        row = run_trial(returns, alpha, beta, window)
        elapsed = time.time() - trial_start

        all_rows.append(row)

        if "error" in row:
            print(f"ERROR: {row['error']}")
        else:
            print(f"viol_rate={row['violation_rate']:.1%}  "
                  f"avg_var={row['avg_var']:+.4f}  "
                  f"crash_prob={row['mean_crash_prob']:.4f}  "
                  f"({elapsed:.0f}s)")

    output_path = PROJECT_ROOT / "data" / "calibration_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    total_time = time.time() - t_start
    print(f"\n[3/3] Results saved → {output_path}  ({len(df)} rows)")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
