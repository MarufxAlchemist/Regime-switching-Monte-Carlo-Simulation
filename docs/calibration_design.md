# Sensitivity Analysis Experiment Design
## Regime-Switching Systemic Risk Model

---

## 1. Objective

Understand how **three key parameters** shape the model's risk estimates by systematically sweeping their values across a calibrated grid and observing the effect on four target metrics recorded at each backtest window.

| Parameter | Symbol | Lives in | Mechanism |
|---|---|---|---|
| Contagion strength | `alpha` | `contagion_alpha` in CONFIG | Amplifies cross-asset volatility when a z-score crash is detected |
| Sentiment amplification | `beta` | `sentiment_beta` in CONFIG | Scales how far a sentiment score shifts realised volatility |
| Correlation window | `W` | `corr_window` + `hmm_corr_win` in CONFIG | Controls the lookback used to build the network adjacency matrix |

---

## 2. Parameter Grid.

### 2.1 Recommended Ranges & Rationale

#### `alpha` — Contagion Strength
> Controls the multiplier `σ_j *= (1 + α · A_ij)` applied to connected assets during a stress event.

| Level | Value | Interpretation |
|---|---|---|
| Minimal | 0.10 | Weak linkage; contagion barely propagates |
| Low | 0.20 | Mild spill-over |
| Baseline | 0.40 | Current production setting |
| High | 0.60 | Strong contagion (stressed markets) |
| Extreme | 0.80 | Near-GFC-level amplification |

**Grid:** `[0.10, 0.20, 0.40, 0.60, 0.80]` — 5 levels.

> [!NOTE]
> Values above 0.80 push sigma beyond the `vol_cap * sig0` ceiling on nearly every step, making the cap (not alpha) the binding constraint — avoid

---

#### `beta` — Sentiment Amplification
> Governs `σ_adj = σ_baseline × (1 + β · |S|)`. With `|S| ≤ 1`, beta is effectively a percentage vol-bump per full sentiment unit.

| Level | Value | Interpretation |
|---|---|---|
| Off | 0.00 | Sentiment has zero vol impact |
| Minimal | 0.05 | 5% vol bump at peak sentiment |
| Baseline | 0.10 | Current production setting |
| Elevated | 0.20 | Strong sentiment response |
| Extreme | 0.35 | Maximum plausible for Indian equity markets |

**Grid:** `[0.00, 0.05, 0.10, 0.20, 0.35]` — 5 levels

> [!NOTE]
> In the backtest pipeline, sentiment is set to neutral (S=0), so beta has no effect there unless you extend the backtest to feed live headlines. This axis is therefore most informative in **single-run** cross-sections, not rolling backtests.

---

#### Correlation Window `W` (days)
> Sets the lookback for both the rolling correlation network and HMM initialization

| Level | Value | Interpretation |
|---|---|---|
| Very short | 20 | ~1 month; highly reactive to recent shocks |
| Short | 40 | ~2 months |
| Baseline | 60 | Current production setting |
| Medium | 90 | ~1 quarter; smoother but lagged |
| Long | 120 | ~6 months; structurally stable |

**Grid:** `[20, 40, 60, 90, 120]` — 5 levels.

> [!CAUTION]
> Window sizes below 20 risk rank-deficient correlation matrices (you have 10 assets), causing eigenvector centrality failures. Stay at or above 20

---

### 2.2 Full Grid Summary

| Axis | Values | Count |
|---|---|---|
| alpha | 0.10, 0.20, 0.40, 0.60, 0.80 | 5 |
| beta | 0.00, 0.05, 0.10, 0.20, 0.35 | 5 |
| window | 20, 40, 60, 90, 120 | 5 |
| **Total combinations** | | **125** |

---

## 3. Target Metrics (What to Record)

For each `(alpha, beta, window)` combination, one rolling backtest is executed producing a list of `BacktestResult` records. The following four scalars are then **aggregated** across all windows in that run.

| Metric | Source | Description |
|---|---|---|
| **VaR violation rate** | `summary()["violation_rate"]` | Fraction of backtest windows where `realized_loss < predicted_var_95` |
| **Average VaR** | `summary()["mean_predicted_var"]` | Mean of `predicted_var_95` across all origins |
| **Average realized loss** | `summary()["mean_realized_loss"]` | Mean of `realized_loss` across all origins |
| **Systemic crash probability** | Extended `BacktestResult` field | Must be captured per-window and averaged; currently not stored in `BacktestResult` — see §5 |

> [!IMPORTANT]
> **Systemic crash probability** is currently returned by `compute_risk_metrics()` inside `make_pipeline_fn()` but is **not** persisted in `BacktestResult`. The pipeline factory returns only `var_95`, `expected_shortfall`, and `regime`. You will need to thread this value through when extending the experiment.

---

## 4. Data Schema for Results

### 4.1 Per-Window Record (row-level grain)

Each row represents **one forecast origin** within **one experiment trial**

```
sensitivity_results (row-level)
──────────────────────────────────────────────────────
trial_id            : str     "alpha=0.40_beta=0.10_W=60"
alpha               : float
beta                : float
window_size         : int
origin_date         : date    (from BacktestResult.origin_date)
train_size          : int     (days in expanding window)
predicted_var_95    : float
realized_loss       : float
violation           : bool
systemic_crash_prob : float   (to be added)
```

### 4.2 Trial-Level Summary (aggregated grain)

Each row represents **one (alpha, beta, window) combination** — 125 rows total

```
sensitivity_summary (trial-level)
──────────────────────────────────────────────────────
trial_id              : str
alpha                 : float
beta                  : float
window_size           : int
var_violation_rate    : float   [0, 1]
mean_predicted_var    : float   (negative; deeper = higher risk)
mean_realized_loss    : float   (negative = loss)
mean_crash_prob       : float   [0, 1]
total_origins         : int
violation_count       : int
```

### 4.3 Recommended File Layout

```
experiments/
└── sensitivity/
    ├── row_level/
    │   ├── alpha=0.10_beta=0.00_W=20.csv
    │   ├── alpha=0.10_beta=0.00_W=40.csv
    │   └── ... (125 files)
    └── sensitivity_summary.csv      ← aggregated, single file
```

> [!TIP]
> Keeping row-level files separate means a single failed trial can be rerun without re-running the entire grid.

---

## 5. Experimental Workflow

### Phase 0 — Setup

1. Load the full historical return panel once (2021–2024).
2. Decide on fixed `BacktestConfig` settings that stay constant across all trials:
   - `min_train_days = 252`
   - `step_size = 5` (advance every 5 days to reduce runtime)
   - `forecast_horizon = 30`
   - `mc_n_paths = 1_000` *(reduce from 5k to keep 125-trial runtime feasible)*

### Phase 1 — Grid Execution

```
FOR EACH alpha IN [0.10, 0.20, 0.40, 0.60, 0.80]:
  FOR EACH beta IN [0.00, 0.05, 0.10, 0.20, 0.35]:
    FOR EACH window IN [20, 40, 60, 90, 120]:

      trial_id  = f"alpha={alpha}_beta={beta}_W={window}"
      trial_cfg = merge(BASE_CONFIG, {
                    "contagion_alpha": alpha,
                    "sentiment_beta":  beta,
                    "corr_window":     window,
                    "hmm_corr_win":    window,
                  })

      backtester = RollingBacktester(
                    returns      = full_returns,
                    pipeline_fn  = make_pipeline_fn(trial_cfg),
                    config       = backtest_cfg,
                    pipeline_cfg = trial_cfg,
                  )

      per_window_results = backtester.run()
      summary            = backtester.summary()

      ── RECORD ──
      Save per_window_results → row_level/{trial_id}.csv
      Append summary row     → sensitivity_summary.csv

END LOOP
```

### Phase 2 — Aggregation

After all 125 trials complete:
1. Load `sensitivity_summary.csv`.
2. Compute derived columns: `excess_violation = var_violation_rate - 0.05` (expected at 95% VaR).
3. Rank trials by absolute deviation of violation rate from 5%.

### Phase 3 — Analysis

| Question | How to answer |
|---|---|
| Which parameter dominates VaR forecasts? | Sort `mean_predicted_var` by axis; compute variance decomposition |
| Where does the model over-conserve? | Filter `var_violation_rate < 0.03` |
| Where does the model under-cover? | Filter `var_violation_rate > 0.08` |
| Alpha vs. crash probability | Fix `beta`, `window`; plot `alpha` vs. `mean_crash_prob` |
| Window regime sensitivity | Fix `alpha`, `beta`; compare Bear vs. Bull regime frequency across windows |

---

## 6. Runtime Estimate

| Setting | Value |
|---|---|
| Trials | 125 |
| Backtest origins per trial | ~(750 − 252 − 30) / 5 ≈ **94 origins** |
| Pipeline time per origin | ~0.5 s (CPU, 1k paths) |
| **Total estimate** | **125 × 94 × 0.5 s ≈ 1.6 hours** |

> [!TIP]
> To run faster: (a) use `mc_n_paths = 500`, or (b) parallelise the outer `alpha` loop across CPU cores using `concurrent.futures.ProcessPoolExecutor`.

---

## 7. Key Design Decisions

1. **Fixed `beta` in backtest**: Since the backtest engine sets sentiment to `0.0` (neutral), beta's sensitivity surface is flat in the rolling-backtest context. If you want to test `beta`, replace neutral sentiment with a precomputed historical sentiment series.

2. **Window applies to both network and HMM**: `corr_window` and `hmm_corr_win` should always move together. Decoupling them is a potential second-order experiment.

3. **step_size = 5**: Reduces trial count from ~470 origins to ~94 without meaningfully changing the violation rate estimate — consecutive windows are highly autocorrelated anyway.

4. **VaR level held constant at 95%**: Crossing `var_confidence` into the sensitivity grid would create a 4-dimensional tensors — keep that as a separate future experiment.
