# Comparative Experimental Framework: Systemic Risk Model Evaluation

---

## 1  Scope and Objective

This document specifies the design of a controlled rolling-backtest experiment to compare three systemic risk Monte Carlo models across statistical and economic performance dimensions.

| ID | Model | Description |
|---|---|---|
| **A** | Static GBM Monte Carlo | Constant drift $\mu$ and volatility $\sigma$ estimated over the full training window; no regime or network components |
| **B** | Regime-Switching Monte Carlo | HMM-detected regime ($K=3$: Bull / Bear / Crisis) drives state-conditional $\mu_k$, $\sigma_k$; no network or sentiment |
| **C** | Regime + Network + Contagion + Sentiment | Full pipeline: HMM regime, correlation network, contagion-amplified volatility propagation, FinBERT sentiment adjustment |

The experiment answers a single research question:

> *Do the incremental components added in Models B and C produce statistically better-calibrated and more informative VaR forecasts than the baseline in A, and by how much?*

---

## 2  Experimental Controls

All three models must operate under **identical conditions** to ensure fair comparison. The following are fixed across all models:

| Control factor | Value |
|---|---|
| Asset universe | Same 10 Indian equity sector tickers |
| Return type | Daily log returns |
| Training window type | Expanding (no data from the future ever exposed) |
| Minimum training window | 252 trading days |
| Step size | 5 trading days |
| Forecast horizon | 30 trading days |
| VaR confidence level | $1 - \alpha = 0.95$ ($\alpha = 0.05$) |
| Monte Carlo paths | 5,000 per forecast origin (per model) |
| MC random seed | Fixed identical seed per origin across models (ensures same noise draws where applicable) |
| Evaluation period | Fixed held-out period after the minimum training window |
| Violation definition | $I_t = \mathbf{1}\{r_t^{\text{realised}} < -\widehat{\text{VaR}}_t^{(0.95)}\}$ |
| Equal-weight portfolio return | $r_t = \frac{1}{N}\sum_{i=1}^{N} r_{t,i}$ (cumulative over 30-day forward window) |

> [!IMPORTANT]
> The use of the **same random seed per origin** is critical: it eliminates Monte Carlo noise as a confounding factor, isolating the contribution of each model's structural component (regime detection, contagion, sentiment).

---

## 3  Data Protocol

### 3.1  Full dataset partitioning

```
|<—————————— Full sample [2021-01-01, 2024-12-31] ——————————>|
|                                                              |
| [2021-01-01, 2022-12-31]    [2023-01-01, 2024-11-30]        |
|   Burn-in (min 252 days)    Rolling evaluation period        |
```

- **Burn-in period**: Used only to satisfy the minimum training window. No performance metrics are computed here.
- **Rolling evaluation period**: Every 5 trading days, a new forecast origin is created. Performance metrics are computed over all forecast origins in this period.

### 3.2  Storage schema

Each model stores a separate result table with the following columns per origin:

| Column | Type | Description |
|---|---|---|
| `origin_date` | date | Last day of the training window |
| `train_size` | int | Number of trading days in the training window |
| `predicted_var_95` | float | Portfolio-level VaR(95%) from the MC simulation |
| `realized_loss` | float | Cumulative 30-day equal-weight portfolio return |
| `violation` | bool | $I_t = \mathbf{1}\{\text{realized\_loss} < -\text{predicted\_var\_95}\}$ |
| `regime_id` | int or None | Detected regime at origin (0/1/2; None for Model A) |
| `network_density` | float or None | Correlation network density (None for Models A, B) |
| `systemic_crash_flag` | bool | $\mathbf{1}\{$realised 30-day period contained a systemic crash event$\}$ (see §6) |
| `model_id` | str | `"A"`, `"B"`, or `"C"` |

All three result tables share the **same set of origin dates** (defined by Model A's schedule, which has no components that could fail to produce an origin).

---

## 4  Primary Evaluation Metrics

### 4.1  Violation rate

$$\hat{p}_m = \frac{1}{T}\sum_{t=1}^{T} I_t^{(m)}, \quad m \in \{A, B, C\}$$

A correctly calibrated model has $\hat{p}_m \approx \alpha = 0.05$.

**Interpretation**:
- $\hat{p}_m \gg \alpha$: Model $m$ systematically under-estimates risk.
- $\hat{p}_m \ll \alpha$: Model $m$ is overly conservative.

### 4.2  Kupiec Proportion of Failures (POF) test

For each model $m$, compute:

$$\text{LR}_{\text{POF}}^{(m)} = -2\left[n_m \ln\!\frac{\alpha}{\hat{p}_m} + (T - n_m)\ln\!\frac{1-\alpha}{1-\hat{p}_m}\right] \sim \chi^2(1) \text{ under } H_0$$

where $n_m = \sum_t I_t^{(m)}$.

Report: test statistic, $p$-value, rejection decision at 5% level.

A model that **does not reject** $H_0$ has statistically correct unconditional coverage.

### 4.3  Christoffersen Independence test

For each model $m$, extract transition counts $n_{ij}^{(m)}$ from the hit sequence and compute:

$$\text{LR}_{\text{ind}}^{(m)} \sim \chi^2(1) \text{ under } H_0$$

A model that **does not reject** $H_0$ produces violations that are serially uncorrelated — no clustering.

### 4.4  Christoffersen Conditional Coverage test

$$\text{LR}_{\text{cc}}^{(m)} = \text{LR}_{\text{POF}}^{(m)} + \text{LR}_{\text{ind}}^{(m)} \sim \chi^2(2) \text{ under } H_0$$

This is the **primary statistical test** for model validity. A model passes if it achieves both correct level and independence jointly.

---

## 5  Secondary Evaluation Metrics

### 5.1  Average VaR and VaR stability

$$\overline{\text{VaR}}_m = \frac{1}{T}\sum_{t=1}^{T} \widehat{\text{VaR}}_t^{(m)}, \qquad \sigma_{\text{VaR}}^{(m)} = \text{std}\!\left(\widehat{\text{VaR}}_t^{(m)}\right)$$

- A larger $\overline{\text{VaR}}_m$ indicates a more conservative (and more capital-intensive) model.
- Higher $\sigma_{\text{VaR}}^{(m)}$ indicates a model that is more responsive to changing market conditions — desirable if correctly calibrated; dangerous if noisy.

**Capital efficiency ratio** (penalises unnecessary conservatism):

$$\text{CER}_m = \frac{|\overline{\text{VaR}}_m|}{|\bar{r}^{\text{realised}}|} \geq 1$$

A ratio close to 1 indicates a model that reserves just enough capital.

### 5.2  Average realised loss (model-independent, sanity check)

$$\bar{r} = \frac{1}{T}\sum_{t=1}^{T} r_t^{\text{realised}}$$

This quantity is the same for all three models (since realised returns are fixed). Its primary role is to confirm that the evaluation period is representative (e.g., non-trivially profitable or loss-making).

### 5.3  Conditional violation rate by regime

For Model C (and B), break down violation rates by the regime active at the forecast origin:

$$\hat{p}_m^{(k)} = \frac{\sum_t I_t^{(m)} \cdot \mathbf{1}\{\text{regime}_t = k\}}{\sum_t \mathbf{1}\{\text{regime}_t = k\}}, \quad k \in \{0, 1, 2\}$$

**Interpretation**: A well-calibrated regime-switching model should exhibit regime-conditional violation rates all close to $\alpha = 0.05$. A static model (A) is expected to show elevated violation rates in Crisis regimes.

---

## 6  Systemic Crash Prediction Accuracy

### 6.1  Defining a systemic crash event

A 30-day forward window starting at origin $t$ is classified as a **systemic crash** if at least $N_{\min}$ assets simultaneously decline by more than $\delta$ over the window:

$$\text{CRASH}_t = \mathbf{1}\!\left\{\sum_{i=1}^{N} \mathbf{1}\!\left\{r_{t,i}^{30d} < \delta\right\} \geq N_{\min}\right\}$$

Recommended defaults (consistent with `main.py` config): $\delta = -10\%$, $N_{\min} = 3$.

This label is **model-independent** and is computed once from the raw return panel.

### 6.2  Crash prediction score

Each model implicitly predicts systemic crash risk via its **systemic crash probability** output (if available — only Model C produces this directly). For Models A and B, use a derived proxy: the probability that the simulated portfolio return falls below a crash threshold $\delta_p$ (e.g. $-15\%$ cumulative over 30 days):

$$\hat{q}_t^{(m)} = \Pr\!\left(r_t^{\text{MC}} < \delta_p\right)$$

Evaluate using **area under the ROC curve (AUROC)** treating $\text{CRASH}_t$ as the binary label and $\hat{q}_t^{(m)}$ as the predicted score:

$$\text{AUROC}_m = \int_0^1 \text{TPR}\!\left(\text{FPR}^{-1}(\theta)\right) d\theta$$

Compute via the empirical ROC curve across all origins.

**Interpretation**:
- $\text{AUROC}_m = 0.5$: Model is no better than random at predicting crashes.
- $\text{AUROC}_m > 0.7$: Useful discriminatory power.
- Compare $\text{AUROC}_A$ vs. $\text{AUROC}_B$ vs. $\text{AUROC}_C$ to quantify the marginal value of each architectural addition.

### 6.3  Brier score

$$\text{BS}_m = \frac{1}{T}\sum_{t=1}^{T}\!\left(\hat{q}_t^{(m)} - \text{CRASH}_t\right)^2$$

Lower is better. Unlike AUROC, the Brier score penalises calibration errors in the crash probability estimate, not just ranking.

---

## 7  Comparison Summary Table

At the conclusion of the experiment, populate the following table:

| Metric | Model A (Static GBM) | Model B (Regime) | Model C (Full) | Winner |
|---|---|---|---|---|
| Violation rate $\hat{p}_m$ | — | — | — | Closest to 0.05 |
| $\text{LR}_{\text{POF}}$ ($p$-value) | — | — | — | Largest $p$-value |
| POF reject $H_0$? | — | — | — | No rejection preferred |
| $\text{LR}_{\text{ind}}$ ($p$-value) | — | — | — | Largest $p$-value |
| Ind. reject $H_0$? | — | — | — | No rejection preferred |
| $\text{LR}_{\text{cc}}$ ($p$-value) | — | — | — | Largest $p$-value |
| CC reject $H_0$? | — | — | — | No rejection preferred |
| $\overline{\text{VaR}}_m$ | — | — | — | — |
| $\sigma_{\text{VaR}}^{(m)}$ | — | — | — | Higher (if calibrated) |
| Capital efficiency ratio | — | — | — | Closest to 1.0 |
| AUROC (crash prediction) | — | — | — | Highest |
| Brier score (crash) | — | — | — | Lowest |

---

## 8  Interpretation Protocol

### Step 1 — Validity gate

Before any comparative analysis, check each model against the conditional coverage test:

- **Model fails CC test** ($p < 0.05$): The model is statistically invalid. It should not be used for regulatory VaR reporting. Further comparative analysis can still proceed for research purposes but must be clearly labelled.
- **All models fail CC test**: The evaluation period may contain an extreme structural break. Segment the evaluation period by regime and repeat.

### Step 2 — Statistical comparison

Among models that pass the validity gate:

1. Compare $p$-values of $\text{LR}_{\text{POF}}$ and $\text{LR}_{\text{ind}}$ — a model with much larger $p$-values (further from rejection) has better statistical calibration.
2. Compute the **violation rate gap**: $|\hat{p}_m - \alpha|$. Smaller gap is better.
3. Note whether violations are clustered (high $\pi_{11}^{(m)}$) — this reflects regime-adaptation speed.

### Step 3 — Economic comparison

1. Assess the **capital efficiency ratio**: A model with substantially higher $\overline{\text{VaR}}_m$ but passing the CC test is over-conservative and economically inefficient.
2. Assess crash prediction via AUROC — this is the only metric that directly evaluates the value of the network + contagion + sentiment components in Model C.

### Step 4 — Attribution of improvement

Use pairwise comparison to attribute performance gains:

| Comparison | What is isolated |
|---|---|
| B vs. A | Marginal value of regime detection |
| C vs. B | Marginal value of network + contagion + sentiment |
| C vs. A | Total value of the full architecture |

A component adds value if the model including it has a statistically higher AUROC, lower $|\hat{p}_m - \alpha|$, and passes the CC test when the baseline does not.

### Step 5 — Robustness checks

- **Subsample analysis**: Repeat over Bull-only, Bear-only, and Crisis-only sub-periods to verify regime-specific performance.
- **Sensitivity to $\alpha$**: Repeat at 99% confidence ($\alpha = 0.01$) to assess tail risk calibration.
- **Alternative crash thresholds**: Vary $\delta$ and $N_{\min}$ in §6.1 to confirm AUROC robustness.

---

## 9  Known Limitations and Biases

| Limitation | Description | Mitigation |
|---|---|---|
| Model C parameter leakage | HMM and FinBERT are fitted on a training window that may still share macro conditions with the evaluation period | Ensure expanding-window fitting; never fit on the forward window |
| Overlapping forecast horizons | If step size $< H = 30$, successive forecasts share forward-return observations | Use step size $\geq 30$ for fully non-overlapping windows; note the dependence structure when using step $= 5$ |
| Short evaluation sample | With $T = 250$ origins and $\alpha = 0.05$, expected violations $\approx 12.5$; LR tests have limited power | Report exact $p$-values from simulation alongside asymptotic values |
| Sentiment neutrality in backtest | FinBERT is applied with a neutral headline assumption (no historical headlines ingested) in the rolling backtest | Document this assumption explicitly; treat crash AUROC as the primary signal of Model C's incremental value |
| Regime label instability | HMM regime labels may not be comparable across expanding windows | Use sorted state assignment (Bull $\geq$ Bear $\geq$ Crisis by mean return) as standard normalisation |
