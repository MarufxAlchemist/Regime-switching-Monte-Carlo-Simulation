# Statistical Tests for Value-at-Risk Backtesting Validation

## Formal Derivations

---

## Notation and Setup

Let $\{r_t\}_{t=1}^{T}$ denote a sequence of realised portfolio returns over $T$ non-overlapping forecast windows, each of length $H = 30$ trading days.

At each forecast origin $t$, the model produces a predicted Value-at-Risk at confidence level $1 - \alpha$:

$$\widehat{\text{VaR}}_t^{(\alpha)}$$

such that, under correct model specification:

$$\Pr\!\bigl(r_t < -\widehat{\text{VaR}}_t^{(\alpha)}\bigr) = \alpha$$

Define the **violation indicator** (hit sequence):

$$I_t = \mathbf{1}\!\bigl\{r_t < -\widehat{\text{VaR}}_t^{(\alpha)}\bigr\}, \quad t = 1, \ldots, T$$

where $I_t \in \{0, 1\}$.  A violation ($I_t = 1$) occurs whenever the realised loss exceeds the predicted VaR.

Let:

- $n = \sum_{t=1}^{T} I_t$ — total number of violations
- $\hat{p} = n / T$ — empirical violation rate
- $\alpha$ — nominal violation probability (e.g. $\alpha = 0.05$ for VaR at 95% confidence)

### Null hypothesis framework

A correctly specified VaR model implies two properties of $\{I_t\}$:

1. **Unconditional coverage**: $\mathbb{E}[I_t] = \alpha$ for all $t$
2. **Independence**: $I_t \perp\!\!\!\perp I_s$ for all $t \neq s$

The three tests below evaluate these properties individually and jointly.

---

## 1  Kupiec Proportion of Failures (POF) Test

### 1.1  Hypotheses

$$H_0: p = \alpha \qquad \text{vs.} \qquad H_1: p \neq \alpha$$

where $p = \Pr(I_t = 1)$ is the true unconditional violation probability.

### 1.2  Assumptions

- The violation indicators $I_1, \ldots, I_T$ are **independent and identically distributed** Bernoulli random variables: $I_t \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$.
- The test examines **only** whether $p = \alpha$; it is silent on serial dependence.

### 1.3  Likelihood construction

Under the i.i.d. Bernoulli assumption, the joint likelihood is:

$$L(p) = \prod_{t=1}^{T} p^{I_t}(1-p)^{1-I_t} = p^{n}(1-p)^{T-n}$$

**Restricted likelihood** (under $H_0$):

$$L_0 = L(\alpha) = \alpha^{n}(1-\alpha)^{T-n}$$

**Unrestricted likelihood** (under $H_1$, maximised over $p$):

The MLE is $\hat{p} = n/T$, giving:

$$L_1 = L(\hat{p}) = \hat{p}^{\,n}(1-\hat{p})^{T-n}$$

### 1.4  Log-likelihood ratio statistic

$$\text{LR}_{\text{POF}} = -2\ln\!\frac{L_0}{L_1} = -2\Bigl[n\ln\alpha + (T-n)\ln(1-\alpha) - n\ln\hat{p} - (T-n)\ln(1-\hat{p})\Bigr]$$

Equivalently:

$$\boxed{\text{LR}_{\text{POF}} = -2\left[n\ln\!\frac{\alpha}{\hat{p}} + (T-n)\ln\!\frac{1-\alpha}{1-\hat{p}}\right]}$$

### 1.5  Asymptotic distribution

Under $H_0$ and standard regularity conditions (since the Bernoulli model is a one-parameter exponential family):

$$\text{LR}_{\text{POF}} \xrightarrow{d} \chi^2(1) \quad \text{as } T \to \infty$$

Reject $H_0$ at significance level $\gamma$ if $\text{LR}_{\text{POF}} > \chi^2_{1,1-\gamma}$.

### 1.6  Edge cases

| Condition | Issue | Treatment |
|---|---|---|
| $n = 0$ (no violations) | $\hat{p} = 0$; terms $n \ln \hat{p}$ are indeterminate as $0 \cdot \ln 0$ | Apply the convention $0 \cdot \ln 0 = 0$. The statistic reduces to $\text{LR}_{\text{POF}} = -2\,T\ln\!\frac{1-\alpha}{1}$. Under $H_0$ with $\alpha = 0.05$ and moderate $T$, zero violations is itself extreme evidence against correct coverage, yet the $\chi^2$ approximation degrades. Use exact binomial $p$-value: $P(n = 0) = (1-\alpha)^T$. |
| $n = T$ (all violations) | $1 - \hat{p} = 0$; analogous singularity | Convention $0 \cdot \ln 0 = 0$. Statistic becomes $-2\,T\ln\!\frac{\alpha}{1}$. In practice this signals catastrophic model failure. |
| Small $T$ ($T < 50$) | $\chi^2(1)$ approximation is unreliable | Use the exact binomial test: $p\text{-value} = P_{\text{Bin}(T,\alpha)}(X = n)$ via two-tailed summation, or compute critical values by direct enumeration of $\text{LR}_{\text{POF}}$ over $n \in \{0, \ldots, T\}$. |

### 1.7  Financial risk interpretation

- **Rejection with $\hat{p} > \alpha$**: The model **under-estimates** risk — VaR is too permissive. Realised losses breach the threshold more often than the $\alpha$ nominal rate. This is the primary regulatory concern (Basel III green/yellow/red zone classification is based on this count).
- **Rejection with $\hat{p} < \alpha$**: The model is **too conservative** — capital reserves are inefficiently large. While less dangerous from a prudential standpoint, this indicates poor calibration and potential misallocation of risk capital.
- **Limitation**: The POF test has **low power** for small $T$. With $T = 250$ and $\alpha = 0.05$, the expected number of violations is only 12.5, making it difficult to distinguish $p = 0.05$ from $p = 0.07$.

---

## 2  Christoffersen Independence Test

### 2.1  Motivation

Even if $\hat{p} \approx \alpha$, violations may cluster temporally. A model that produces five consecutive violations followed by 95 non-violations achieves the correct unconditional rate but fails to capture volatility dynamics. The independence test detects such clustering.

### 2.2  Hypotheses

Model $\{I_t\}$ as a first-order Markov chain with transition matrix:

$$\Pi = \begin{pmatrix} 1 - \pi_{01} & \pi_{01} \\ 1 - \pi_{11} & \pi_{11} \end{pmatrix}$$

where:

- $\pi_{ij} = \Pr(I_t = j \mid I_{t-1} = i), \quad i, j \in \{0, 1\}$

$$H_0: \pi_{01} = \pi_{11} \equiv \pi \qquad \text{(independence: transition probabilities do not depend on prior state)}$$

$$H_1: \pi_{01} \neq \pi_{11} \qquad \text{(serial dependence exists)}$$

### 2.3  Assumptions

- $\{I_t\}_{t=1}^{T}$ follows a **first-order stationary Markov chain**.
- The test is designed to detect first-order dependence. Higher-order dependence structures (e.g. $I_t$ depending on $I_{t-2}$) require extensions.
- Stationarity of the transition probabilities across the sample.

### 2.4  Sufficient statistics

Define the transition counts from the observed hit sequence:

$$n_{ij} = \sum_{t=2}^{T} \mathbf{1}\{I_{t-1} = i,\; I_t = j\}, \quad i, j \in \{0, 1\}$$

and the row sums:

$$n_{i\cdot} = n_{i0} + n_{i1}, \quad i \in \{0, 1\}$$

The **MLE transition probabilities** under the alternative are:

$$\hat{\pi}_{ij} = \frac{n_{ij}}{n_{i\cdot}}, \quad i, j \in \{0, 1\}$$

Under $H_0$ (independence), the common violation probability is estimated as:

$$\hat{\pi} = \frac{n_{01} + n_{11}}{n_{0\cdot} + n_{1\cdot}} = \frac{n_{01} + n_{11}}{T - 1}$$

### 2.5  Likelihood construction

**Restricted likelihood** (under $H_0$: $\pi_{01} = \pi_{11} = \pi$):

$$L_0^{\text{ind}} = (1 - \hat{\pi})^{n_{00} + n_{10}} \cdot \hat{\pi}^{\,n_{01} + n_{11}}$$

**Unrestricted likelihood** (under $H_1$: separate transition probabilities):

$$L_1^{\text{ind}} = (1 - \hat{\pi}_{01})^{n_{00}} \cdot \hat{\pi}_{01}^{\,n_{01}} \cdot (1 - \hat{\pi}_{11})^{n_{10}} \cdot \hat{\pi}_{11}^{\,n_{11}}$$

### 2.6  Log-likelihood ratio statistic

$$\text{LR}_{\text{ind}} = -2\ln\!\frac{L_0^{\text{ind}}}{L_1^{\text{ind}}}$$

Expanding:

$$\boxed{\text{LR}_{\text{ind}} = -2\Bigl[(n_{00} + n_{10})\ln(1 - \hat{\pi}) + (n_{01} + n_{11})\ln\hat{\pi} - n_{00}\ln(1 - \hat{\pi}_{01}) - n_{01}\ln\hat{\pi}_{01} - n_{10}\ln(1 - \hat{\pi}_{11}) - n_{11}\ln\hat{\pi}_{11}\Bigr]}$$

### 2.7  Asymptotic distribution

Under $H_0$, the restricted model has 1 free parameter ($\pi$) and the unrestricted model has 2 free parameters ($\pi_{01}, \pi_{11}$). Therefore:

$$\text{LR}_{\text{ind}} \xrightarrow{d} \chi^2(1) \quad \text{as } T \to \infty$$

Reject $H_0$ at level $\gamma$ if $\text{LR}_{\text{ind}} > \chi^2_{1,1-\gamma}$.

### 2.8  Edge cases

| Condition | Issue | Treatment |
|---|---|---|
| $n_{1\cdot} = 0$ (no transitions from state 1, i.e., $n = 0$ or $n = 1$ with the single violation at $t = T$) | $\hat{\pi}_{11}$ is undefined (division by zero) | The Markov chain has never been observed in state 1 long enough to estimate $\pi_{11}$. Set $\text{LR}_{\text{ind}} = 0$ (cannot reject independence) or report "insufficient data". |
| $n_{0\cdot} = 0$ (never in state 0) | $\hat{\pi}_{01}$ is undefined | Analogous: all observations are violations. The test is degenerate. |
| $n_{ij} = 0$ for some $(i,j)$ | Log terms $n_{ij} \ln \hat{\pi}_{ij}$ involve $0 \cdot \ln 0$ | Apply convention $0 \cdot \ln 0 = 0$. |
| Small $T$ or sparse violations | $\chi^2(1)$ approximation poor; transition counts $n_{ij}$ may be $< 5$ | Use exact permutation test or Monte Carlo simulation of the null distribution. Fisher's exact test on the $2 \times 2$ transition count matrix $(n_{00}, n_{01}; n_{10}, n_{11})$ is an alternative. |

### 2.9  Financial risk interpretation

- **Rejection**: Violations are serially dependent — they tend to **cluster**. This implies the model fails to adapt to changing volatility regimes: once a VaR breach occurs, the model does not adequately revise risk estimates, leading to runs of consecutive violations.
- **$\hat{\pi}_{11} > \hat{\pi}_{01}$**: Positive autocorrelation of violations — violation clustering. Most common failure mode; indicates sluggish regime adaptation.
- **$\hat{\pi}_{11} < \hat{\pi}_{01}$**: Negative autocorrelation — violations alternate with non-violations. Rare in practice but could indicate over-reactive model recalibration.
- **Regulatory relevance**: Basel III does not explicitly mandate an independence test, but the 2019 FRTB (Fundamental Review of the Trading Book) emphasises that VaR models should not exhibit predictable patterns of failure.

---

## 3  Christoffersen Conditional Coverage Test

### 3.1  Motivation

The POF test examines $\mathbb{E}[I_t] = \alpha$ (correct level) and the independence test examines $I_t \perp\!\!\!\perp I_{t-1}$ (no clustering). A correctly specified VaR model must satisfy **both** simultaneously. The conditional coverage test is a joint test that combines these requirements.

### 3.2  Hypotheses

$$H_0: \bigl(p = \alpha\bigr) \;\cap\; \bigl(\pi_{01} = \pi_{11}\bigr) \qquad \text{(correct unconditional coverage AND independence)}$$

$$H_1: \bigl(p \neq \alpha\bigr) \;\cup\; \bigl(\pi_{01} \neq \pi_{11}\bigr) \qquad \text{(either or both properties fail)}$$

Equivalently, under $H_0$ the violation sequence is i.i.d. Bernoulli$(\alpha)$.

### 3.3  Assumptions

- Same as the independence test: $\{I_t\}$ is modelled as a first-order Markov chain.
- Additionally, the unconditional violation probability is specified as $\alpha$ under $H_0$.
- The test is therefore a joint test against a composite alternative that nests both the POF and independence alternatives.

### 3.4  Likelihood construction

**Restricted likelihood** (under $H_0$: i.i.d. Bernoulli$(\alpha)$):

This is identical to the POF null likelihood, evaluated on the Markov chain sufficient statistics:

$$L_0^{\text{cc}} = \alpha^{n_{01} + n_{11}}(1-\alpha)^{n_{00} + n_{10}}$$

> [!NOTE]
> We use $(T-1)$ transitions here. The contribution of the initial observation $I_1$ is a constant $\alpha^{I_1}(1-\alpha)^{1-I_1}$ under $H_0$. It can be included for exactness; in large samples the effect is negligible. The standard Christoffersen (1998) formulation conditions on $I_1$ and uses only transition counts.

**Unrestricted likelihood** (under $H_1$: first-order Markov with free $\pi_{01}, \pi_{11}$):

$$L_1^{\text{cc}} = (1 - \hat{\pi}_{01})^{n_{00}} \cdot \hat{\pi}_{01}^{\,n_{01}} \cdot (1 - \hat{\pi}_{11})^{n_{10}} \cdot \hat{\pi}_{11}^{\,n_{11}}$$

This is the same unrestricted likelihood used in the independence test.

### 3.5  Log-likelihood ratio statistic

$$\text{LR}_{\text{cc}} = -2\ln\!\frac{L_0^{\text{cc}}}{L_1^{\text{cc}}}$$

Expanding:

$$\boxed{\text{LR}_{\text{cc}} = -2\Bigl[(n_{00} + n_{10})\ln(1-\alpha) + (n_{01} + n_{11})\ln\alpha - n_{00}\ln(1-\hat{\pi}_{01}) - n_{01}\ln\hat{\pi}_{01} - n_{10}\ln(1-\hat{\pi}_{11}) - n_{11}\ln\hat{\pi}_{11}\Bigr]}$$

### 3.6  Decomposition property

A key structural result: the conditional coverage statistic **decomposes additively** into the POF and independence components:

$$\text{LR}_{\text{cc}} = \text{LR}_{\text{POF}} + \text{LR}_{\text{ind}}$$

**Proof sketch**:

$$\text{LR}_{\text{cc}} = -2\ln\!\frac{L_0^{\text{cc}}}{L_1^{\text{cc}}} = -2\ln\!\frac{L_0^{\text{cc}}}{L_0^{\text{ind}}} \cdot \frac{L_0^{\text{ind}}}{L_1^{\text{cc}}}$$

$$= \underbrace{-2\ln\!\frac{L(\alpha)}{L(\hat{p})}}_{\text{LR}_{\text{POF}}} + \underbrace{-2\ln\!\frac{L_0^{\text{ind}}}{L_1^{\text{ind}}}}_{\text{LR}_{\text{ind}}}$$

where we used $L_0^{\text{cc}} = L(\alpha)$, $L_0^{\text{ind}} = L(\hat{p})$ (both evaluated on the Bernoulli likelihood over transitions), and $L_1^{\text{cc}} = L_1^{\text{ind}}$.

This decomposition holds exactly.

### 3.7  Asymptotic distribution

Under $H_0$, the restricted model has 0 free parameters ($p$ fixed at $\alpha$, no dependence parameters) and the unrestricted model has 2 free parameters ($\pi_{01}, \pi_{11}$). Therefore:

$$\text{LR}_{\text{cc}} \xrightarrow{d} \chi^2(2) \quad \text{as } T \to \infty$$

Reject $H_0$ at level $\gamma$ if $\text{LR}_{\text{cc}} > \chi^2_{2,1-\gamma}$.

By the additive decomposition:

$$\chi^2(2) = \chi^2(1)_{\text{POF}} + \chi^2(1)_{\text{ind}}$$

which holds because the POF and independence components are asymptotically independent under $H_0$ (they test orthogonal restrictions on the parameter space).

### 3.8  Edge cases

All edge cases from Sections 1.6 and 2.8 apply jointly:

| Condition | Impact |
|---|---|
| $n = 0$ or $n = T$ | Both $\text{LR}_{\text{POF}}$ and $\text{LR}_{\text{ind}}$ are degenerate. Use exact tests. |
| $n_{1\cdot} = 0$ | Independence component is degenerate. $\text{LR}_{\text{cc}}$ reduces to $\text{LR}_{\text{POF}}$. |
| Small $T$ ($< 100$) | With $\alpha = 0.05$ and $T = 100$, expected violations $\approx 5$. Transition counts will typically have cells $< 5$, making $\chi^2$ unreliable. **Recommendation**: use Monte Carlo simulation of $\text{LR}_{\text{cc}}$ under $H_0$ (draw $B = 10{,}000$ i.i.d. Bernoulli$(\alpha)$ sequences of length $T$, compute the statistic for each, and use the empirical distribution for $p$-values). |
| Overlapping forecast horizons ($H > 1$ with step size $< H$) | Violations are mechanically autocorrelated due to overlapping return windows. The i.i.d. Bernoulli null is misspecified. **Correction**: use non-overlapping windows only, or apply a Newey–West type spectral correction to the likelihood. |

### 3.9  Financial risk interpretation

- **Rejection via $\text{LR}_{\text{POF}}$ only (and not $\text{LR}_{\text{ind}}$)**: The model has incorrect *level* — either too many or too few violations — but violations occur at random times. Recalibration of model parameters (e.g. volatility scaling, regime thresholds) is needed.
- **Rejection via $\text{LR}_{\text{ind}}$ only (and not $\text{LR}_{\text{POF}}$)**: The model has correct average coverage but violations cluster. This is the hallmark of **regime-switching models that are too slow to transition**, or of models that fail to incorporate time-varying volatility. In the context of this project's HMM-based engine, this finding would suggest the HMM's transition matrix or the contagion propagation mechanism is not capturing abrupt regime shifts.
- **Rejection via both**: The model is fundamentally miscalibrated. Both the level and dynamics are wrong. A complete model revision is warranted.
- **Non-rejection does not prove correctness**: The test has finite power. With $T = 250$ and $\alpha = 0.05$, the power against a true violation rate of $p = 0.08$ is approximately 30–40%. Longer backtesting histories or higher $\alpha$ (e.g. VaR at 99% with $\alpha = 0.01$) improve discriminatory ability.

---

## Summary of Test Statistics

| Test | Statistic | $H_0$ | Degrees of freedom | Detects |
|---|---|---|---|---|
| Kupiec POF | $\text{LR}_{\text{POF}} = -2\bigl[n\ln\frac{\alpha}{\hat{p}} + (T-n)\ln\frac{1-\alpha}{1-\hat{p}}\bigr]$ | $p = \alpha$ | 1 | Incorrect coverage level |
| Christoffersen Ind. | $\text{LR}_{\text{ind}} = -2\ln\frac{L_0^{\text{ind}}}{L_1^{\text{ind}}}$ | $\pi_{01} = \pi_{11}$ | 1 | Violation clustering |
| Christoffersen CC | $\text{LR}_{\text{cc}} = \text{LR}_{\text{POF}} + \text{LR}_{\text{ind}}$ | $p = \alpha$ and $\pi_{01} = \pi_{11}$ | 2 | Both simultaneously |

---

## References

1. Kupiec, P. H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 3(2), 73–84.
2. Christoffersen, P. F. (1998). "Evaluating Interval Forecasts." *International Economic Review*, 39(4), 841–862.
3. Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk*. Bank for International Settlements.
4. Campbell, S. D. (2006). "A Review of Backtesting and Backtesting Procedures." *Journal of Risk*, 9(2), 1–17.
