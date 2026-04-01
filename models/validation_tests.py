"""
VaR Backtesting Diagnostic Tests
=================================
Standard implementations of Kupiec (1995) Proportion-of-Failures test
and Christoffersen (1998) conditional coverage test.

References:
  Kupiec, P. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives.
  Christoffersen, P. (1998). Evaluating interval forecasts.
    International Economic Review.
"""

import numpy as np
from scipy import stats


def kupiec_pof_test(violations, alpha):
    """
    Kupiec (1995) Proportion-of-Failures likelihood ratio test.

    Tests H0: the true violation rate equals `alpha`
    against H1: the true violation rate differs from `alpha`.

    Parameters
    ----------
    violations : array-like of {0, 1}
        1 = VaR violation (loss exceeded VaR), 0 = no violation.
    alpha : float
        Expected violation rate under the null (e.g. 0.05 for 95% VaR).

    Returns
    -------
    dict with test_statistic, p_value, reject_null
    """
    violations = np.asarray(violations, dtype=int)
    n = int(violations.sum())       # observed violations
    T = len(violations)             # total observations

    if T == 0:
        return {"test_statistic": None, "p_value": None, "reject_null": False}

    if n == 0 or n == T:
        # Degenerate cases: use one-sided limit of the LR
        # If n=0 and alpha>0, or n=T and alpha<1, strong evidence against H0
        # but the LR formula has log(0). Use a large statistic.
        if (n == 0 and alpha > 0) or (n == T and alpha < 1):
            return {"test_statistic": 999.0, "p_value": 0.0, "reject_null": True}
        return {"test_statistic": 0.0, "p_value": 1.0, "reject_null": False}

    # Observed violation rate
    p_hat = n / T

    # Standard Kupiec LR statistic (always non-negative):
    #   LR = -2 [ n·log(α) + (T-n)·log(1-α) - n·log(p̂) - (T-n)·log(1-p̂) ]
    log_L0 = n * np.log(alpha) + (T - n) * np.log(1 - alpha)
    log_L1 = n * np.log(p_hat) + (T - n) * np.log(1 - p_hat)
    test_statistic = -2 * (log_L0 - log_L1)

    p_value = 1 - stats.chi2.cdf(test_statistic, df=1)

    return {
        "test_statistic": round(float(test_statistic), 6),
        "p_value": round(float(p_value), 6),
        "reject_null": bool(p_value < 0.05),
    }


def christoffersen_independence_test(violations):
    """
    Christoffersen (1998) independence test for VaR violations.

    Tests H0: violations are serially independent (i.i.d. Bernoulli)
    against H1: violations exhibit first-order Markov dependence.

    Uses the 2×2 transition matrix of consecutive (viol_t, viol_{t+1}) pairs.

    Parameters
    ----------
    violations : array-like of {0, 1}

    Returns
    -------
    dict with test_statistic, p_value, reject_null
    """
    violations = np.asarray(violations, dtype=int)
    T = len(violations)

    if T < 2:
        return {"test_statistic": None, "p_value": None, "reject_null": False}

    # Build 2×2 transition count matrix
    # n_ij = number of transitions from state i to state j
    n00 = n01 = n10 = n11 = 0
    for t in range(T - 1):
        i, j = violations[t], violations[t + 1]
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        else:
            n11 += 1

    # Row totals
    n0 = n00 + n01       # transitions from state 0
    n1 = n10 + n11       # transitions from state 1

    # Guard against degenerate cases
    if n0 == 0 or n1 == 0:
        return {"test_statistic": 0.0, "p_value": 1.0, "reject_null": False}

    # Transition probabilities
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0

    # Unconditional violation probability
    total_violations = n01 + n11
    total_transitions = n0 + n1
    pi = total_violations / total_transitions if total_transitions > 0 else 0

    # Guard against log(0)
    if pi == 0 or pi == 1 or pi01 == 0 or pi11 == 0:
        return {"test_statistic": 0.0, "p_value": 1.0, "reject_null": False}
    if pi01 == 1 or pi11 == 1:
        return {"test_statistic": 0.0, "p_value": 1.0, "reject_null": False}

    # Log-likelihood under independence (H0: pi01 = pi11 = pi)
    log_L0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

    # Log-likelihood under Markov dependence (H1)
    log_L1 = 0.0
    if n00 > 0:
        log_L1 += n00 * np.log(1 - pi01)
    if n01 > 0:
        log_L1 += n01 * np.log(pi01)
    if n10 > 0:
        log_L1 += n10 * np.log(1 - pi11)
    if n11 > 0:
        log_L1 += n11 * np.log(pi11)

    test_statistic = -2 * (log_L0 - log_L1)

    p_value = 1 - stats.chi2.cdf(test_statistic, df=1)

    return {
        "test_statistic": round(float(test_statistic), 6),
        "p_value": round(float(p_value), 6),
        "reject_null": bool(p_value < 0.05),
    }


def christoffersen_conditional_coverage_test(violations, alpha):
    """
    Christoffersen (1998) conditional coverage test.

    Combines the Kupiec POF test (unconditional coverage) and the
    independence test into a joint LR statistic with df=2.

    Parameters
    ----------
    violations : array-like of {0, 1}
    alpha : float
        Expected violation rate under the null.

    Returns
    -------
    dict with test_statistic, p_value, reject_null
    """
    kupiec = kupiec_pof_test(violations, alpha)
    indep  = christoffersen_independence_test(violations)

    # If either sub-test is degenerate, return what we can
    if kupiec["test_statistic"] is None or indep["test_statistic"] is None:
        return {
            "test_statistic": None,
            "p_value": None,
            "reject_null": False,
        }

    test_statistic = kupiec["test_statistic"] + indep["test_statistic"]
    p_value = 1 - stats.chi2.cdf(test_statistic, df=2)

    return {
        "test_statistic": round(float(test_statistic), 6),
        "p_value": round(float(p_value), 6),
        "reject_null": bool(p_value < 0.05),
    }
