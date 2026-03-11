import numpy as np
from scipy import stats
import math

def kupiec_pof_test(violations, alpha):
    n = violations.sum()  # number of violations
    T = len(violations)   # total number of observations
    
    if n==0 or n==T:      # If all or no violations occur, we cannot calculate test statistic and p-value
        return {
            "test_statistic": None,
            "p_value": None,
            "reject_null": False   # Can't reject H0 if n is 0 or T
        }
    else:
        lr = math.log(alpha / (1-alpha)) * np.sqrt(n/T)       # log likelihood ratio under the null
        test_statistic = -2*(lr + n*np.log((T-n)/n))          # Kupiec's POF statistic
        
        p_value = 1 - stats.chi2.cdf(test_statistic, 1)        # Calculate p-value using chi square distribution
    
    return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "reject_null": (p_value < 0.05)                      # If p value is less than 0.05 then we reject H0
        }
        
def christoffersen_independence_test(violations):
    violations_next = np.roll(violations, -1) 
    
    valid_observation = ~((violations == 0)&(violations_next==0)) # ignore the observations where both are zero
    
    violations_r = violations[valid_observation]
    violations_next_r = violations_next[valid_observation] 
        
    contingency_table = np.array([violations_r, violations_next_r])
    
    chi2_statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
       
    return {
            "test_statistic": chi2_statistic,
            "p_value": p_value,
            "reject_null": (p_value < 0.05)                      # If p value is less than <｜begin▁of▁sentence｜> then we reject H0
        }
    
def christoffersen_conditional_coverage_test(violations, alpha):
    kupiec_result = kupiec_pof_test(violations, alpha)
    independence_result = christoffersen_independence_test(violations) 
    
    test_statistic = kupiec_result["test_statistic"] + independence_result["test_statistic"]
    p_value = 1 - stats.chi2.cdf(test_statistic, 2)         # Calculate combined p-value using chi square distribution with df=2
    
    return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "reject_null": (p_value < 0.05)                   # If p value is less than <｜begin▁of▁sentence｜> then we reject H0
        }
