import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2
class VaRBacktester: 


    def backtest(self, daily_returns, daily_var, confidence_level):
        """
        calculate the Kupiec test statistic of VaR
        """

        N = len(daily_returns)
        exceptions = np.sum(daily_returns < -daily_var)
        p = 1 - confidence_level
        if exceptions == 0:
            test_statistic = -2 * (np.log(1 - p) * N)
        else:
            test_statistic = -2 * np.log(((1 - p) ** (N - exceptions)) * (p ** exceptions) / ((1 - exceptions / N) ** (N - exceptions) * (exceptions / N) ** exceptions))
        p_value = 1 - chi2.cdf(test_statistic, df=1)
        
        if p_value < 0.05:
            print(f"Reject the null hypothesis: The VaR model is not accurate at {confidence_level*100}% confidence level.")
        return test_statistic, p_value 