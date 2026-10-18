import numpy as np
from scipy.stats import norm


class RiskCalculator:
    def __init__(self, portfolio_value, confidence_level = 0.99, horizon_days = 10):
        """
        initialize the risk engine with core parameters
        """
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days

        # calculate the z-score for the given confidence level
        self.z_score = norm.ppf(self.confidence_level)
    def parametric_var(self, daily_returns, weights):
        """
        calculates and returns the parametric VaR for the portfolio
        """

        mean_return = daily_returns.mean() 
        cov_return = daily_returns.cov() 

        portfolio_return = np.dot(weights, mean_return)
        portfolio_covariance = np.dot(weights.T, np.dot(cov_return, weights))
        portfolio_volatility = np.sqrt(portfolio_covariance)

        period_day_var = portfolio_volatility * np.sqrt(self.horizon_days)
        VaR_horizon = self.portfolio_value * (self.z_score * period_day_var - portfolio_return*self.horizon_days)

        return VaR_horizon

    def historical_var_and_ES(self, daily_returns, weights):
        """
        calculates and returns a tuple: n-day Historical VaR and n-day Expected Shortfall for the portfolio
        """

        percentile_threshhold = (1 - self.confidence_level) * 100

        daily_portfolio_return = daily_returns.dot(weights) 
        worst_1percent = np.percentile(daily_portfolio_return, percentile_threshhold, method='lower')
        Hist_VaR_horizon = np.abs(self.portfolio_value * worst_1percent * np.sqrt(self.horizon_days))
        ES = daily_portfolio_return[daily_portfolio_return <= worst_1percent].mean()
        ES_horizon =np.abs( self.portfolio_value * (ES * np.sqrt(self.horizon_days)))

        return Hist_VaR_horizon, ES_horizon 