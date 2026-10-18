from risk_models import RiskCalculator
from backtesting import VaRBacktester
from config import PORTFOLIO_CONFIG
import yfinance as yf
import numpy as np

data = yf.download(PORTFOLIO_CONFIG["tickers"], start=PORTFOLIO_CONFIG["start_date"], end=PORTFOLIO_CONFIG["end_date"])
cleaned_data = data.ffill()  
daily_returns = np.log(cleaned_data['Close'] / cleaned_data['Close'].shift(1)).dropna()

if not np.isclose(np.sum(PORTFOLIO_CONFIG["weights"]), 1.0):
    raise ValueError("Portfolio weights must sum to 1.")
if len(PORTFOLIO_CONFIG["weights"]) != len(PORTFOLIO_CONFIG["tickers"]):
    raise ValueError("Number of weights must match number of tickers.")

# initialize the RiskCalculator with parameters from the config
risk_calculator = RiskCalculator(
    portfolio_value=PORTFOLIO_CONFIG["portfolio_value"],
    confidence_level=PORTFOLIO_CONFIG["confidence_level"],
    horizon_days=PORTFOLIO_CONFIG["horizon_days"]
)
parametric_var = risk_calculator.parametric_var(daily_returns, PORTFOLIO_CONFIG["weights"])
historical_var, expected_shortfall = risk_calculator.historical_var_and_ES(daily_returns, PORTFOLIO_CONFIG["weights"])

print(f"Parametric VaR: €{parametric_var:,.2f}")
print(f"Historical VaR: €{historical_var:,.2f}")
print(f"Expected Shortfall: €{expected_shortfall:,.2f}")


print ("\n--- Running Kupiec POF backtest ---")
# Real 1 day percentage VaR limit
daily_mean = daily_returns.mean()
daily_cov = daily_returns.cov()
portfolio_daily_mean = np.dot(PORTFOLIO_CONFIG["weights"], daily_mean)
portfolio_daily_var = PORTFOLIO_CONFIG["weights"].T @ daily_cov @ PORTFOLIO_CONFIG["weights"]
portfolio_daily_vol = np.sqrt(portfolio_daily_var)
daily_var_pct = risk_calculator.z_score * portfolio_daily_vol - portfolio_daily_mean
historical_portfolio_returns = daily_returns.dot(PORTFOLIO_CONFIG["weights"])

backtester = VaRBacktester()
test_statistic, p_value = backtester.backtest(daily_returns = historical_portfolio_returns, 
                                              daily_var = daily_var_pct, 
                                              confidence_level = PORTFOLIO_CONFIG["confidence_level"])
print(f"Kupiec test statistic: {test_statistic:.4f}")