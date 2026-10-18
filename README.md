# Institutional Portfolio Risk & Backtesting Engine

A robust, object-oriented market risk engine built in Python to calculate, compare, and statistically validate regulatory risk metrics for a multi-asset equity portfolio. 

**Motivation:** Built to translate rigorous mathematical modeling from mechanical engineering into quantitative finance. This project applies core principles from the FRM and CFA curricula—specifically the transition from theoretical risk distributions to empirical tail-risk modeling—into a modular, production-ready software architecture.

---

## 🏗️ Mathematical Architecture

The engine is decoupled into configuration, calculation, and validation modules, implementing the following core market risk models:

### 1. Value at Risk (VaR)
*   **Parametric VaR:** Assumes returns follow a normal distribution. Calculated using the portfolio's covariance matrix, scaled over a 10-day horizon under the i.i.d. assumption:
    $$\text{VaR}_{10} = \text{Portfolio Value} \times (z_{\alpha} \cdot \sigma_{10} - \mu_{10})$$
*   **Historical Simulation VaR:** Makes no distributional assumptions. Computes the empirical percentile of historical portfolio returns to determine the tail loss threshold.

### 2. Expected Shortfall (CVaR)
Aligning with the Basel Committee's Fundamental Review of the Trading Book (FRTB), the engine calculates Expected Shortfall to quantify the severity of losses beyond the VaR threshold:
$$\text{ES} = \mathbb{E}[L \mid L > \text{VaR}]$$

### 3. Model Validation (Kupiec POF Test)
A VaR model is strictly theoretical until proven empirically. The engine utilizes the Kupiec Proportion of Failures (POF) Likelihood Ratio test to statistically validate the accuracy of the 99% VaR model based on a Chi-Square distribution ($df=1$):
$$LR_{\text{POF}} = -2 \ln\left( \frac{(1-p)^{N-x} p^x}{(1-\frac{x}{N})^{N-x} (\frac{x}{N})^x} \right)$$

---

## 📊 Key Findings: The "Fat Tail" Reality

Backtesting the 99% Parametric VaR limit against 3 years of actual market data (2023–2026) for a €1,000,000 DAX/Tech portfolio yielded critical insights:

*   **Parametric VaR (10-Day):** €67,406.57
*   **Expected Shortfall (10-Day):** €116,047.10
*   **Kupiec Test Statistic:** 5.4366 (Rejected)

**Conclusion:** The Kupiec test successfully rejected the Parametric VaR model. The Expected Shortfall being nearly double the Parametric VaR empirically proves that equity returns exhibit "fat tails" and non-normality. Standard bell-curve assumptions severely underestimate actual tail risk during market shocks, validating the regulatory shift toward Expected Shortfall for capital buffer requirements.

---

## ⚙️ Tech Stack & Project Structure

*   **Language:** Python 3.x
*   **Core Libraries:** `numpy` (vectorized matrix operations), `pandas` (time-series data wrangling), `scipy.stats` (statistical distributions and hypothesis testing), `yfinance` (market data extraction).

```text
portfolio-risk-engine/
│
├── src/
│   ├── __init__.py          
│   ├── risk_models.py       # Core RiskCalculator class (Parametric, Historical, ES)
│   └── backtesting.py       # Kupiec POF Likelihood Ratio validation class
│
├── config.py                # Decoupled portfolio parameters and limits
├── main.py                  # Execution driver and data pipeline
└── README.md