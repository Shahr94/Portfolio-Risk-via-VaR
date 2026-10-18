import numpy as np

PORTFOLIO_CONFIG = {
    "tickers" : ["DTE.DE", "SIE.DE", "SAP.DE", "AAPL"],
    "weights" : np.array([0.2, 0.3, 0.25, 0.25]),
    "portfolio_value" : 1_000_000,
    "confidence_level" : 0.99,
    "horizon_days" : 10,
    "start_date" : "2023-10-14",
    "end_date" : "2026-10-14",
}