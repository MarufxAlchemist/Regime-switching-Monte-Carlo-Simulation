"""
models — Regime-switching Monte Carlo systemic risk engine.

Sub-packages
------------
monte_carlo  : GBM path simulation and Cholesky verification
regime       : HMM-based market regime detection
sentiment    : FinBERT sentiment scoring and GBM parameter adjustment
network      : Correlation network engine and contagion simulation

Top-level modules
-----------------
backtest_engine   : Expanding-window walk-forward VaR backtester
validation_tests  : Kupiec POF and Christoffersen independence tests
comparative_study : Model comparison harness (baseline / regime / full)
"""
