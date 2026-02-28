import torch
import numpy as np
from torch import distributions as dist
from scipy.stats import chi2

# Simulation parameters.
num_paths = 50000
num_assets = 7
T = 1       # 1 year horizon
dt = T / num_paths  # time step

# --- Asset parameters (example values) ---
mu = torch.tensor([0.08, 0.10, 0.07, 0.09, 0.06, 0.11, 0.08])  # annual expected returns.

# Example covariance matrix (symmetric positive definite)
raw = torch.randn(num_assets, num_assets) * 0.02
Sigma = raw @ raw.T + torch.eye(num_assets) * 0.01  # ensures PD

# --- Initial stock prices ---
S0 = torch.tensor([100.0, 150.0, 200.0, 120.0, 90.0, 180.0, 110.0])

# --- Brownian motion increments ---
W = dist.Normal(0, 1).sample([num_paths, num_assets]) * np.sqrt(dt)

# --- Cholesky decomposition for correlated returns ---
L = torch.linalg.cholesky(Sigma)           # lower-triangular factor.
correlated_W = W @ L.T                     # shape: (num_paths, num_assets)

# --- Geometric Brownian Motion paths ---
# S(t) = S0 * exp((mu - 0.5 * diag(Sigma)) * dt + correlated_dW)
sigma_diag = Sigma.diag()                  # variance of each asset
drift = (mu - 0.5 * sigma_diag) * dt      # shape: (num_assets,)

paths = S0.unsqueeze(0) * torch.exp(drift.unsqueeze(0) + correlated_W)
# paths shape: (num_paths, num_assets)

terminal_prices = paths[-1]  # last simulated prices for each path

# --- Value at Risk (VaR) ---
alpha = 0.05  # 95% confidence level
returns = (paths / S0.unsqueeze(0)) - 1    # relative returns, shape: (num_paths, num_assets)

# Portfolio-level VaR (equal-weighted portfolio)
weights = torch.ones(num_assets) / num_assets
portfolio_returns = returns @ weights       # shape: (num_paths,)
var_quantile = torch.quantile(portfolio_returns, alpha)

# Chi-squared based parametric VaR per asset
chi2_factor = chi2.ppf(1 - alpha, df=num_assets - 1)
var_parametric = torch.tensor(chi2_factor) * sigma_diag

# --- Results ---
print("=" * 50)
print(f"Monte Carlo GBM Simulation")
print(f"  Paths      : {num_paths}")
print(f"  Assets     : {num_assets}")
print(f"  Horizon    : {T} year(s)")
print("=" * 50)
print(f"\nMean terminal price per asset:")
for i, p in enumerate(paths.mean(dim=0)):
    print(f"  Asset {i+1}: {p:.4f}  (S0={S0[i]:.1f})")
print(f"\nPortfolio VaR (95%, simulation) : {var_quantile:.6f}")
print(f"\nParametric VaR per asset (chi2-based):")
for i, v in enumerate(var_parametric):
    print(f"  Asset {i+1}: {v:.6f}")
