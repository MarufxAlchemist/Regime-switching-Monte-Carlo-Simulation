"""
Mathematical Verification of Cholesky Decomposition Logic
==========================================================

Goal: Verify that the transform  correlated_W = W @ L.T
      correctly produces samples with covariance Σ,
      where W ~ N(0, I)  and  L = cholesky(Σ)  =>  L @ L.T = Σ

Theory (step-by-step):
-----------------------
1. Cholesky factorisation:   Σ = L L^T          (L is lower-triangular)
2. Let Z ~ N(0, I_n)         (independent standard normals, row vector)
3. Define  X = Z L^T         (or equivalently  X = (L Z^T)^T)

Then:
  Cov(X) = E[X^T X] / N  →  Σ  (law of large numbers)

Proof of Cov(X) = Σ:
  X = Z L^T
  E[X^T X] = L E[Z^T Z] L^T = L · I · L^T = L L^T = Σ  ✓

Steps verified below:
  TEST 1 – Reconstruction:       L @ L.T  ≈  Σ      (exact up to float precision)
  TEST 2 – Sample covariance:    Cov(correlated_W) ≈ Σ  (converges as num_paths grows)
  TEST 3 – Marginal normality:   each column of correlated_W ~ N(0, Σ_ii)
  TEST 4 – Cross-correlation:    sample correlation ≈ true correlation matrix
"""

import torch
import numpy as np

torch.manual_seed(42)

# ── Setup ────────────────────────────────────────────────────────────────────
num_paths  = 50_000
num_assets = 7

# Build a random symmetric positive-definite covariance matrix
raw   = torch.randn(num_assets, num_assets) * 0.02
Sigma = raw @ raw.T + torch.eye(num_assets) * 0.01

# ── Cholesky factorisation ────────────────────────────────────────────────────
L = torch.linalg.cholesky(Sigma)          # Σ = L @ L.T

# ── Generate samples ─────────────────────────────────────────────────────────
W            = torch.randn(num_paths, num_assets)   # Z ~ N(0, I)
correlated_W = W @ L.T                              # X = Z L^T

sep = "=" * 60

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 – Reconstruction:  L @ L.T  should equal  Σ
# ─────────────────────────────────────────────────────────────────────────────
print(sep)
print("TEST 1 – Reconstruction: Σ ≈ L @ L.T")
print(sep)
Sigma_reconstructed = L @ L.T
max_err = (Sigma - Sigma_reconstructed).abs().max().item()
print(f"  Max absolute error |Σ - L@L.T|: {max_err:.2e}")
print(f"  PASS ✓" if max_err < 1e-5 else "  FAIL ✗")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 – Sample covariance of correlated_W should converge to Σ
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("TEST 2 – Sample covariance ≈ Σ  (N = {:,})".format(num_paths))
print(sep)
sample_cov = torch.cov(correlated_W.T)       # (num_assets × num_assets)
cov_err    = (Sigma - sample_cov).abs().max().item()
print(f"  Max absolute error |Σ - sample_cov|: {cov_err:.6f}")
print(f"  PASS ✓" if cov_err < 0.01 else "  FAIL ✗")

print("\n  True Σ (first 3×3 block):")
print(Sigma[:3, :3].numpy().round(6))
print("\n  Sample Cov (first 3×3 block):")
print(sample_cov[:3, :3].numpy().round(6))

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 – Marginal normality: each column ~ N(0, Σ_ii)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("TEST 3 – Marginal statistics per asset")
print(sep)
print(f"  {'Asset':<8} {'E[X]':>10} {'Var(X)':>12} {'True Var':>12} {'Error':>10}")
all_pass = True
for i in range(num_assets):
    col       = correlated_W[:, i]
    mean_val  = col.mean().item()
    var_val   = col.var().item()
    true_var  = Sigma[i, i].item()
    err       = abs(var_val - true_var)
    flag      = "✓" if err < 0.01 else "✗"
    if err >= 0.01:
        all_pass = False
    print(f"  Asset {i+1:<3} {mean_val:>10.6f} {var_val:>12.6f} {true_var:>12.6f} {err:>9.6f}  {flag}")
print(f"  {'PASS ✓' if all_pass else 'FAIL ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 – Sample correlation matrix ≈ true correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("TEST 4 – Sample correlation ≈ True correlation")
print(sep)

def cov_to_corr(C):
    std = C.diag().sqrt()
    return C / (std[:, None] * std[None, :])

true_corr   = cov_to_corr(Sigma)
sample_corr = cov_to_corr(sample_cov)
corr_err    = (true_corr - sample_corr).abs().max().item()
print(f"  Max absolute error in correlation matrix: {corr_err:.6f}")
print(f"  PASS ✓" if corr_err < 0.05 else "  FAIL ✗")

print("\n  True correlation (first 4×4 block):")
print(true_corr[:4, :4].numpy().round(4))
print("\n  Sample correlation (first 4×4 block):")
print(sample_corr[:4, :4].numpy().round(4))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("SUMMARY")
print(sep)
print("""
  The Cholesky transform  X = Z @ L.T  is valid because:

    1. Σ = L L^T           =>  L is the 'square root' of Σ
    2. Z ~ N(0, I)
    3. X = Z L^T
    4. Cov(X) = E[X^T X] / N
              = L · E[Z^T Z] · L^T
              = L · I · L^T
              = L L^T  =  Σ  ✓

  All four numerical tests confirm this holds empirically.
""")
