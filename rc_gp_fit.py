#!/usr/bin/env python3
"""
Minimal rotation-curve fit with a residual GP (NumPy + SciPy only).

Model:
  y(R) = V0  +  GP_Matern32(R; amp, scale)  +  noise
  noise^2 = sigma_obs_kms^2 + jitter^2

Parameters we fit (theta):
  V0, log_amp, log_scale, log_jitter

Outputs:
  - PNG plot at data/rotation_curve/beordo2024/rc_gp_fit.png
  - Console print of MLE theta
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# -----------------------
# 1) Load data
# -----------------------
csv_path = Path("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
df = pd.read_csv(csv_path).sort_values("R_kpc").reset_index(drop=True)

R = df["R_kpc"].to_numpy(dtype=float)
y = df["Vphi_kms"].to_numpy(dtype=float)
sigma_obs = df["sigma_obs_kms"].to_numpy(dtype=float)

# -----------------------
# 2) Kernel (Matern-3/2)
# -----------------------
def matern32_kernel(x, y, amp, scale):
    """
    k(r) = amp^2 * (1 + sqrt(3) r/scale) * exp(-sqrt(3) r/scale)
    """
    # pairwise distances
    dx = np.abs(x[:, None] - y[None, :])
    z = np.sqrt(3.0) * dx / scale
    return (amp**2) * (1.0 + z) * np.exp(-z)

# -----------------------
# 3) Mean model (flat for smoke test)
# -----------------------
def mean_model(R, V0):
    return np.full_like(R, float(V0))

# -----------------------
# 4) Negative log marginal likelihood
# -----------------------
def nll(theta):
    V0, log_amp, log_scale, log_jitter = theta
    amp = np.exp(log_amp)
    scale = np.exp(log_scale)
    jitter = np.exp(log_jitter)

    mu = mean_model(R, V0)
    # GP covariance (without noise)
    K = matern32_kernel(R, R, amp, scale)
    # Add heteroscedastic noise on the diagonal
    K[np.diag_indices_from(K)] += (sigma_obs**2 + jitter**2)

    try:
        c, lower = cho_factor(K, check_finite=False)
        alpha = cho_solve((c, lower), (y - mu), check_finite=False)
        # log det from Cholesky
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        return 0.5 * ((y - mu) @ alpha + logdet + len(R) * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        # numerical issue → penalize
        return 1e25

# -----------------------
# 5) Optimize (MLE)
# -----------------------
theta0 = np.array([232.0, np.log(10.0), np.log(2.0), np.log(1.0)], dtype=float)
bounds = [
    (180.0, 280.0),   # V0 bounds (km/s) — loose
    (np.log(1e-3), np.log(1e3)),   # log_amp
    (np.log(0.1), np.log(10.0)),   # log_scale (kpc)
    (np.log(1e-3), np.log(30.0)),  # log_jitter (km/s)
]
res = minimize(lambda th: float(nll(th)), x0=theta0, method="L-BFGS-B", bounds=bounds)
theta_hat = res.x
V0_hat, log_amp_hat, log_scale_hat, log_jitter_hat = theta_hat
amp_hat = np.exp(log_amp_hat)
scale_hat = np.exp(log_scale_hat)
jitter_hat = np.exp(log_jitter_hat)

print("MLE success:", res.success, "| message:", res.message)
print("MLE theta:")
print(f"  V0 = {V0_hat:.3f} km/s")
print(f"  amp = {amp_hat:.3f} km/s")
print(f"  scale = {scale_hat:.3f} kpc")
print(f"  jitter = {jitter_hat:.3f} km/s")

# -----------------------
# 6) Posterior mean & variance at training R
#    (standard GP conditioning)
# -----------------------
mu_hat = mean_model(R, V0_hat)
K = matern32_kernel(R, R, amp_hat, scale_hat)
# Training covariance WITH noise:
K_train = K.copy()
K_train[np.diag_indices_from(K_train)] += (sigma_obs**2 + jitter_hat**2)

c, lower = cho_factor(K_train, check_finite=False)
alpha = cho_solve((c, lower), (y - mu_hat), check_finite=False)

# Predictive mean at training inputs: mu + K * K^{-1} (y - mu)
pred_mu = mu_hat + K @ alpha

# Predictive variance diagonal at training inputs: diag(K - K K^{-1} K)
# (this is the latent function variance; it does not include obs noise)
v = cho_solve((c, lower), K, check_finite=False)
pred_var = np.clip(np.diag(K - K @ v), 0.0, np.inf)
pred_std = np.sqrt(pred_var)

# -----------------------
# 7) Save a quick plot
# -----------------------
ii = np.argsort(R)
plt.figure(figsize=(7, 4))
plt.errorbar(R, y, yerr=sigma_obs, fmt=".", capsize=2, label="data")
plt.plot(R[ii], pred_mu[ii], lw=2, label="mean+GP")
plt.fill_between(R[ii], (pred_mu - pred_std)[ii], (pred_mu + pred_std)[ii],
                 alpha=0.20, label="±1σ (latent)")
plt.xlabel("R [kpc]")
plt.ylabel("Vφ [km/s]")
plt.title("Rotation curve with residual GP (flat mean)")
plt.legend()
plt.tight_layout()
out_png = Path("data/rotation_curve/beordo2024/rc_gp_fit.png")
plt.savefig(out_png, dpi=160)
print(f"Saved plot: {out_png}")
