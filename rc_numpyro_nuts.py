#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

# ----------------- Data -----------------
csv = Path("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
df = pd.read_csv(csv).sort_values("R_kpc").reset_index(drop=True)
R = jnp.asarray(df["R_kpc"].to_numpy(float))
y = jnp.asarray(df["Vphi_kms"].to_numpy(float))
sigma_obs = jnp.asarray(df["sigma_obs_kms"].to_numpy(float))
N = R.size

G = 4.300917270e-6  # (kpc km^2 / s^2) / Msun

# ----------------- JAX helpers -----------------
def trapz_jax(y, x):
    """JAX-safe trapezoid integral ∫ y dx."""
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))

# ----------------- JAX-native physics -----------------
def vc2_disk(R, M, a, b):
    B = a + b
    return G * M * (R*R) / (R*R + B*B)**1.5

def vc2_bulge(R, M, a):
    return G * M * R / (R + a)**2

def rho_gnfw(r, rho0, rs, gamma):
    x = r / rs
    return rho0 * (x**(-gamma)) * (1.0 + x)**(gamma - 3.0)

def mass_enclosed_gnfw(R, rho0, rs, gamma):
    # Vectorized integral M(<R): 4 pi ∫ rho(r) r^2 dr
    def m_of_R(Ri):
        Ri = jnp.maximum(Ri, 1e-6)
        x = jnp.linspace(0.0, Ri, 2048, dtype=Ri.dtype)
        x = x.at[0].set(1e-9)
        y = rho_gnfw(x, rho0, rs, gamma) * x * x
        return 4.0 * jnp.pi * trapz_jax(y, x)
    return jax.vmap(m_of_R)(R)

def vc2_halo(R, rho0, rs, gamma):
    Menc = mass_enclosed_gnfw(R, rho0, rs, gamma)
    return G * Menc / jnp.clip(R, 1e-6, None)

def vphi_model(R, Md, a_d, b_d, Mb, a_b, rho0, rs, gamma):
    vc2 = vc2_disk(R, Md, a_d, b_d) + vc2_bulge(R, Mb, a_b) + vc2_halo(R, rho0, rs, gamma)
    return jnp.sqrt(vc2)

# ----------------- GP kernel -----------------
def matern32_matrix(x, amp, scale):
    dx = jnp.abs(x[:, None] - x[None, :])
    z = jnp.sqrt(3.0) * dx / scale
    return (amp**2) * (1.0 + z) * jnp.exp(-z)

# Effective gNFW slope at radius r (for plotting)
def gamma_eff_gnfw(r, rs, gamma):
    x = r/rs
    return gamma + (3.0 - gamma) * (x/(1.0 + x))

# ----------------- NumPyro model -----------------
def model(R, y, sigma_obs):
    # Priors
    logMd  = numpyro.sample("logMd",  dist.Uniform(jnp.log(1e9), jnp.log(3e11)))
    a_d    = numpyro.sample("a_d",    dist.Uniform(2.0, 10.0))
    b_d    = numpyro.sample("b_d",    dist.Uniform(0.05, 1.5))

    logMb  = numpyro.sample("logMb",  dist.Uniform(jnp.log(1e8), jnp.log(5e10)))
    a_b    = numpyro.sample("a_b",    dist.Uniform(0.1, 2.0))

    logrho0= numpyro.sample("logrho0",dist.Uniform(jnp.log(1e5), jnp.log(1e9)))
    rs     = numpyro.sample("rs",     dist.Uniform(5.0, 50.0))
    gamma  = numpyro.sample("gamma",  dist.Uniform(0.2, 1.8))

    log_amp   = numpyro.sample("log_amp",   dist.Uniform(jnp.log(0.5), jnp.log(50.0)))
    log_scale = numpyro.sample("log_scale", dist.Uniform(jnp.log(0.3), jnp.log(8.0)))
    log_jit   = numpyro.sample("log_jit",   dist.Uniform(jnp.log(0.05), jnp.log(20.0)))

    Md, Mb, rho0 = jnp.exp(logMd), jnp.exp(logMb), jnp.exp(logrho0)
    amp, scale, jitter = jnp.exp(log_amp), jnp.exp(log_scale), jnp.exp(log_jit)

    mu = vphi_model(R, Md, a_d, b_d, Mb, a_b, rho0, rs, gamma)

    K = matern32_matrix(R, amp, scale)
    K = K.at[jnp.diag_indices(R.shape[0])].add(sigma_obs**2 + jitter**2)

    numpyro.sample("y", dist.MultivariateNormal(loc=mu, covariance_matrix=K), obs=y)

# ----------------- Run NUTS, save figures -----------------
def main():
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(1)

    nuts = NUTS(model, init_strategy=init_to_median())
    mcmc = MCMC(nuts, num_warmup=800, num_samples=1200, num_chains=1, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0), R, y, sigma_obs)
    posterior = mcmc.get_samples()
    Path("results").mkdir(parents=True, exist_ok=True)
    az.summary(posterior).to_csv("results/rc_nuts_summary.csv")

    # Predict ribbon for the physics mean only
    nS = min(400, posterior["logMd"].shape[0])
    idx = np.random.choice(posterior["logMd"].shape[0], nS, replace=False)
    curves = []
    for i in idx:
        Md   = float(jnp.exp(posterior["logMd"][i]))
        a_d  = float(posterior["a_d"][i])
        b_d  = float(posterior["b_d"][i])
        Mb   = float(jnp.exp(posterior["logMb"][i]))
        a_b  = float(posterior["a_b"][i])
        rho0 = float(jnp.exp(posterior["logrho0"][i]))
        rs   = float(posterior["rs"][i])
        gam  = float(posterior["gamma"][i])
        curves.append(np.array(vphi_model(R, Md, a_d, b_d, Mb, a_b, rho0, rs, gam)))
    curves = np.asarray(curves)
    lo, med, hi = np.percentile(curves, [16, 50, 84], axis=0)

    ii = np.argsort(np.asarray(R))
    plt.figure(figsize=(7,4))
    plt.errorbar(np.asarray(R), np.asarray(y), yerr=np.asarray(sigma_obs), fmt=".", capsize=2, alpha=0.7, label="data")
    plt.fill_between(np.asarray(R)[ii], lo[ii], hi[ii], color="tab:orange", alpha=0.25, label="model 68%")
    plt.plot(np.asarray(R)[ii], med[ii], color="tab:orange", lw=2, label="model median")
    plt.xlabel("R [kpc]"); plt.ylabel("Vφ [km/s]"); plt.title("RC: physical mean (Bayesian) — 68% credible band")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/rc_nuts_ribbon.png", dpi=160)

    # gamma_eff(r) ribbon
    rgrid = np.linspace(5.0, 20.0, 200)
    ge = []
    for i in idx:
        rs_  = float(posterior["rs"][i])
        gam_ = float(posterior["gamma"][i])
        ge.append(gamma_eff_gnfw(rgrid, rs_, gam_))
    ge = np.asarray(ge)
    ge_lo, ge_med, ge_hi = np.percentile(ge, [16,50,84], axis=0)
    plt.figure(figsize=(6.4,3.8))
    plt.fill_between(rgrid, ge_lo, ge_hi, alpha=0.25, label="68% credible")
    plt.plot(rgrid, ge_med, lw=2, label="median")
    plt.xlabel("R [kpc]"); plt.ylabel(r"$\gamma_{\rm eff}(r)$")
    plt.title(r"Cuspiness $\gamma_{\rm eff}(r)$ — gNFW posterior")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/rc_gammaeff_ribbon.png", dpi=160)

    print("Saved files:\n  results/rc_nuts_summary.csv\n  results/rc_nuts_ribbon.png\n  results/rc_gammaeff_ribbon.png")

if __name__ == "__main__":
    main()
