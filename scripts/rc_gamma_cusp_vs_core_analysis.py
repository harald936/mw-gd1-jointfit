import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_gamma_from_rc_summary(path):
    df = pd.read_csv(path)
    name_col = df.columns[0]
    mask = df[name_col].astype(str) == "gamma"
    if not mask.any():
        raise RuntimeError("No gamma row found in rc_nuts_summary.csv")
    row = df.loc[mask].iloc[0]
    mean_col = None
    sd_col = None
    for c in df.columns:
        lc = c.lower()
        if "mean" in lc:
            mean_col = c
        if "sd" in lc or "sigma" in lc:
            sd_col = c
    if mean_col is None or sd_col is None:
        if df.shape[1] >= 3:
            mean_col = df.columns[1]
            sd_col = df.columns[2]
        else:
            raise RuntimeError("Could not identify mean/sd columns for gamma")
    mu = float(row[mean_col])
    sigma = float(row[sd_col])
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("Invalid gamma mean/sd in rc_nuts_summary.csv")
    return mu, sigma


def gaussian_posterior_grid(mu, sigma, n=2000, nsig=5.0):
    g_min = mu - nsig * sigma
    g_max = mu + nsig * sigma
    gamma = np.linspace(g_min, g_max, n)
    p = np.exp(-0.5 * ((gamma - mu) / sigma) ** 2)
    p /= p.sum()
    return gamma, p


def prob_interval(gamma, p, g_lo, g_hi):
    mask = (gamma >= g_lo) & (gamma <= g_hi)
    return float(p[mask].sum())


def prob_greater(gamma, p, thresh):
    mask = gamma >= thresh
    return float(p[mask].sum())


def prob_less(gamma, p, thresh):
    mask = gamma <= thresh
    return float(p[mask].sum())


def credible_interval(gamma, p, alpha_low=0.16, alpha_high=0.84):
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    lo = np.interp(alpha_low, cdf, gamma)
    hi = np.interp(alpha_high, cdf, gamma)
    return float(lo), float(hi)


def main():
    repo = Path(".").resolve()
    rc_summary = repo / "results" / "rc_nuts_summary.csv"
    if not rc_summary.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")
    mu, sigma = load_gamma_from_rc_summary(str(rc_summary))
    gamma, p = gaussian_posterior_grid(mu, sigma)
    core_width = 0.2
    p_core = prob_interval(gamma, p, -core_width, core_width)
    p_cusp_gt1 = prob_greater(gamma, p, 1.0)
    p_gamma_gt0 = prob_greater(gamma, p, 0.0)
    p_gamma_lt0 = prob_less(gamma, p, 0.0)
    lo68, hi68 = credible_interval(gamma, p, 0.16, 0.84)
    lo95, hi95 = credible_interval(gamma, p, 0.025, 0.975)
    z_core = abs(mu - 0.0) / sigma
    out = repo / "results"
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gamma, p, lw=1.8)
    ax.axvline(0.0, ls="--", lw=1.0, color="k")
    ax.axvline(1.0, ls=":", lw=1.0, color="k")
    ax.axvspan(-core_width, core_width, alpha=0.15)
    ax.set_xlabel(r"$\gamma$ (inner DM slope)")
    ax.set_ylabel(r"$p(\gamma | \mathrm{RC})$")
    ax.set_title("RC-only posterior for inner slope $\\gamma$ (gNFW halo + GP)")
    fig.tight_layout()
    pdf = out / "rc_gamma_posterior_cusp_vs_core.pdf"
    png = out / "rc_gamma_posterior_cusp_vs_core.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    txt = out / "rc_gamma_cusp_vs_core_summary.txt"
    with open(txt, "w") as f:
        f.write("RC-only gamma posterior (Gaussian approximation from rc_nuts_summary.csv)\n")
        f.write("mu {:.6f}, sigma {:.6f}\n".format(mu, sigma))
        f.write("68% CI [{:.6f}, {:.6f}], 95% CI [{:.6f}, {:.6f}]\n".format(lo68, hi68, lo95, hi95))
        f.write("P(|gamma| <= {:.3f}) ~ {:.4f} (core-like region)\n".format(core_width, p_core))
        f.write("P(gamma > 1.0) ~ {:.4f} (strongly cuspy)\n".format(p_cusp_gt1))
        f.write("P(gamma > 0.0) ~ {:.4f}, P(gamma < 0.0) ~ {:.4f}\n".format(p_gamma_gt0, p_gamma_lt0))
        f.write("Core (gamma=0) is disfavoured at ~{:.2f} sigma based on mu/sigma\n".format(z_core))


if __name__ == "__main__":
    main()
