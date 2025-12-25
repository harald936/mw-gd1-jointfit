import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_qzA_grid(path):
    d = np.load(path)
    qz = np.array(d["qz"], copy=True)
    A = np.array(d["A"], copy=True)
    if "posterior" in d.files:
        p = np.array(d["posterior"], copy=True)
    else:
        arrays_2d = []
        for k in d.files:
            v = d[k]
            if v.ndim == 2 and v.shape == (qz.size, A.size):
                arrays_2d.append(np.array(v, copy=True))
        if not arrays_2d:
            raise RuntimeError("No 2D posterior array in grid file")
        p = arrays_2d[0]
    p = np.clip(p, 0.0, np.inf)
    s = p.sum()
    if s <= 0.0:
        raise RuntimeError("qzA posterior sum is nonpositive")
    p /= s
    return qz, A, p


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
        if "mean" in lc and "mcse" not in lc:
            mean_col = c
        if ("sd" in lc or "sigma" in lc) and "mcse" not in lc:
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


def sample_from_2d_grid(qz_grid, A_grid, p_grid, n_samp, rng):
    flat_p = p_grid.ravel()
    flat_p = flat_p / flat_p.sum()
    idx = rng.choice(flat_p.size, size=n_samp, replace=True, p=flat_p)
    iq = idx // A_grid.size
    iA = idx % A_grid.size
    qz = qz_grid[iq]
    A = A_grid[iA]
    return qz, A


def main():
    repo = Path(".").resolve()
    grid_path = repo / "results" / "qzA_posterior_joint_physicalrc_refined_grid.npz"
    if not grid_path.exists():
        grid_path = repo / "results" / "qzA_posterior_joint_physicalrc_grid.npz"
    if not grid_path.exists():
        raise RuntimeError("No qzA posterior grid file found in results/")
    qz_grid, A_grid, p_qzA = load_qzA_grid(str(grid_path))
    rc_summary = repo / "results" / "rc_nuts_summary.csv"
    if not rc_summary.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")
    gamma_mu, gamma_sigma = load_gamma_from_rc_summary(str(rc_summary))
    rng = np.random.default_rng(12345)
    n_samp = 20000
    qz_s, A_s = sample_from_2d_grid(qz_grid, A_grid, p_qzA, n_samp, rng)
    gamma_s = rng.normal(loc=gamma_mu, scale=gamma_sigma, size=n_samp)
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    idx = np.arange(n_samp)
    if n_samp > 5000:
        idx = rng.choice(n_samp, size=5000, replace=False)
    ax.scatter(qz_s[idx], A_s[idx], gamma_s[idx], s=3, alpha=0.35)
    ax.set_xlabel("q_z")
    ax.set_ylabel("A")
    ax.set_zlabel("gamma")
    ax.set_title("Samples from 3D posterior p(q_z, A, gamma)")
    fig.tight_layout()
    pdf_3d = out / "qzAgamma_posterior_3d_scatter.pdf"
    png_3d = out / "qzAgamma_posterior_3d_scatter.png"
    fig.savefig(pdf_3d)
    fig.savefig(png_3d, dpi=200)
    fig2, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    m = 4000
    if n_samp > m:
        idx2 = rng.choice(n_samp, size=m, replace=False)
    else:
        idx2 = np.arange(n_samp)
    axes[0].scatter(qz_s[idx2], A_s[idx2], s=3, alpha=0.4)
    axes[0].set_xlabel("q_z")
    axes[0].set_ylabel("A")
    axes[0].set_title("q_z vs A")
    axes[1].scatter(qz_s[idx2], gamma_s[idx2], s=3, alpha=0.4)
    axes[1].set_xlabel("q_z")
    axes[1].set_ylabel("gamma")
    axes[1].set_title("q_z vs gamma")
    axes[2].scatter(A_s[idx2], gamma_s[idx2], s=3, alpha=0.4)
    axes[2].set_xlabel("A")
    axes[2].set_ylabel("gamma")
    axes[2].set_title("A vs gamma")
    fig2.tight_layout()
    pdf_2d = out / "qzAgamma_posterior_2d_projections.pdf"
    png_2d = out / "qzAgamma_posterior_2d_projections.png"
    fig2.savefig(pdf_2d)
    fig2.savefig(png_2d, dpi=200)
    with open(out / "qzAgamma_posterior_3d_summary.txt", "w") as f:
        f.write("3D posterior samples p(q_z, A, gamma)\n")
        f.write(f"q_z: mean {np.mean(qz_s):.5f}, std {np.std(qz_s):.5f}\n")
        f.write(f"A:   mean {np.mean(A_s):.5f}, std {np.std(A_s):.5f}\n")
        f.write(f"gamma: mean {np.mean(gamma_s):.5f}, std {np.std(gamma_s):.5f}\n")
        f.write(f"gamma prior (from RC): mu {gamma_mu:.5f}, sigma {gamma_sigma:.5f}\n")
    print(str(pdf_3d))
    print(str(pdf_2d))
    print(str(out / "qzAgamma_posterior_3d_summary.txt"))


if __name__ == "__main__":
    main()
