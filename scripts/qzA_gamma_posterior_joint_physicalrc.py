import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
            if v.ndim == 2:
                arrays_2d.append(v)
        if not arrays_2d:
            raise RuntimeError("No 2D array found in qzA grid file")
        p = np.array(arrays_2d[0], copy=True)
    if p.shape != (qz.size, A.size):
        if p.shape == (A.size, qz.size):
            p = p.T
        else:
            raise RuntimeError("posterior shape does not match qz, A grids")
    mask = np.isfinite(p)
    p[~mask] = 0.0
    s = p.sum()
    if s <= 0.0:
        raise RuntimeError("qzA posterior sum is nonpositive")
    p /= s
    return qz, A, p


def load_gamma_from_rc_summary(path):
    df = pd.read_csv(path)
    name_col = df.columns[0]
    if name_col.lower().startswith("unnamed"):
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
    g_min = max(0.0, mu - 4.0 * sigma)
    g_max = mu + 4.0 * sigma
    gamma_grid = np.linspace(g_min, g_max, 200)
    p_gamma = np.exp(-0.5 * ((gamma_grid - mu) / sigma) ** 2)
    s = p_gamma.sum()
    if s <= 0.0:
        raise RuntimeError("Gamma posterior sum is nonpositive")
    p_gamma /= s
    return gamma_grid, p_gamma, mu, sigma


def marginal_1d_from_3d(qz, A, gamma, p3, axis):
    if axis == 0:
        marg = p3.sum(axis=(1, 2))
        coords = qz
    elif axis == 1:
        marg = p3.sum(axis=(0, 2))
        coords = A
    else:
        marg = p3.sum(axis=(0, 1))
        coords = gamma
    total = marg.sum()
    if total <= 0.0:
        raise RuntimeError("Marginal sum is nonpositive")
    marg /= total
    cdf = np.cumsum(marg)
    cdf /= cdf[-1]
    def q(x):
        return np.interp(x, cdf, coords)
    q16 = q(0.158655)
    q50 = q(0.5)
    q84 = q(0.841345)
    q2_5 = q(0.025)
    q97_5 = q(0.975)
    return marg, (q16, q50, q84), (q2_5, q97_5)


def hpd_levels(p_grid, probs=(0.68, 0.95)):
    flat = p_grid.ravel()
    order = np.argsort(flat)[::-1]
    flat_sorted = flat[order]
    csum = np.cumsum(flat_sorted)
    csum /= csum[-1]
    levels = []
    for p in probs:
        idx = np.searchsorted(csum, p)
        if idx >= flat_sorted.size:
            idx = flat_sorted.size - 1
        levels.append(flat_sorted[idx])
    return tuple(levels)


def main():
    qz, A, p_qzA = load_qzA_grid("results/qzA_posterior_joint_physicalrc_refined_grid.npz")
    gamma, p_gamma, gamma_mu, gamma_sigma = load_gamma_from_rc_summary("results/rc_nuts_summary.csv")
    p3 = p_qzA[:, :, None] * p_gamma[None, None, :]
    s = p3.sum()
    if s <= 0.0:
        raise RuntimeError("3D posterior sum is nonpositive")
    p3 /= s
    marg_qz, qz_68, qz_95 = marginal_1d_from_3d(qz, A, gamma, p3, axis=0)
    marg_A, A_68, A_95 = marginal_1d_from_3d(qz, A, gamma, p3, axis=1)
    marg_g, g_68, g_95 = marginal_1d_from_3d(qz, A, gamma, p3, axis=2)
    qzA_marg = p3.sum(axis=2)
    qzg_marg = p3.sum(axis=1)
    Ag_marg = p3.sum(axis=0)
    level_68_qzA, level_95_qzA = hpd_levels(qzA_marg, probs=(0.68, 0.95))
    level_68_qzg, level_95_qzg = hpd_levels(qzg_marg, probs=(0.68, 0.95))
    level_68_Ag, level_95_Ag = hpd_levels(Ag_marg, probs=(0.68, 0.95))
    np.savez(
        "results/qzA_gamma_posterior_joint_physicalrc_factored_grid.npz",
        qz=qz,
        A=A,
        gamma=gamma,
        posterior=p3,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    A_grid, qz_grid = np.meshgrid(A, qz)
    g_grid_qz, qz_grid_g = np.meshgrid(gamma, qz)
    g_grid_A, A_grid_g = np.meshgrid(gamma, A)
    ax = axes[0]
    ax.contourf(A_grid, qz_grid, qzA_marg, levels=40)
    ax.contour(A_grid, qz_grid, qzA_marg, levels=[level_68_qzA], colors="white", linewidths=1.5)
    ax.contour(A_grid, qz_grid, qzA_marg, levels=[level_95_qzA], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("A")
    ax.set_ylabel("q_z")
    axes[0].set_title("p(q_z, A)")
    ax = axes[1]
    ax.contourf(g_grid_qz, qz_grid_g, qzg_marg, levels=40)
    ax.contour(g_grid_qz, qz_grid_g, qzg_marg, levels=[level_68_qzg], colors="white", linewidths=1.5)
    ax.contour(g_grid_qz, qz_grid_g, qzg_marg, levels=[level_95_qzg], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("gamma")
    ax.set_ylabel("q_z")
    axes[1].set_title("p(q_z, gamma)")
    ax = axes[2]
    ax.contourf(g_grid_A, A_grid_g, Ag_marg, levels=40)
    ax.contour(g_grid_A, A_grid_g, Ag_marg, levels=[level_68_Ag], colors="white", linewidths=1.5)
    ax.contour(g_grid_A, A_grid_g, Ag_marg, levels=[level_95_Ag], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("gamma")
    ax.set_ylabel("A")
    axes[2].set_title("p(A, gamma)")
    fig.tight_layout()
    fig.savefig("results/qzA_gamma_posterior_joint_physicalrc_corner.pdf")
    fig.savefig("results/qzA_gamma_posterior_joint_physicalrc_corner.png", dpi=200)
    plt.close(fig)
    with open("results/qzA_gamma_posterior_joint_physicalrc_summary.txt", "w") as f:
        f.write("3D factored posterior p(q_z, A, gamma | data) = p(q_z, A | GD-1+RC) * p(gamma | RC)\n")
        f.write("q_z median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(qz_68[1], qz_68[0], qz_68[2], qz_95[0], qz_95[1]))
        f.write("A   median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(A_68[1], A_68[0], A_68[2], A_95[0], A_95[1]))
        f.write("gamma median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(g_68[1], g_68[0], g_68[2], g_95[0], g_95[1]))
        f.write("RC-only gamma mean {:.6f}, sd {:.6f} (from rc_nuts_summary.csv)\n".format(gamma_mu, gamma_sigma))


if __name__ == "__main__":
    main()
