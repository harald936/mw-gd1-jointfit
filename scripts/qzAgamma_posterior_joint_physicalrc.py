import numpy as np
import matplotlib.pyplot as plt


def joint_loglike(qz, A, gamma):
    raise RuntimeError("Implement joint_loglike(qz, A, gamma) using your existing GD-1+RC likelihood")


def make_grids():
    qz_grid = np.linspace(1.02, 1.16, 30)
    A_grid = np.linspace(0.98, 1.12, 30)
    gamma_grid = np.linspace(0.2, 2.2, 30)
    return qz_grid, A_grid, gamma_grid


def build_loglike_grid(qz_grid, A_grid, gamma_grid):
    nq = qz_grid.size
    nA = A_grid.size
    ng = gamma_grid.size
    logL = np.empty((nq, nA, ng), dtype=float)
    for i, qz in enumerate(qz_grid):
        for j, A in enumerate(A_grid):
            for k, g in enumerate(gamma_grid):
                logL[i, j, k] = joint_loglike(qz, A, g)
    return logL


def logL_to_posterior(logL):
    m = np.nanmax(logL)
    x = np.exp(logL - m)
    x[~np.isfinite(x)] = 0.0
    s = x.sum()
    if s <= 0.0:
        raise RuntimeError("Posterior sum is nonpositive")
    x /= s
    return x


def marginal_1d(p3, axis, coords):
    if axis == 0:
        marg = p3.sum(axis=(1, 2))
    elif axis == 1:
        marg = p3.sum(axis=(0, 2))
    else:
        marg = p3.sum(axis=(0, 1))
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
    qz_grid, A_grid, gamma_grid = make_grids()
    logL_grid = build_loglike_grid(qz_grid, A_grid, gamma_grid)
    p3 = logL_to_posterior(logL_grid)
    marg_qz, qz_68, qz_95 = marginal_1d(p3, axis=0, coords=qz_grid)
    marg_A, A_68, A_95 = marginal_1d(p3, axis=1, coords=A_grid)
    marg_g, g_68, g_95 = marginal_1d(p3, axis=2, coords=gamma_grid)
    qzA_marg = p3.sum(axis=2)
    qzg_marg = p3.sum(axis=1)
    Ag_marg = p3.sum(axis=0)
    level_68_qzA, level_95_qzA = hpd_levels(qzA_marg, probs=(0.68, 0.95))
    level_68_qzg, level_95_qzg = hpd_levels(qzg_marg, probs=(0.68, 0.95))
    level_68_Ag, level_95_Ag = hpd_levels(Ag_marg, probs=(0.68, 0.95))
    np.savez(
        "results/qzAgamma_posterior_joint_physicalrc_grid.npz",
        qz=qz_grid,
        A=A_grid,
        gamma=gamma_grid,
        logL=logL_grid,
        posterior=p3,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    A_mesh, qz_mesh = np.meshgrid(A_grid, qz_grid)
    g_mesh_qz, qz_mesh_g = np.meshgrid(gamma_grid, qz_grid)
    g_mesh_A, A_mesh_g = np.meshgrid(gamma_grid, A_grid)
    ax = axes[0]
    cf0 = ax.contourf(A_mesh, qz_mesh, qzA_marg, levels=40)
    ax.contour(A_mesh, qz_mesh, qzA_marg, levels=[level_68_qzA], colors="white", linewidths=1.5)
    ax.contour(A_mesh, qz_mesh, qzA_marg, levels=[level_95_qzA], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("A")
    ax.set_ylabel("q_z")
    axes[0].set_title("p(q_z, A)")
    ax = axes[1]
    cf1 = ax.contourf(g_mesh_qz, qz_mesh_g, qzg_marg, levels=40)
    ax.contour(g_mesh_qz, qz_mesh_g, qzg_marg, levels=[level_68_qzg], colors="white", linewidths=1.5)
    ax.contour(g_mesh_qz, qz_mesh_g, qzg_marg, levels=[level_95_qzg], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("gamma")
    ax.set_ylabel("q_z")
    axes[1].set_title("p(q_z, gamma)")
    ax = axes[2]
    cf2 = ax.contourf(g_mesh_A, A_mesh_g, Ag_marg, levels=40)
    ax.contour(g_mesh_A, A_mesh_g, Ag_marg, levels=[level_68_Ag], colors="white", linewidths=1.5)
    ax.contour(g_mesh_A, A_mesh_g, Ag_marg, levels=[level_95_Ag], colors="white", linewidths=1.0, linestyles="dashed")
    ax.set_xlabel("gamma")
    ax.set_ylabel("A")
    axes[2].set_title("p(A, gamma)")
    fig.tight_layout()
    fig.savefig("results/qzAgamma_posterior_joint_physicalrc_corner.pdf")
    fig.savefig("results/qzAgamma_posterior_joint_physicalrc_corner.png", dpi=200)
    plt.close(fig)
    with open("results/qzAgamma_posterior_joint_physicalrc_summary.txt", "w") as f:
        f.write("3D joint posterior p(q_z, A, gamma | GD-1 + RC)\n")
        f.write("q_z median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(qz_68[1], qz_68[0], qz_68[2], qz_95[0], qz_95[1]))
        f.write("A   median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(A_68[1], A_68[0], A_68[2], A_95[0], A_95[1]))
        f.write("gamma median {:.6f}, 68% [{:.6f}, {:.6f}], 95% [{:.6f}, {:.6f}]\n".format(g_68[1], g_68[0], g_68[2], g_95[0], g_95[1]))


if __name__ == "__main__":
    main()
