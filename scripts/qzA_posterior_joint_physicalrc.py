import numpy as np
import matplotlib.pyplot as plt


def load_joint_grid(path):
    d = np.load(path)
    arrays_1d = []
    arrays_2d = []
    for k in d.files:
        v = d[k]
        if v.ndim == 1 and v.size > 3:
            arrays_1d.append(v)
        elif v.ndim == 2:
            arrays_2d.append(v)
    if len(arrays_1d) < 2:
        raise RuntimeError("Could not find two 1D arrays in grid file")
    qz = np.array(arrays_1d[0], copy=True)
    A = np.array(arrays_1d[1], copy=True)
    if len(arrays_2d) == 0:
        raise RuntimeError("Could not find a 2D array in grid file")
    grid = np.array(arrays_2d[0], copy=True)
    if grid.shape != (qz.size, A.size):
        if grid.shape == (A.size, qz.size):
            grid = grid.T
        else:
            raise RuntimeError("2D array shape does not match 1D grids")
    return qz, A, grid


def grid_to_posterior(grid):
    x = np.array(grid, dtype=float)
    mask = np.isfinite(x)
    if not np.any(mask):
        raise RuntimeError("No finite entries in input grid")
    x[~mask] = 0.0
    s = x.sum()
    if s <= 0.0:
        raise RuntimeError("Posterior sum is nonpositive")
    x /= s
    return x


def refine_posterior(qz, A, p, nq=200, nA=200):
    qz_fine = np.linspace(qz.min(), qz.max(), nq)
    A_fine = np.linspace(A.min(), A.max(), nA)
    p_step = np.empty((qz.size, nA))
    for i in range(qz.size):
        p_step[i] = np.interp(A_fine, A, p[i])
    p_fine = np.empty((nq, nA))
    for j in range(nA):
        p_fine[:, j] = np.interp(qz_fine, qz, p_step[:, j])
    p_fine[p_fine < 0.0] = 0.0
    s = p_fine.sum()
    if s > 0.0:
        p_fine /= s
    return qz_fine, A_fine, p_fine


def marginal_and_intervals(p_grid, axis, coords):
    if axis == 0:
        marg = p_grid.sum(axis=1)
    else:
        marg = p_grid.sum(axis=0)
    total = marg.sum()
    if total <= 0.0:
        raise RuntimeError("Marginal distribution sum is nonpositive")
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


def correlation_qz_A(qz, A, p_grid):
    qz_vec = qz[:, None]
    A_vec = A[None, :]
    mean_qz = (p_grid * qz_vec).sum()
    mean_A = (p_grid * A_vec).sum()
    dqz = qz_vec - mean_qz
    dA = A_vec - mean_A
    cov = (p_grid * dqz * dA).sum()
    var_qz = (p_grid * dqz * dqz).sum()
    var_A = (p_grid * dA * dA).sum()
    if var_qz <= 0.0 or var_A <= 0.0:
        return 0.0
    return cov / np.sqrt(var_qz * var_A)


def main():
    base_grid_path = "results/qzA_posterior_joint_physicalrc_grid.npz"
    qz, A, grid = load_joint_grid(base_grid_path)
    p_raw = grid_to_posterior(grid)
    qz_f, A_f, p_f = refine_posterior(qz, A, p_raw, nq=200, nA=200)
    marg_qz, qz_68, qz_95 = marginal_and_intervals(p_f, axis=0, coords=qz_f)
    marg_A, A_68, A_95 = marginal_and_intervals(p_f, axis=1, coords=A_f)
    level_68, level_95 = hpd_levels(p_f, probs=(0.68, 0.95))
    rho = correlation_qz_A(qz_f, A_f, p_f)
    out_grid_path = "results/qzA_posterior_joint_physicalrc_refined_grid.npz"
    np.savez(out_grid_path, qz=qz_f, A=A_f, posterior=p_f)
    dqz = qz_95[1] - qz_95[0]
    dA = A_95[1] - A_95[0]
    qz_lo = max(qz_f.min(), qz_95[0] - 0.5 * dqz)
    qz_hi = min(qz_f.max(), qz_95[1] + 0.5 * dqz)
    A_lo = max(A_f.min(), A_95[0] - 0.5 * dA)
    A_hi = min(A_f.max(), A_95[1] + 0.5 * dA)
    fig, ax = plt.subplots(figsize=(5, 4))
    cf = ax.contourf(A_f, qz_f, p_f, levels=40)
    ax.contour(A_f, qz_f, p_f, levels=[level_68], colors="white", linewidths=1.5)
    ax.contour(A_f, qz_f, p_f, levels=[level_95], colors="white", linewidths=1.0, linestyles="dashed")
    AA, QQ = np.meshgrid(A, qz)
    ax.scatter(AA.ravel(), QQ.ravel(), s=5, c="k", alpha=0.3)
    ax.set_xlim(A_lo, A_hi)
    ax.set_ylim(qz_lo, qz_hi)
    ax.set_xlabel("A")
    ax.set_ylabel("q_z")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("p(q_z, A)")
    fig.tight_layout()
    fig.savefig("results/qzA_posterior_joint_physicalrc_refined.pdf")
    fig.savefig("results/qzA_posterior_joint_physicalrc_refined.png", dpi=200)
    plt.close(fig)
    with open("results/qzA_posterior_joint_physicalrc_refined.txt", "w") as f:
        f.write("Refined joint posterior for q_z and A using existing grid\n")
        f.write("q_z median {:.6f}, 68% interval [{:.6f}, {:.6f}], 95% interval [{:.6f}, {:.6f}]\n".format(qz_68[1], qz_68[0], qz_68[2], qz_95[0], qz_95[1]))
        f.write("A   median {:.6f}, 68% interval [{:.6f}, {:.6f}], 95% interval [{:.6f}, {:.6f}]\n".format(A_68[1], A_68[0], A_68[2], A_95[0], A_95[1]))
        f.write("Corr(q_z, A) {:.6f}\n".format(rho))


if __name__ == "__main__":
    main()
