import numpy as np
import matplotlib.pyplot as plt
import os


def load_refined_grid(path):
    d = np.load(path)
    qz = np.array(d["qz"], copy=True)
    A = np.array(d["A"], copy=True)
    p = np.array(d["posterior"], copy=True)
    if p.shape != (qz.size, A.size):
        if p.shape == (A.size, qz.size):
            p = p.T
        else:
            raise RuntimeError("posterior shape does not match qz, A grids")
    return qz, A, p


def marginal_qz(p_grid, qz):
    marg = p_grid.sum(axis=1)
    total = marg.sum()
    if total <= 0.0:
        raise RuntimeError("q_z marginal sum is nonpositive")
    marg /= total
    cdf = np.cumsum(marg)
    cdf /= cdf[-1]
    def q(x):
        return np.interp(x, cdf, qz)
    q16 = q(0.158655)
    q50 = q(0.5)
    q84 = q(0.841345)
    q2_5 = q(0.025)
    q97_5 = q(0.975)
    return marg, (q16, q50, q84), (q2_5, q97_5)


def posterior_to_delta_chi2(qz, p):
    mask = p > 0.0
    if not np.any(mask):
        raise RuntimeError("no positive posterior values for q_z")
    logp = np.full_like(p, -np.inf, dtype=float)
    logp[mask] = np.log(p[mask])
    m = np.max(logp[mask])
    dchi2 = -2.0 * (logp - m)
    dchi2[~mask] = np.nan
    return dchi2


def load_old_profile(path):
    if not os.path.exists(path):
        return None, None
    try:
        arr = np.loadtxt(path, comments="#", skiprows=1)
    except Exception:
        return None, None
    if arr.ndim == 1:
        if arr.size < 2:
            return None, None
        qz = arr[0:1]
        prof = arr[1:2]
    else:
        qz = arr[:, 0]
        prof = arr[:, 1]
    prof = prof - prof.min()
    p = np.exp(-0.5 * prof)
    s = p.sum()
    if s <= 0.0:
        return None, None
    p /= s
    return qz, p


def main():
    qz_refined, A_refined, p_2d = load_refined_grid("results/qzA_posterior_joint_physicalrc_refined_grid.npz")
    p_qz, qz_68, qz_95 = marginal_qz(p_2d, qz_refined)
    dchi2_new = posterior_to_delta_chi2(qz_refined, p_qz)
    qz_old, p_old = load_old_profile("results/joint_qz_physicalrc.txt")
    fig, ax = plt.subplots()
    ax.plot(qz_refined, p_qz, label="From 2D posterior")
    if qz_old is not None and p_old is not None:
        ax.plot(qz_old, p_old, linestyle="--", label="Existing joint_qz_physicalrc")
    ax.set_xlabel("q_z")
    ax.set_ylabel("normalized posterior")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/joint_qz_from_qzA_posterior.pdf")
    fig.savefig("results/joint_qz_from_qzA_posterior.png", dpi=200)
    plt.close(fig)
    out_tab = np.column_stack([qz_refined, dchi2_new])
    header = "qz delta_chi2_from_qzA_posterior"
    np.savetxt("results/joint_qz_from_qzA_posterior.txt", out_tab, header=header)
    with open("results/joint_qz_from_qzA_posterior_summary.txt", "w") as f:
        f.write("1D q_z posterior derived by marginalizing A from refined 2D q_z–A grid\n")
        f.write("q_z median {:.6f}, 68% interval [{:.6f}, {:.6f}], 95% interval [{:.6f}, {:.6f}]\n".format(qz_68[1], qz_68[0], qz_68[2], qz_95[0], qz_95[1]))


if __name__ == "__main__":
    main()
