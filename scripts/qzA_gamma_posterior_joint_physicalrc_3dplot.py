import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_posterior(path):
    d = np.load(path)
    qz = np.array(d["qz"], copy=True)
    A = np.array(d["A"], copy=True)
    gamma = np.array(d["gamma"], copy=True)
    p3 = np.array(d["posterior"], copy=True)
    if p3.shape != (qz.size, A.size, gamma.size):
        raise RuntimeError("posterior shape does not match qz, A, gamma grids")
    mask = np.isfinite(p3)
    p3[~mask] = 0.0
    s = p3.sum()
    if s <= 0.0:
        raise RuntimeError("3D posterior sum is nonpositive")
    p3 /= s
    return qz, A, gamma, p3


def select_high_prob_points(qz, A, gamma, p3, frac=0.95, nmax=8000):
    QZ, Agrid, G = np.meshgrid(qz, A, gamma, indexing="ij")
    q_flat = QZ.ravel()
    A_flat = Agrid.ravel()
    g_flat = G.ravel()
    p_flat = p3.ravel()
    order = np.argsort(p_flat)[::-1]
    p_sorted = p_flat[order]
    csum = np.cumsum(p_sorted)
    csum /= csum[-1]
    k = np.searchsorted(csum, frac) + 1
    if k > nmax:
        k = nmax
    idx = order[:k]
    return q_flat[idx], A_flat[idx], g_flat[idx], p_flat[idx]


def main():
    qz, A, gamma, p3 = load_posterior("results/qzA_gamma_posterior_joint_physicalrc_factored_grid.npz")
    q_sel, A_sel, g_sel, p_sel = select_high_prob_points(qz, A, gamma, p3, frac=0.95, nmax=8000)
    fig = plt.figure(figsize=(7.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(A_sel, q_sel, g_sel, c=p_sel, s=6.0, alpha=0.8)
    ax.set_xlabel("A")
    ax.set_ylabel("q_z")
    ax.set_zlabel("gamma")
    ax.set_title("High-probability region of 3D posterior p(q_z, A, gamma)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="posterior density (relative)")
    fig.tight_layout()
    out_pdf = "results/qzA_gamma_posterior_joint_physicalrc_3dscatter.pdf"
    out_png = "results/qzA_gamma_posterior_joint_physicalrc_3dscatter.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
