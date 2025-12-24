import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_qzA_grid(path):
    d = np.load(path)
    if all(k in d.files for k in ("qz", "A", "posterior")):
        qz = d["qz"]
        A = d["A"]
        post = d["posterior"]
    else:
        qz = None
        A = None
        post = None
        for k in d.files:
            arr = d[k]
            if arr.ndim == 1:
                if qz is None:
                    qz = arr
                elif A is None:
                    A = arr
            elif arr.ndim == 2 and post is None:
                post = arr
    if qz is None or A is None or post is None:
        raise RuntimeError("could not identify qz, A, posterior in qzA grid file")
    post = np.clip(post, 0.0, np.inf)
    s = post.sum()
    if s <= 0.0:
        raise RuntimeError("qzA posterior sum nonpositive")
    post = post / s
    return qz, A, post

def load_gammaA_grid(path):
    d = np.load(path)
    if all(k in d.files for k in ("gamma", "A", "posterior")):
        gamma = d["gamma"]
        A = d["A"]
        post = d["posterior"]
    else:
        gamma = None
        A = None
        post = None
        for k in d.files:
            arr = d[k]
            if arr.ndim == 1:
                if gamma is None:
                    gamma = arr
                elif A is None:
                    A = arr
            elif arr.ndim == 2 and post is None:
                post = arr
    if gamma is None or A is None or post is None:
        raise RuntimeError("could not identify gamma, A, posterior in gammaA grid file")
    post = np.clip(post, 0.0, np.inf)
    s = post.sum()
    if s <= 0.0:
        raise RuntimeError("gammaA posterior sum nonpositive")
    post = post / s
    return gamma, A, post

def credible_from_1d(x, p, a_lo, a_hi):
    p = np.clip(p, 0.0, np.inf)
    s = p.sum()
    if s <= 0.0:
        return float(x.min()), float(x.max())
    p = p / s
    c = np.cumsum(p)
    c = c / c[-1]
    lo = np.interp(a_lo, c, x)
    hi = np.interp(a_hi, c, x)
    return float(lo), float(hi)

def contour_levels_from_pdf(pdf, levels):
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    flat_sorted = flat[order]
    c = np.cumsum(flat_sorted)
    c = c / c[-1]
    thr = []
    for lev in levels:
        idx = np.searchsorted(c, lev)
        if idx >= flat_sorted.size:
            idx = flat_sorted.size - 1
        thr.append(flat_sorted[idx])
    return thr

def core_cusp_bands(ax, gamma):
    colors = ["#d0f0ff", "#e0ffe0", "#fff0d0", "#ffd0d0"]
    ranges = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, gamma.max()),
    ]
    for (lo, hi), col in zip(ranges, colors):
        mask = (gamma >= lo) & (gamma <= hi)
        if np.any(mask):
            ax.axvspan(gamma[mask].min(), gamma[mask].max(), color=col, alpha=0.4)

def main():
    repo = Path(".").resolve()
    qzA_path = repo / "results" / "qzA_posterior_joint_physicalrc_refined_grid.npz"
    gammaA_path = repo / "results" / "rc_gammaA_posterior_grid.npz"
    if not qzA_path.exists():
        raise RuntimeError("missing qzA grid file")
    if not gammaA_path.exists():
        raise RuntimeError("missing gammaA grid file")
    qz, A_qz, post_qzA = load_qzA_grid(str(qzA_path))
    gamma, A_gam, post_gA = load_gammaA_grid(str(gammaA_path))
    p_qz = post_qzA.sum(axis=1)
    p_qz = p_qz / p_qz.sum()
    p_gamma = post_gA.sum(axis=1)
    p_gamma = p_gamma / p_gamma.sum()
    g_med, _ = credible_from_1d(gamma, p_gamma, 0.5, 0.5)
    g68_lo, g68_hi = credible_from_1d(gamma, p_gamma, 0.16, 0.84)
    levs_qzA = contour_levels_from_pdf(post_qzA, [0.68, 0.95])
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    ax0 = axes[0]
    Aq, Qq = np.meshgrid(A_qz, qz)
    im0 = ax0.contourf(Aq, Qq, post_qzA, levels=40)
    for lev in levs_qzA:
        ax0.contour(Aq, Qq, post_qzA, levels=[lev], colors="w", linestyles="--", linewidths=1.0)
    ax0.set_xlabel("A (halo normalization)")
    ax0.set_ylabel("q_z")
    ax0.set_title("Joint GD-1 + RC posterior p(q_z, A)")
    fig.colorbar(im0, ax=ax0, label="p(q_z, A)")
    ax1 = axes[1]
    core_cusp_bands(ax1, gamma)
    ax1.plot(gamma, p_gamma, lw=1.8, color="k")
    ax1.axvline(g_med, ls="--", lw=1.2, color="k")
    ax1.axvline(g68_lo, ls=":", lw=1.0, color="k")
    ax1.axvline(g68_hi, ls=":", lw=1.0, color="k")
    ax1.set_xlabel("gamma (inner DM slope)")
    ax1.set_ylabel("normalized posterior p(gamma | RC, model)")
    ax1.set_title("RC-only posterior for gamma (core vs cusp)")
    fig.tight_layout()
    pdf = out / "final_summary_qz_gamma.pdf"
    png = out / "final_summary_qz_gamma.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(str(pdf))
    print(str(png))

if __name__ == "__main__":
    main()
