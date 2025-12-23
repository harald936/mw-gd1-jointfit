import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def credible_from_1d(x, p, alpha_low, alpha_high):
    p = np.clip(p, 0.0, np.inf)
    s = p.sum()
    if s <= 0.0:
        return float(x.min()), float(x.max())
    p = p / s
    c = np.cumsum(p)
    c = c / c[-1]
    lo = np.interp(alpha_low, c, x)
    hi = np.interp(alpha_high, c, x)
    return float(lo), float(hi)

def main():
    repo = Path(".").resolve()
    grid_path = repo / "results" / "rc_gammaA_posterior_grid.npz"
    if not grid_path.exists():
        raise RuntimeError("results/rc_gammaA_posterior_grid.npz not found, run rc_gammaA_posterior_grid.py first")
    data = np.load(grid_path)
    gamma = data["gamma"]
    p2 = data["posterior"]
    if p2.ndim != 2 or p2.shape[0] != gamma.size:
        raise RuntimeError("posterior array has unexpected shape")
    p_gamma = p2.sum(axis=1)
    s = p_gamma.sum()
    if s <= 0.0:
        raise RuntimeError("gamma-marginal posterior sum nonpositive")
    p_gamma = p_gamma / s
    g_med_lo, g_med_hi = credible_from_1d(gamma, p_gamma, 0.5, 0.5)
    g68_lo, g68_hi = credible_from_1d(gamma, p_gamma, 0.16, 0.84)
    g95_lo, g95_hi = credible_from_1d(gamma, p_gamma, 0.025, 0.975)
    ranges = [
        ("core_0_0.5", 0.0, 0.5),
        ("shallow_0.5_1.0", 0.5, 1.0),
        ("nfw_1.0_1.5", 1.0, 1.5),
        ("steep_>1.5", 1.5, gamma.max() + 1e-6),
    ]
    probs = []
    for name, lo, hi in ranges:
        if name == "steep_>1.5":
            mask = (gamma >= lo) & (gamma <= hi)
        else:
            mask = (gamma >= lo) & (gamma < hi)
        probs.append((name, float(p_gamma[mask].sum())))
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(gamma, p_gamma, lw=1.8)
    ax.set_xlabel("gamma (inner DM slope)")
    ax.set_ylabel("normalized posterior p(gamma | RC, model)")
    ax.set_title("RC-only posterior for inner slope gamma")
    colors = ["#d0f0ff", "#e0ffe0", "#fff0d0", "#ffd0d0"]
    for (name, lo, hi), col in zip(ranges, colors):
        if name == "steep_>1.5":
            mask_band = (gamma >= lo) & (gamma <= hi)
        else:
            mask_band = (gamma >= lo) & (gamma < hi)
        if np.any(mask_band):
            ax.axvspan(gamma[mask_band].min(), gamma[mask_band].max(), color=col, alpha=0.4)
    ax.axvline(g_med_lo, ls="--", lw=1.2, color="k")
    ax.axvline(g68_lo, ls=":", lw=1.0, color="k")
    ax.axvline(g68_hi, ls=":", lw=1.0, color="k")
    fig.tight_layout()
    pdf = out / "rc_gamma_posterior_core_vs_cusp.pdf"
    png = out / "rc_gamma_posterior_core_vs_cusp.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    txt = out / "rc_gamma_posterior_core_vs_cusp.txt"
    with open(txt, "w") as f:
        f.write(f"gamma_grid_min {gamma.min():.6f} gamma_grid_max {gamma.max():.6f} N_gamma {gamma.size}\n")
        f.write(f"gamma_median {g_med_lo:.6f}\n")
        f.write(f"gamma_68 {g68_lo:.6f} {g68_hi:.6f}\n")
        f.write(f"gamma_95 {g95_lo:.6f} {g95_hi:.6f}\n")
        for name, prob in probs:
            f.write(f"P_{name} {prob:.6f}\n")
    print(str(pdf))
    print(str(txt))
    print(f"gamma_median {g_med_lo:.6f}")
    print(f"gamma_68 {g68_lo:.6f} {g68_hi:.6f}")
    print(f"gamma_95 {g95_lo:.6f} {g95_hi:.6f}")
    for name, prob in probs:
        print(f"P({name}) {prob:.6f}")

if __name__ == "__main__":
    main()
