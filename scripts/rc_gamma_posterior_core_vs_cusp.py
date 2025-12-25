from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def quantile_from_grid(x, p, q):
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    p = np.clip(p, 0.0, np.inf)
    s = float(p.sum())
    if s <= 0.0:
        return float(x[len(x)//2])
    p = p / s
    c = np.cumsum(p)
    return float(np.interp(q, c, x))

def main():
    repo = Path(__file__).resolve().parents[1]
    grid_path = repo / "results" / "rc_gammaA_from_nuts_samples.npz"
    if not grid_path.exists():
        raise RuntimeError("results/rc_gammaA_from_nuts_samples.npz not found; run scripts/rc_gammaA_from_nuts_samples_build.py first")

    data = np.load(grid_path)
    g = np.asarray(data["gamma"], float)
    P = np.asarray(data["posterior"], float)

    p_gamma = P.sum(axis=1)
    p_gamma = p_gamma / p_gamma.sum()

    g03 = quantile_from_grid(g, p_gamma, 0.03)
    g16 = quantile_from_grid(g, p_gamma, 0.16)
    g50 = quantile_from_grid(g, p_gamma, 0.50)
    g84 = quantile_from_grid(g, p_gamma, 0.84)
    g97 = quantile_from_grid(g, p_gamma, 0.97)

    def mass(mask):
        return float(p_gamma[mask].sum())

    P_core   = mass(g < 0.5)
    P_shal   = mass((g >= 0.5) & (g < 1.0))
    P_nfw    = mass((g >= 1.0) & (g < 1.5))
    P_steep  = mass(g >= 1.5)

    def mass_at(val):
        idx = np.where(np.isclose(g, val))[0]
        if len(idx) == 0:
            return None
        return float(p_gamma[int(idx[0])])

    m05 = mass_at(0.5)
    m10 = mass_at(1.0)
    m15 = mass_at(1.5)

    out_txt = repo / "results" / "rc_gamma_posterior_core_vs_cusp.txt"
    out_pdf = repo / "results" / "rc_gamma_posterior_core_vs_cusp.pdf"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"gamma_grid_min {float(g.min()):.6f} gamma_grid_max {float(g.max()):.6f} N_gamma {int(g.size)}")
    lines.append(f"gamma_median {g50:.6f}")
    lines.append(f"gamma_68 {g16:.6f} {g84:.6f}")
    lines.append(f"gamma_95 {g03:.6f} {g97:.6f}")
    lines.append("binning_convention: core <0.5, shallow [0.5,1.0), nfw [1.0,1.5), steep >=1.5 (half-open bins)")
    lines.append(f"P_core_<0.5 {P_core:.6f}")
    lines.append(f"P_shallow_[0.5,1.0) {P_shal:.6f}")
    lines.append(f"P_nfw_[1.0,1.5) {P_nfw:.6f}")
    lines.append(f"P_steep_>=1.5 {P_steep:.6f}")
    lines.append(f"mass_at_gamma_0.5 {('NONE' if m05 is None else f'{m05:.6f}')}")
    lines.append(f"mass_at_gamma_1.0 {('NONE' if m10 is None else f'{m10:.6f}')}")
    lines.append(f"mass_at_gamma_1.5 {('NONE' if m15 is None else f'{m15:.6f}')}")
    out_txt.write_text("\n".join(lines) + "\n")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(g, p_gamma, lw=2)
    for x in [0.5, 1.0, 1.5]:
        plt.axvline(x, linestyle="--", linewidth=1)
    plt.xlabel("gamma")
    plt.ylabel("marginal posterior density (binned)")
    plt.title("RC: marginal posterior over gamma (from NUTS samples)")
    plt.tight_layout()
    plt.savefig(out_pdf)

    print(str(out_pdf))
    print(str(out_txt))
    for s in lines[1:9]:
        print(s)

if __name__ == "__main__":
    main()
