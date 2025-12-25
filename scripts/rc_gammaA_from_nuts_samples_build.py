from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _binned_quantiles(x_cent, p, qs):
    p = np.asarray(p, float)
    p = p / p.sum()
    c = np.cumsum(p)
    out = []
    for q in qs:
        out.append(float(np.interp(q, c, x_cent)))
    return out

def main():
    repo = Path(__file__).resolve().parents[1]
    out_npz = repo / "results" / "rc_gammaA_from_nuts_samples.npz"
    out_txt = repo / "results" / "rc_gammaA_from_nuts_samples.txt"
    out_pdf = repo / "results" / "rc_gammaA_from_nuts_samples.pdf"

    quick = repo / "results" / "rc_nuts_quick_posterior.npz"
    summary = repo / "results" / "rc_nuts_summary.csv"

    if quick.exists() and summary.exists():
        npz = np.load(quick)
        gamma = np.asarray(npz["gamma"], float)
        logrho0 = np.asarray(npz["logrho0"], float)
        rho0 = np.exp(logrho0)

        df = pd.read_csv(summary)
        name = df.columns[0]
        df[name] = df[name].astype(str).str.strip()
        logrho0_ref = float(df.loc[df[name] == "logrho0", "mean"].iloc[0])
        rho0_ref = float(np.exp(logrho0_ref))

        A = rho0 / rho0_ref

        gmin, gmax = 0.2, 1.8
        Amin = float(max(0.05, np.quantile(A, 0.001)))
        Amax = float(min(20.0, np.quantile(A, 0.999)))

        gamma_clip = np.clip(gamma, gmin, gmax)
        A_clip = np.clip(A, Amin, Amax)

        Ng, NA = 161, 281
        g_edges = np.linspace(gmin, gmax, Ng + 1)
        A_edges = np.linspace(Amin, Amax, NA + 1)

        H, _, _ = np.histogram2d(gamma_clip, A_clip, bins=[g_edges, A_edges])
        P = H / H.sum()

        g_cent = 0.5 * (g_edges[:-1] + g_edges[1:])
        A_cent = 0.5 * (A_edges[:-1] + A_edges[1:])

        eps = 1e-300
        dchi2 = -2.0 * np.log(P + eps)
        dchi2 -= float(dchi2.min())

        np.savez(out_npz, gamma=g_cent, A=A_cent, posterior=P, chi2=dchi2)

    elif out_npz.exists():
        z = np.load(out_npz)
        g_cent = np.asarray(z["gamma"], float)
        A_cent = np.asarray(z["A"], float)
        P = np.asarray(z["posterior"], float)
    else:
        raise RuntimeError("missing rc_nuts_quick_posterior.npz and rc_gammaA_from_nuts_samples.npz in results/")

    eps = 1e-300
    p_gamma = P.sum(axis=1)
    p_gamma = p_gamma / p_gamma.sum()

    g03, g16, g50, g84, g97 = _binned_quantiles(g_cent, p_gamma, [0.03, 0.16, 0.50, 0.84, 0.97])

    def mass(mask):
        return float(p_gamma[mask].sum())

    lines = []
    lines.append(f"gamma_grid_min {float(g_cent.min()):.6f} gamma_grid_max {float(g_cent.max()):.6f} N_gamma {g_cent.size}")
    lines.append(f"gamma_median {g50:.6f}")
    lines.append(f"gamma_68 {g16:.6f} {g84:.6f}")
    lines.append(f"gamma_95 {g03:.6f} {g97:.6f}")
    lines.append("binning_convention: core <0.5, shallow [0.5,1.0), nfw [1.0,1.5), steep >=1.5 (half-open bins)")
    lines.append(f"P_core_<0.5 {mass(g_cent < 0.5):.6f}")
    lines.append(f"P_shallow_[0.5,1.0) {mass((g_cent >= 0.5) & (g_cent < 1.0)):.6f}")
    lines.append(f"P_nfw_[1.0,1.5) {mass((g_cent >= 1.0) & (g_cent < 1.5)):.6f}")
    lines.append(f"P_steep_>=1.5 {mass(g_cent >= 1.5):.6f}")

    for val in [0.5, 1.0, 1.5]:
        idx = np.where(np.isclose(g_cent, val))[0]
        if len(idx) == 0:
            lines.append(f"mass_at_gamma_{val} NONE")
        else:
            lines.append(f"mass_at_gamma_{val} {float(p_gamma[int(idx[0])]):.6f}")

    out_txt.write_text("\n".join(lines) + "\n")

    plt.figure(figsize=(7.2, 5.2))
    plt.imshow(np.log(P.T + eps), origin="lower", aspect="auto",
               extent=[g_cent.min(), g_cent.max(), A_cent.min(), A_cent.max()])
    plt.colorbar(label="log posterior density (binned)")
    plt.xlabel("gamma")
    plt.ylabel("A (rho0 / rho0_mean_from_summary)")
    plt.title("RC: (gamma, A) from NUTS samples (binned)")
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=160)

    print(str(out_npz))
    print(str(out_txt))
    print(str(out_pdf))
    print("gamma_median", g50, "gamma_68", g16, g84, "gamma_95", g03, g97)

if __name__ == "__main__":
    main()
