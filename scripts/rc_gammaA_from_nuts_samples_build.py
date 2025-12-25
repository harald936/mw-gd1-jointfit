from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def credible_from_1d(x, p, qlo, qhi):
    p = np.clip(p, 0.0, np.inf)
    s = float(p.sum())
    if s <= 0.0:
        return float(x.min()), float(x.max())
    p = p / s
    c = np.cumsum(p)
    lo = float(np.interp(qlo, c, x))
    hi = float(np.interp(qhi, c, x))
    return lo, hi

def main():
    repo = Path(".").resolve()
    npz_path = repo / "results" / "rc_nuts_quick_posterior.npz"
    if not npz_path.exists():
        raise RuntimeError("missing results/rc_nuts_quick_posterior.npz (generate it from your NUTS run first)")

    z = np.load(npz_path)
    if "gamma" not in z:
        raise RuntimeError("gamma not found in rc_nuts_quick_posterior.npz")
    if "logrho0" not in z and "rho0" not in z:
        raise RuntimeError("need logrho0 or rho0 in rc_nuts_quick_posterior.npz")

    gamma = np.asarray(z["gamma"], float)
    if "logrho0" in z:
        rho0 = np.exp(np.asarray(z["logrho0"], float))
    else:
        rho0 = np.asarray(z["rho0"], float)

    df = pd.read_csv(repo / "results" / "rc_nuts_summary.csv")
    name = df.columns[0]
    df[name] = df[name].astype(str).str.strip()
    logrho0_ref = float(df.loc[df[name] == "logrho0", "mean"].iloc[0])
    rho0_ref = float(np.exp(logrho0_ref))

    A = rho0 / rho0_ref

    gmin, gmax = 0.2, 1.8
    gamma_clip = np.clip(gamma, gmin, gmax)

    Amin = float(max(0.05, np.quantile(A, 0.01)))
    Amax = float(min(20.0, np.quantile(A, 0.99)))
    A_clip = np.clip(A, Amin, Amax)

    Ng, NA = 161, 281
    g_edges = np.linspace(gmin, gmax, Ng + 1)
    A_edges = np.linspace(Amin, Amax, NA + 1)

    H, _, _ = np.histogram2d(gamma_clip, A_clip, bins=[g_edges, A_edges])
    P = H / float(H.sum())

    g_cent = 0.5 * (g_edges[:-1] + g_edges[1:])
    A_cent = 0.5 * (A_edges[:-1] + A_edges[1:])

    eps = 1e-300
    dchi2 = -2.0 * np.log(P + eps)
    dchi2 -= float(dchi2.min())

    out_npz = repo / "results" / "rc_gammaA_from_nuts_samples.npz"
    np.savez(out_npz, gamma=g_cent, A=A_cent, posterior=P, chi2=dchi2)

    p_g = P.sum(axis=1)
    p_g = p_g / float(p_g.sum())
    g16, g84 = credible_from_1d(g_cent, p_g, 0.16, 0.84)
    g03, g97 = credible_from_1d(g_cent, p_g, 0.03, 0.97)
    gmed = float(np.interp(0.50, np.cumsum(p_g), g_cent))

    out_txt = repo / "results" / "rc_gammaA_from_nuts_samples.txt"
    out_pdf = repo / "results" / "rc_gammaA_from_nuts_samples.pdf"

    corr = float(np.corrcoef(gamma, A)[0, 1])

    txt = []
    txt.append(f"npz {out_npz}")
    txt.append(f"N_samples {gamma.size}")
    txt.append(f"gamma_clip_range {gmin} {gmax}")
    txt.append(f"A_clip_range {Amin} {Amax}")
    txt.append(f"frac_clipped_low {float(np.mean(A < Amin))}")
    txt.append(f"frac_clipped_high {float(np.mean(A > Amax))}")
    txt.append(f"corr_gamma_A_samples {corr}")
    txt.append(f"gamma_median {gmed}")
    txt.append(f"gamma_68 {g16} {g84}")
    txt.append(f"gamma_95 {g03} {g97}")
    out_txt.write_text("\n".join(txt) + "\n")

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
    print("gamma_median", gmed, "gamma_68", g16, g84, "gamma_95", g03, g97)

if __name__ == "__main__":
    main()
