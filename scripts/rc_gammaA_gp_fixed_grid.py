import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wquant(x, w, qs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    if w.sum() <= 0:
        return [float(np.nan) for _ in qs]
    o = np.argsort(x)
    x = x[o]
    w = w[o]
    w = w / w.sum()
    c = np.concatenate([[0.0], np.cumsum(w)])
    xx = np.concatenate([[x[0]], x])
    return [float(np.interp(q, c, xx)) for q in qs]

def read_rc(path):
    df = pd.read_csv(path)
    if "R_kpc" not in df.columns:
        raise RuntimeError("R_kpc missing")
    R = df["R_kpc"].to_numpy(float)
    v_cols = ["Vphi_kms", "Vc_kms", "v_c_kms", "vphi_kms"]
    v_name = next((c for c in v_cols if c in df.columns), None)
    if v_name is None:
        raise RuntimeError("no rotation speed column found")
    y = df[v_name].to_numpy(float)
    if "sigma_obs_kms" in df.columns:
        sigma = df["sigma_obs_kms"].to_numpy(float)
    else:
        plus = next((c for c in df.columns if "evphi_plus" in c.lower() or "ephi_plus" in c.lower()), None)
        minus = next((c for c in df.columns if "evphi_minus" in c.lower() or "ephi_minus" in c.lower()), None)
        if plus is None or minus is None:
            raise RuntimeError("no error columns found")
        sigma = 0.5 * (df[plus].to_numpy(float) + df[minus].to_numpy(float))
    m = np.isfinite(R) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    R = R[m]; y = y[m]; sigma = sigma[m]
    ii = np.argsort(R)
    return R[ii], y[ii], sigma[ii]

def read_summary(path):
    df = pd.read_csv(path)
    name_col = df.columns[0]
    df[name_col] = df[name_col].astype(str).str.strip()
    if "mean" not in df.columns:
        raise RuntimeError("rc_nuts_summary.csv has no 'mean' column")
    lookup = {str(df[name_col].iloc[i]).strip(): i for i in range(df.shape[0])}
    def mean_of(name):
        if name not in lookup:
            raise RuntimeError(f"{name} not found in rc_nuts_summary.csv")
        return float(df.iloc[lookup[name]]["mean"])
    out = {}
    for k in ["a_b","a_d","b_d","logMb","logMd","logrho0","rs","log_amp","log_scale","log_jit","gamma"]:
        out[k] = mean_of(k)
    return out

def matern32_matrix(x, amp, scale):
    x = np.asarray(x, float)
    dx = np.abs(x[:, None] - x[None, :])
    z = np.sqrt(3.0) * dx / float(scale)
    return (float(amp) ** 2) * (1.0 + z) * np.exp(-z)

def main():
    repo = Path(".").resolve()
    outdir = repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    rc_path = repo / "data" / "rotation_curve" / "beordo2024_allstars_rc_ready.csv"
    if not rc_path.exists():
        rc_path = repo / "data" / "rotation_curve" / "beordo2024" / "beordo2024_allstars_rc_ready.csv"
    if not rc_path.exists():
        raise RuntimeError("rotation curve CSV not found")

    summ_path = repo / "results" / "rc_nuts_summary.csv"
    if not summ_path.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")

    R, y, sigma_obs = read_rc(rc_path)
    s = read_summary(summ_path)

    Mb = float(np.exp(s["logMb"]))
    Md = float(np.exp(s["logMd"]))
    rho0 = float(np.exp(s["logrho0"]))
    rs = float(s["rs"])
    a_b = float(s["a_b"])
    a_d = float(s["a_d"])
    b_d = float(s["b_d"])

    amp = float(np.exp(s["log_amp"]))
    scale = float(np.exp(s["log_scale"]))
    jitter = float(np.exp(s["log_jit"]))

    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.vphi_axisym import DiskMN, BulgeHernquist, Halo_gNFW, vphi_total

    disk = DiskMN(M=Md, a=a_d, b=b_d)
    bulge = BulgeHernquist(M=Mb, a=a_b)

    K = matern32_matrix(R, amp=amp, scale=scale)
    K[np.diag_indices_from(K)] += sigma_obs**2 + jitter**2
    L = np.linalg.cholesky(K)

    def quadform_Kinv(r):
        z = np.linalg.solve(L, r)
        return float(np.dot(z, z))

    gamma_grid = np.linspace(0.2, 1.8, 161)
    A_grid = np.linspace(0.2, 1.6, 281)

    chi2 = np.zeros((gamma_grid.size, A_grid.size), float)
    for i, g in enumerate(gamma_grid):
        for j, A in enumerate(A_grid):
            halo = Halo_gNFW(rho0=rho0 * float(A), rs=rs, gamma=float(g), q=1.0)
            mu = np.asarray(vphi_total(R, disk, bulge, halo), float)
            r = y - mu
            chi2[i, j] = quadform_Kinv(r)

    chi2_min = float(chi2.min())
    dchi2 = chi2 - chi2_min
    p = np.exp(-0.5 * dchi2)
    p /= p.sum()

    p_gamma = p.sum(axis=1)
    p_A = p.sum(axis=0)

    g16, g50, g84 = wquant(gamma_grid, p_gamma, [0.16, 0.5, 0.84])
    g025, g975 = wquant(gamma_grid, p_gamma, [0.025, 0.975])
    a16, a50, a84 = wquant(A_grid, p_A, [0.16, 0.5, 0.84])
    a025, a975 = wquant(A_grid, p_A, [0.025, 0.975])

    imax = np.unravel_index(np.argmax(p), p.shape)
    g_best = float(gamma_grid[imax[0]])
    A_best = float(A_grid[imax[1]])

    G, Amesh = np.meshgrid(gamma_grid, A_grid, indexing="ij")
    g_mean = float((G * p).sum())
    a_mean = float((Amesh * p).sum())
    g_var = float(((G - g_mean) ** 2 * p).sum())
    a_var = float(((Amesh - a_mean) ** 2 * p).sum())
    cov = float(((G - g_mean) * (Amesh - a_mean) * p).sum())
    corr = float(cov / (np.sqrt(g_var * a_var) + 1e-300))

    txt = outdir / "rc_gammaA_gp_fixed_grid.txt"
    pdf = outdir / "rc_gammaA_gp_fixed_grid.pdf"
    npz = outdir / "rc_gammaA_gp_fixed_grid.npz"

    with txt.open("w") as f:
        f.write(f"RC_file {rc_path}\n")
        f.write(f"N {R.size}\n")
        f.write(f"GP_fixed amp {amp:.6f} scale {scale:.6f} jitter {jitter:.6f}\n")
        f.write(f"chi2_min {chi2_min:.6f}\n")
        f.write(f"gamma_best {g_best:.6f} A_best {A_best:.6f}\n")
        f.write(f"gamma_68 {g16:.6f} {g84:.6f}\n")
        f.write(f"gamma_95 {g025:.6f} {g975:.6f}\n")
        f.write(f"A_68 {a16:.6f} {a84:.6f}\n")
        f.write(f"A_95 {a025:.6f} {a975:.6f}\n")
        f.write(f"corr_gamma_A {corr:.6f}\n")
        f.write(f"nuts_gamma_mean {float(s['gamma']):.6f}\n")

    np.savez(npz, gamma=gamma_grid, A=A_grid, chi2=chi2, posterior=p)

    plt.figure(figsize=(7.2, 5.2))
    plt.imshow(-0.5 * dchi2, origin="lower", aspect="auto",
               extent=[A_grid.min(), A_grid.max(), gamma_grid.min(), gamma_grid.max()])
    plt.colorbar(label="-0.5 Δ(quadform)")
    plt.xlabel("A (rho0 scale)")
    plt.ylabel("gamma")
    plt.title("RC grid with GP-fixed quadratic form")
    plt.tight_layout()
    plt.savefig(pdf, dpi=160)

    print(str(txt))
    print(str(pdf))
    print("gamma_best", g_best, "A_best", A_best, "gamma_68", g16, g84, "A_68", a16, a84)

if __name__ == "__main__":
    main()
