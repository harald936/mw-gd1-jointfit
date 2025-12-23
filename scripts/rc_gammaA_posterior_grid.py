import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_rotation_curve(repo):
    base = repo / "data" / "rotation_curve"
    path = base / "beordo2024_allstars_rc_ready.csv"
    if not path.exists():
        raise RuntimeError("rotation curve CSV not found at data/rotation_curve/beordo2024_allstars_rc_ready.csv")
    df = pd.read_csv(path)
    if "R_kpc" not in df.columns:
        raise RuntimeError("R_kpc column missing in rotation curve CSV")
    R = df["R_kpc"].to_numpy(float)
    v_cols = ["Vphi_kms", "Vc_kms", "v_c_kms", "vphi_kms"]
    v_name = None
    for c in v_cols:
        if c in df.columns:
            v_name = c
            break
    if v_name is None:
        raise RuntimeError("no rotation speed column found in rotation curve CSV")
    V = df[v_name].to_numpy(float)
    if "sigma_obs_kms" in df.columns:
        E = df["sigma_obs_kms"].to_numpy(float)
    else:
        plus = None
        minus = None
        for c in df.columns:
            lc = c.lower()
            if "ephi_plus" in lc or "evphi_plus" in lc:
                plus = c
            if "ephi_minus" in lc or "evphi_minus" in lc:
                minus = c
        if plus is None or minus is None:
            raise RuntimeError("no error columns found in rotation curve CSV")
        e_plus = df[plus].to_numpy(float)
        e_minus = df[minus].to_numpy(float)
        E = 0.5 * (e_plus + e_minus)
    mask = np.isfinite(R) & np.isfinite(V) & np.isfinite(E) & (E > 0)
    R = R[mask]
    V = V[mask]
    E = E[mask]
    return R, V, E, str(path)

def load_rc_params(summary_path):
    df = pd.read_csv(summary_path)
    name_col = df.columns[0]
    lookup = {str(df[name_col].iloc[i]): i for i in range(df.shape[0])}
    def get_mean_sd(name):
        if name not in lookup:
            raise RuntimeError(f"parameter {name} not found in rc summary")
        row = df.iloc[lookup[name]]
        mean_col = None
        sd_col = None
        for c in df.columns:
            lc = c.lower()
            if lc == "mean":
                mean_col = c
            if "sd" in lc or "sigma" in lc:
                sd_col = c
        if mean_col is None:
            mean_col = df.columns[1]
        if sd_col is None:
            sd_col = df.columns[2]
        return float(row[mean_col]), float(row[sd_col])
    a_b_mean, _ = get_mean_sd("a_b")
    a_d_mean, _ = get_mean_sd("a_d")
    b_d_mean, _ = get_mean_sd("b_d")
    gamma_mean, gamma_sd = get_mean_sd("gamma")
    logMb_mean, _ = get_mean_sd("logMb")
    logMd_mean, _ = get_mean_sd("logMd")
    logrho0_mean, _ = get_mean_sd("logrho0")
    rs_mean, _ = get_mean_sd("rs")
    Mb = float(np.exp(logMb_mean))
    Md = float(np.exp(logMd_mean))
    rho0 = float(np.exp(logrho0_mean))
    return a_b_mean, a_d_mean, b_d_mean, gamma_mean, gamma_sd, Mb, Md, rho0, rs_mean

def build_components(a_b, a_d, b_d, gamma, Mb, Md, rho0, rs, A_scale):
    repo = Path(".").resolve()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.vphi_axisym import DiskMN, BulgeHernquist, Halo_gNFW, vphi_total
    disk = DiskMN(M=Md, a=a_d, b=b_d)
    bulge = BulgeHernquist(M=Mb, a=a_b)
    halo = Halo_gNFW(rho0=rho0 * A_scale, rs=rs, gamma=gamma, q=1.0)
    return disk, bulge, halo, vphi_total

def chi2_rc(R, V_obs, E_obs, model_v):
    r = (V_obs - model_v) / E_obs
    return float(np.sum(r * r))

def credible_from_1d(x, p, alpha_low, alpha_high):
    p = np.clip(p, 0.0, np.inf)
    if p.sum() <= 0.0:
        return float(x.min()), float(x.max())
    p = p / p.sum()
    c = np.cumsum(p)
    c = c / c[-1]
    lo = np.interp(alpha_low, c, x)
    hi = np.interp(alpha_high, c, x)
    return float(lo), float(hi)

def contour_levels_from_pdf(pdf, levels):
    flat = pdf.ravel()
    order = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[order])
    cumsum /= cumsum[-1]
    levs = []
    for p in levels:
        k = np.searchsorted(cumsum, p)
        if k >= len(order):
            val = flat[order[-1]]
        else:
            val = flat[order[k]]
        levs.append(val)
    return np.array(levs)

def main():
    repo = Path(".").resolve()
    R, V, E, rc_path = load_rotation_curve(repo)
    summary_path = repo / "results" / "rc_nuts_summary.csv"
    if not summary_path.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")
    a_b, a_d, b_d, gamma_mean, gamma_sd, Mb, Md, rho0, rs = load_rc_params(str(summary_path))
    gamma_grid = np.linspace(0.0, 2.0, 161)
    A_grid = np.linspace(0.6, 1.4, 121)
    chi2_grid = np.zeros((gamma_grid.size, A_grid.size))
    for i, g in enumerate(gamma_grid):
        for j, Aval in enumerate(A_grid):
            disk, bulge, halo, vphi_total = build_components(a_b, a_d, b_d, g, Mb, Md, rho0, rs, Aval)
            v_model = vphi_total(R, disk, bulge, halo)
            chi2_grid[i, j] = chi2_rc(R, V, E, v_model)
    chi2_min = float(chi2_grid.min())
    dchi2 = chi2_grid - chi2_min
    p = np.exp(-0.5 * dchi2)
    p_sum = p.sum()
    if p_sum <= 0.0:
        raise RuntimeError("posterior sum nonpositive in gamma-A grid")
    p /= p_sum
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rc_gammaA_posterior_grid.npz", gamma=gamma_grid, A=A_grid, chi2=chi2_grid, posterior=p)
    p_gamma = p.sum(axis=1)
    p_A = p.sum(axis=0)
    g_lo68, g_hi68 = credible_from_1d(gamma_grid, p_gamma, 0.16, 0.84)
    g_lo95, g_hi95 = credible_from_1d(gamma_grid, p_gamma, 0.025, 0.975)
    A_lo68, A_hi68 = credible_from_1d(A_grid, p_A, 0.16, 0.84)
    A_lo95, A_hi95 = credible_from_1d(A_grid, p_A, 0.025, 0.975)
    idx_max = np.unravel_index(np.argmax(p), p.shape)
    g_best = float(gamma_grid[idx_max[0]])
    A_best = float(A_grid[idx_max[1]])
    g_mean = float((gamma_grid[:, None] * p).sum())
    A_mean = float((A_grid[None, :] * p).sum())
    g_var = float(((gamma_grid[:, None] - g_mean) ** 2 * p).sum())
    A_var = float(((A_grid[None, :] - A_mean) ** 2 * p).sum())
    cov = float(((gamma_grid[:, None] - g_mean) * (A_grid[None, :] - A_mean) * p).sum())
    if g_var > 0.0 and A_var > 0.0:
        corr = cov / np.sqrt(g_var * A_var)
    else:
        corr = 0.0
    levs = contour_levels_from_pdf(p, [0.68, 0.95])
    levs = np.sort(levs)
    G, Amesh = np.meshgrid(gamma_grid, A_grid, indexing="ij")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cf = ax.contourf(Amesh, G, p, levels=40)
    ax.contour(Amesh, G, p, levels=levs, colors="w", linestyles=["--", ":"], linewidths=1.2)
    ax.plot(A_best, g_best, "wo", ms=5)
    ax.set_xlabel("A (halo normalization)")
    ax.set_ylabel("gamma (inner slope)")
    ax.set_title("RC-only posterior p(gamma, A)")
    fig.colorbar(cf, ax=ax, label="p(gamma, A)")
    fig.tight_layout()
    pdf = out / "rc_gammaA_posterior_grid.pdf"
    png = out / "rc_gammaA_posterior_grid.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    txt = out / "rc_gammaA_posterior_grid.txt"
    with open(txt, "w") as f:
        f.write(f"RC file {rc_path}\n")
        f.write(f"rc_nuts_summary {summary_path}\n")
        f.write(f"gamma_mean_prior {gamma_mean:.6f} gamma_sd_prior {gamma_sd:.6f}\n")
        f.write(f"grid_gamma_min {gamma_grid.min():.6f} grid_gamma_max {gamma_grid.max():.6f} N_gamma {gamma_grid.size}\n")
        f.write(f"grid_A_min {A_grid.min():.6f} grid_A_max {A_grid.max():.6f} N_A {A_grid.size}\n")
        f.write(f"chi2_min {chi2_min:.3f}\n")
        f.write(f"gamma_best {g_best:.6f} A_best {A_best:.6f}\n")
        f.write(f"gamma_68 {g_lo68:.6f} {g_hi68:.6f}\n")
        f.write(f"gamma_95 {g_lo95:.6f} {g_hi95:.6f}\n")
        f.write(f"A_68 {A_lo68:.6f} {A_hi68:.6f}\n")
        f.write(f"A_95 {A_lo95:.6f} {A_hi95:.6f}\n")
        f.write(f"corr_gamma_A {corr:.6f}\n")
    print(str(pdf))
    print(str(txt))

if __name__ == "__main__":
    main()
