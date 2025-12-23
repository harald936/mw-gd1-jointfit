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

def chi2_rc(R, V_obs, E_obs, V_model):
    r = (V_obs - V_model) / E_obs
    return float(np.sum(r * r))

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

def core_cusp_probs(gamma, p_gamma):
    p = np.clip(p_gamma, 0.0, np.inf)
    s = p.sum()
    if s <= 0.0:
        return [(name, 0.0) for name in ["core_0_0.5","shallow_0.5_1.0","nfw_1.0_1.5","steep_>1.5"]]
    p = p / s
    ranges = [
        ("core_0_0.5", 0.0, 0.5),
        ("shallow_0.5_1.0", 0.5, 1.0),
        ("nfw_1.0_1.5", 1.0, 1.5),
        ("steep_>1.5", 1.5, gamma.max() + 1e-6),
    ]
    out = []
    for name, lo, hi in ranges:
        if name == "steep_>1.5":
            mask = (gamma >= lo) & (gamma <= hi)
        else:
            mask = (gamma >= lo) & (gamma < hi)
        out.append((name, float(p[mask].sum())))
    return out

def gamma_posterior_for_dataset(R, V_obs, E_obs, gamma_grid, A_grid, a_b, a_d, b_d, Mb, Md, rho0, rs):
    chi2_grid = np.zeros((gamma_grid.size, A_grid.size))
    for i, g in enumerate(gamma_grid):
        for j, Aval in enumerate(A_grid):
            disk, bulge, halo, vphi_total = build_components(a_b, a_d, b_d, g, Mb, Md, rho0, rs, Aval)
            V_model = vphi_total(R, disk, bulge, halo)
            chi2_grid[i, j] = chi2_rc(R, V_obs, E_obs, V_model)
    chi2_min = float(chi2_grid.min())
    dchi2 = chi2_grid - chi2_min
    p = np.exp(-0.5 * dchi2)
    s = p.sum()
    if s <= 0.0:
        raise RuntimeError("posterior sum nonpositive in gamma-A grid")
    p = p / s
    p_gamma = p.sum(axis=1)
    s1 = p_gamma.sum()
    if s1 <= 0.0:
        raise RuntimeError("gamma-marginal posterior sum nonpositive")
    p_gamma = p_gamma / s1
    g_med_lo, g_med_hi = credible_from_1d(gamma_grid, p_gamma, 0.5, 0.5)
    g68_lo, g68_hi = credible_from_1d(gamma_grid, p_gamma, 0.16, 0.84)
    g95_lo, g95_hi = credible_from_1d(gamma_grid, p_gamma, 0.025, 0.975)
    probs = core_cusp_probs(gamma_grid, p_gamma)
    return p_gamma, g_med_lo, g68_lo, g68_hi, g95_lo, g95_hi, probs

def main():
    repo = Path(".").resolve()
    R, V_real, E_real, rc_path = load_rotation_curve(repo)
    summary_path = repo / "results" / "rc_nuts_summary.csv"
    if not summary_path.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")
    a_b, a_d, b_d, gamma_mean, gamma_sd, Mb, Md, rho0, rs = load_rc_params(str(summary_path))
    gamma_grid = np.linspace(0.0, 2.0, 161)
    A_grid = np.linspace(0.6, 1.4, 121)
    rng = np.random.default_rng(12345)
    scenarios = [
        ("core_true", 0.0, 1.0),
        ("nfw_true", 1.0, 1.0),
        ("steep_true", 1.6, 1.0),
    ]
    results = []
    for label, g_true, A_true in scenarios:
        disk_t, bulge_t, halo_t, vphi_total = build_components(a_b, a_d, b_d, g_true, Mb, Md, rho0, rs, A_true)
        V_true = vphi_total(R, disk_t, bulge_t, halo_t)
        V_mock = V_true + rng.normal(0.0, E_real)
        p_gamma, g_med, g68_lo, g68_hi, g95_lo, g95_hi, probs = gamma_posterior_for_dataset(
            R, V_mock, E_real, gamma_grid, A_grid, a_b, a_d, b_d, Mb, Md, rho0, rs
        )
        results.append((label, g_true, p_gamma, g_med, g68_lo, g68_hi, g95_lo, g95_hi, probs))
    p_gamma_real, g_med_r, g68_lo_r, g68_hi_r, g95_lo_r, g95_hi_r, probs_r = gamma_posterior_for_dataset(
        R, V_real, E_real, gamma_grid, A_grid, a_b, a_d, b_d, Mb, Md, rho0, rs
    )
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, g_true, p_gamma, g_med, g68_lo, g68_hi, g95_lo, g95_hi, probs in results:
        ax.plot(gamma_grid, p_gamma, lw=1.4, label=f"mock {label} (gamma_true={g_true:.1f})")
    ax.plot(gamma_grid, p_gamma_real, lw=2.0, color="k", label="real Beordo RC")
    ax.set_xlabel("gamma (inner DM slope)")
    ax.set_ylabel("normalized posterior p(gamma)")
    ax.set_title("Mock core/NFW/steep tests vs real RC")
    ax.legend()
    fig.tight_layout()
    pdf = out / "rc_gamma_posterior_mock_core_vs_cusp.pdf"
    png = out / "rc_gamma_posterior_mock_core_vs_cusp.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    txt = out / "rc_gamma_posterior_mock_core_vs_cusp.txt"
    with open(txt, "w") as f:
        f.write(f"RC file {rc_path}\n")
        f.write(f"rc_nuts_summary {summary_path}\n")
        f.write(f"gamma_grid_min {gamma_grid.min():.6f} gamma_grid_max {gamma_grid.max():.6f} N_gamma {gamma_grid.size}\n")
        f.write(f"A_grid_min {A_grid.min():.6f} A_grid_max {A_grid.max():.6f} N_A {A_grid.size}\n")
        f.write("scenario gamma_true gamma_median gamma_68_lo gamma_68_hi gamma_95_lo gamma_95_hi P_core P_shallow P_nfw P_steep\n")
        def probs_to_row(pr):
            d = {name: prob for name, prob in pr}
            return d.get("core_0_0.5", 0.0), d.get("shallow_0.5_1.0", 0.0), d.get("nfw_1.0_1.5", 0.0), d.get("steep_>1.5", 0.0)
        for label, g_true, p_gamma, g_med, g68_lo, g68_hi, g95_lo, g95_hi, probs in results:
            P_core, P_sh, P_nfw, P_steep = probs_to_row(probs)
            f.write(f"{label} {g_true:.6f} {g_med:.6f} {g68_lo:.6f} {g68_hi:.6f} {g95_lo:.6f} {g95_hi:.6f} {P_core:.6f} {P_sh:.6f} {P_nfw:.6f} {P_steep:.6f}\n")
        P_core_r, P_sh_r, P_nfw_r, P_steep_r = probs_to_row(probs_r)
        f.write(f"real_data NaN {g_med_r:.6f} {g68_lo_r:.6f} {g68_hi_r:.6f} {g95_lo_r:.6f} {g95_hi_r:.6f} {P_core_r:.6f} {P_sh_r:.6f} {P_nfw_r:.6f} {P_steep_r:.6f}\n")
    print(str(pdf))
    print(str(txt))
    for label, g_true, p_gamma, g_med, g68_lo, g68_hi, g95_lo, g95_hi, probs in results:
        print(label, "gamma_true", f"{g_true:.3f}", "gamma_median", f"{g_med:.3f}", "gamma_68", f"{g68_lo:.3f}", f"{g68_hi:.3f}")
        for name, prob in probs:
            print(" ", label, name, f"{prob:.3f}")
    print("real_data gamma_median", f"{g_med_r:.3f}", "gamma_68", f"{g68_lo_r:.3f}", f"{g68_hi_r:.3f}")
    for name, prob in probs_r:
        print(" ", "real_data", name, f"{prob:.3f}")

if __name__ == "__main__":
    main()
