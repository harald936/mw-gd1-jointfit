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
    def get_mean(name):
        if name not in lookup:
            raise RuntimeError(f"parameter {name} not found in rc summary")
        row = df.iloc[lookup[name]]
        mean_col = None
        for c in df.columns:
            if c.lower() == "mean":
                mean_col = c
                break
        if mean_col is None:
            mean_col = df.columns[1]
        return float(row[mean_col])
    a_b = get_mean("a_b")
    a_d = get_mean("a_d")
    b_d = get_mean("b_d")
    gamma = get_mean("gamma")
    logMb = get_mean("logMb")
    logMd = get_mean("logMd")
    logrho0 = get_mean("logrho0")
    rs = get_mean("rs")
    Mb = float(np.exp(logMb))
    Md = float(np.exp(logMd))
    rho0 = float(np.exp(logrho0))
    return a_b, a_d, b_d, gamma, Mb, Md, rho0, rs

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

def main():
    repo = Path(".").resolve()
    R, V, E, rc_path = load_rotation_curve(repo)
    summary_path = repo / "results" / "rc_nuts_summary.csv"
    if not summary_path.exists():
        raise RuntimeError("results/rc_nuts_summary.csv not found")
    a_b, a_d, b_d, gamma_mean, Mb, Md, rho0, rs = load_rc_params(str(summary_path))
    A_values = np.array([0.5, 1.0, 1.5])
    chi2_values = []
    model_curves = []
    for A_scale in A_values:
        disk, bulge, halo, vphi_total = build_components(a_b, a_d, b_d, gamma_mean, Mb, Md, rho0, rs, A_scale)
        v_model = vphi_total(R, disk, bulge, halo)
        chi2_values.append(chi2_rc(R, V, E, v_model))
        model_curves.append(v_model)
    chi2_values = np.array(chi2_values)
    out = repo / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(R, V, yerr=E, fmt="o", ms=3.5, capsize=2.0, label="Beordo2024 RC data")
    colors = ["C1", "C2", "C3"]
    for i, A_scale in enumerate(A_values):
        ax.plot(R, model_curves[i], color=colors[i % len(colors)], lw=1.8, label=f"gNFW model A={A_scale:.2f}")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("v_c [km/s]")
    ax.set_title("Rotation curve test for spherical gNFW halo (gamma free, A scaling)")
    ax.legend()
    fig.tight_layout()
    pdf = out / "rc_gnfw_gammaA_test.pdf"
    png = out / "rc_gnfw_gammaA_test.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    txt = out / "rc_gnfw_gammaA_test.txt"
    with open(txt, "w") as f:
        f.write(f"RC file {rc_path}\n")
        f.write(f"rc_nuts_summary {summary_path}\n")
        f.write(f"gamma_mean {gamma_mean:.6f}\n")
        f.write(f"a_b {a_b:.6f} a_d {a_d:.6f} b_d {b_d:.6f}\n")
        f.write(f"logMb {logMb:.6f} logMd {logMd:.6f}\n")
        f.write(f"logrho0 {logrho0:.6f} rs {rs:.6f}\n")
        for i, A_scale in enumerate(A_values):
            f.write(f"A {A_scale:.3f} chi2 {chi2_values[i]:.3f}\n")
    print(str(pdf))
    print(str(txt))

if __name__ == "__main__":
    main()
