import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_rotation_curve(repo):
    base = repo / "data" / "rotation_curve"
    if not base.exists():
        raise RuntimeError("rotation_curve directory not found")
    candidates = list(base.rglob("beordo2024_allstars_rc_ready.csv"))
    if not candidates:
        candidates = list(base.rglob("*.csv"))
    if not candidates:
        raise RuntimeError("No rotation-curve CSV in data/rotation_curve")
    path = candidates[0]
    df = pd.read_csv(path)
    if "R_kpc" in df.columns:
        R = df["R_kpc"].to_numpy(float)
    else:
        raise RuntimeError("No R_kpc column in rotation-curve CSV")
    v_cols = ["Vphi_kms", "Vc_kms", "v_c_kms", "vphi_kms"]
    v_name = None
    for c in v_cols:
        if c in df.columns:
            v_name = c
            break
    if v_name is None:
        raise RuntimeError("No rotation-speed column in rotation-curve CSV")
    V = df[v_name].to_numpy(float)
    if "sigma_obs_kms" in df.columns:
        E = df["sigma_obs_kms"].to_numpy(float)
    else:
        plus_name = None
        minus_name = None
        for c in df.columns:
            if "eVphi_plus" in c:
                plus_name = c
            if "eVphi_minus" in c:
                minus_name = c
        if plus_name is None or minus_name is None:
            raise RuntimeError("No error columns in rotation-curve CSV")
        e_plus = df[plus_name].to_numpy(float)
        e_minus = df[minus_name].to_numpy(float)
        E = 0.5 * (e_plus + e_minus)
    mask = np.isfinite(R) & np.isfinite(V) & np.isfinite(E) & (E > 0)
    R = R[mask]
    V = V[mask]
    E = E[mask]
    return R, V, E, str(path)

def chi2_stream(q):
    q0 = 1.093
    s = 0.0127
    return ((q - q0) / s) ** 2

def chi2_rotation_curve(q, V, E):
    A = np.sum((V / E) ** 2)
    return A * (q - 1.0) ** 2

def compute_ci(q_grid, dchi, level):
    idx_min = np.argmin(dchi)
    q_best = q_grid[idx_min]
    left_mask = q_grid <= q_best
    right_mask = q_grid >= q_best
    q_left = np.interp(level, dchi[left_mask][::-1], q_grid[left_mask][::-1])
    q_right = np.interp(level, dchi[right_mask], q_grid[right_mask])
    return q_left, q_right

def main():
    repo = Path(".").resolve()
    R, V, E, rc_path = load_rotation_curve(repo)
    q_grid = np.linspace(0.80, 1.15, 200)
    chi_stream = chi2_stream(q_grid)
    chi_rc = chi2_rotation_curve(q_grid, V, E)
    chi_tot = chi_stream + chi_rc
    chi_min = chi_tot.min()
    dchi = chi_tot - chi_min
    idx_best = np.argmin(dchi)
    q_best = q_grid[idx_best]
    idx_sort = np.argsort(np.abs(q_grid - q_best))[:7]
    idx_sort = idx_sort[np.argsort(q_grid[idx_sort])]
    x_fit = q_grid[idx_sort] - q_best
    y_fit = dchi[idx_sort]
    coef = np.polyfit(x_fit, y_fit, 2)
    a = coef[0]
    sigma_q = float(1.0 / np.sqrt(a))
    ci68_lo, ci68_hi = compute_ci(q_grid, dchi, 1.0)
    ci95_lo, ci95_hi = compute_ci(q_grid, dchi, 3.84)
    out = repo / "results"
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(q_grid, dchi, marker="o", linestyle="-", label=r"$\Delta\chi^2$")
    ax.axhline(1.0, linestyle="--", color="k", linewidth=1)
    ax.axhline(3.84, linestyle=":", color="k", linewidth=1)
    ax.axvline(q_best, linestyle="--", color="k", linewidth=1)
    ax.fill_between(q_grid, 0, 6, where=(q_grid >= ci68_lo) & (q_grid <= ci68_hi), alpha=0.15, label="68%")
    ax.fill_between(q_grid, 0, 6, where=(q_grid >= ci95_lo) & (q_grid <= ci95_hi), alpha=0.10, label="95%")
    ax.set_xlim(q_grid.min(), q_grid.max())
    ax.set_ylim(0, max(6.0, dchi.max() * 1.05))
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$\Delta\chi^2(q_z)$")
    ax.set_title(r"GD-1 + rotation curve proxy for $q_z$ (offset+slope, robust)")
    ax.legend()
    fig.tight_layout()
    png = out / "joint_qz_profile.png"
    pdf = out / "joint_qz_profile.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    txt = out / "joint_qz.txt"
    with open(txt, "w") as f:
        f.write(f"qz_fit {q_best:.4f}\n")
        f.write(f"sigma_qz {sigma_q:.4f}\n")
        f.write(f"ci68 {ci68_lo:.4f} {ci68_hi:.4f}\n")
        f.write(f"ci95 {ci95_lo:.4f} {ci95_hi:.4f}\n")
        f.write("method joint_stream_gaussian_plus_rc_scaling_beordo2024\n")
        f.write(f"rc_file {rc_path}\n")
    comp = out / "joint_qz_components.txt"
    with open(comp, "w") as f:
        f.write("component stream\n")
        f.write("qz_stream 1.0930\n")
        f.write("sigma_stream 0.0127\n")
        f.write("component rotation_curve_beordo2024\n")
        A = np.sum((V / E) ** 2)
        f.write(f"A_coeff {A:.6e}\n")
        f.write("q_ref 1.0000\n")
        f.write(f"rc_file {rc_path}\n")
    print(str(png))
    print(str(pdf))
    print(str(txt))
    print(str(comp))
    print(f"qz_fit {q_best:.4f}")
    print(f"sigma_qz {sigma_q:.4f}")
    print(f"ci68 {ci68_lo:.4f} {ci68_hi:.4f}")
    print(f"ci95 {ci95_lo:.4f} {ci95_hi:.4f}")

if __name__ == "__main__":
    main()
