import sys
from pathlib import Path
import numpy as np
import pandas as pd

def read_rc(path):
    df = pd.read_csv(path).sort_values("R_kpc").reset_index(drop=True)
    R = df["R_kpc"].to_numpy(float)
    y = df["Vphi_kms"].to_numpy(float)
    sigma = df["sigma_obs_kms"].to_numpy(float)
    return R, y, sigma

def read_means(summary_path):
    df = pd.read_csv(summary_path)
    name_col = df.columns[0]
    df[name_col] = df[name_col].astype(str).str.strip()
    lookup = {str(df[name_col].iloc[i]).strip(): i for i in range(df.shape[0])}
    def mean_of(name):
        return float(df.iloc[lookup[name]]["mean"])
    s = {k: mean_of(k) for k in ["a_b","a_d","b_d","logMb","logMd","logrho0","rs","gamma"]}
    Mb = float(np.exp(s["logMb"]))
    Md = float(np.exp(s["logMd"]))
    rho0 = float(np.exp(s["logrho0"]))
    return float(s["a_b"]), float(s["a_d"]), float(s["b_d"]), Mb, Md, rho0, float(s["rs"]), float(s["gamma"])

def stats(tag, d):
    d = np.asarray(d, float)
    print(tag,
          "max_abs", float(np.max(np.abs(d))),
          "rms", float(np.sqrt(np.mean(d*d))),
          "med_abs", float(np.median(np.abs(d))))

def main():
    repo = Path(".").resolve()
    rc_path = repo / "data" / "rotation_curve" / "beordo2024_allstars_rc_ready.csv"
    if not rc_path.exists():
        rc_path = repo / "data" / "rotation_curve" / "beordo2024" / "beordo2024_allstars_rc_ready.csv"
    summ = repo / "results" / "rc_nuts_summary.csv"

    R, y, _ = read_rc(rc_path)
    a_b, a_d, b_d, Mb, Md, rho0, rs, gmean = read_means(summ)

    # axisym model
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.vphi_axisym import DiskMN, BulgeHernquist, Halo_gNFW, vphi_total
    disk = DiskMN(M=Md, a=a_d, b=b_d)
    bulge = BulgeHernquist(M=Mb, a=a_b)

    # nuts physical model
    import jax.numpy as jnp
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import rc_numpyro_nuts as rn  # reads its CSV at import time (should exist)

    Rj = jnp.asarray(R)

    # use the A_best from your GP-fixed grid as a test point
    A_eff = 0.41
    test_gammas = [gmean, 1.8, 1.875]

    print("Using params from rc_nuts_summary.csv means")
    print("a_b,a_d,b_d", a_b, a_d, b_d)
    print("Mb,Md,rho0,rs", Mb, Md, rho0, rs)
    print("Compare at A_eff", A_eff, "gammas", test_gammas)
    print("If models match, differences should be ~< 1 km/s everywhere.\n")

    for g in test_gammas:
        halo = Halo_gNFW(rho0=rho0*A_eff, rs=rs, gamma=float(g), q=1.0)
        mu_axis = np.asarray(vphi_total(R, disk, bulge, halo), float)

        mu_nuts = np.asarray(rn.vphi_model(Rj, Md, a_d, b_d, Mb, a_b, rho0*A_eff, rs, float(g)), float)

        d = mu_axis - mu_nuts
        print("gamma", float(g))
        stats("  axisym - nuts:", d)
        print("  head3 axisym", mu_axis[:3])
        print("  head3 nuts  ", mu_nuts[:3])
        print("  tail3 axisym", mu_axis[-3:])
        print("  tail3 nuts  ", mu_nuts[-3:])
        print("")

if __name__ == "__main__":
    main()
