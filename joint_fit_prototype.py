
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, ".")
from src.models.vphi_axisym import vcirc

def pick(df, names, default=None, dtype=float):
    for n in names:
        if n in df:
            return df[n].to_numpy(dtype=dtype)
    if default is None:
        raise KeyError(f"None of the columns {names} found.")
    return np.array(default, dtype=dtype)

def load_tuned(path="results/joint_mle_params.txt"):
    p = Path(path)
    out = {}
    if not p.exists(): 
        return out
    for line in p.read_text().splitlines():
        if "=" not in line: 
            continue
        k,v = line.split("=",1)
        k = k.strip().lower()
        v = v.strip().split()[0]
        try:
            out[k] = float(v)
        except:
            pass
    return out

# --- data ---
rc_csv = Path("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
df = pd.read_csv(rc_csv)
R = pick(df, ["R_kpc","R","Rkpc"])
V = pick(df, ["Vphi_kms","V_kms","Vphi"])
e = df["eVphi_kms"].to_numpy(float) if "eVphi_kms" in df else np.full_like(V, 5.0)

# --- base params ---
params = {
    "bulge": {"Mb": 1.0e8,  "b": 0.5},            # Msun, kpc
    "disk" : {"Md": 6.2e10, "a": 6.5, "b": 0.26}, # Msun, kpc, kpc
    "halo" : {"vh": 180.0,  "rh": 18.9, "gamma": 1.3, "qz": 0.9},
}

# --- override from tuned file, if present (e.g., vh=267, rh=12.0) ---
tuned = load_tuned()
for k in ("vh","rh","gamma","qz"):
    if k in tuned:
        params["halo"][k] = tuned[k]

# --- model curve ---
Rgrid = np.linspace(max(3.0, np.nanmin(R)), min(25.0, np.nanmax(R)), 400)
Vmod  = vcirc(Rgrid, 0.0, params)
vc_solar = float(vcirc(8.2, 0.0, params))

# --- plot RC ---
fig, ax = plt.subplots(figsize=(7.2,4.8), dpi=150)
ax.scatter(R, V, s=8, alpha=0.7, label="data")
ax.plot(Rgrid, Vmod, lw=2.2, label=f"model (vh={params['halo']['vh']:.0f}, rh={params['halo']['rh']:.1f})")
ax.set_title("RC joint-proto fit")
ax.set_xlabel("R [kpc]")
ax.set_ylabel(r"$V_\phi$ [km/s]")
ax.legend()
fig.savefig("results/joint_rc_fit.png", bbox_inches="tight")

# --- quick stream preview (keeps layout non-degenerate) ---
g = pd.read_csv("data/gd1/gd1_lite.csv")
phi1 = g.get("phi1_deg", g.iloc[:,0]).to_numpy()
mu1  = g.get("mu_phi1_masyr", g.iloc[:,1]).to_numpy()

phi1_clean = np.linspace(np.nanpercentile(phi1,5), np.nanpercentile(phi1,95), 300)
R_path = 14.0 + 2.0*np.sin(np.deg2rad(phi1_clean))
V_path = vcirc(R_path, 0.0, params)

fig2, ax2 = plt.subplots(1,2, figsize=(11,4), dpi=150)
ax2[0].scatter(phi1, mu1, s=4, alpha=0.5, label="GD-1 (Gaia DR3)")
ax2[0].set_xlabel(r"$\phi_1$ [deg]")
ax2[0].set_ylabel(r"$\mu_{\phi_1}$ [mas/yr]")
ax2[0].set_title("GD-1 sky-plane (preview)")
ax2[0].legend(loc="best", markerscale=2)
ax2[1].plot(phi1_clean, V_path, lw=2, label=r"$V_c(R(\phi_1))$")
ax2[1].set_xlabel(r"$\phi_1$ [deg]")
ax2[1].set_ylabel(r"$V_c$ [km/s]")
ax2[1].set_title("Stream-driven shape sensitivity (preview)")
ax2[1].legend()
fig2.savefig("results/joint_stream_fit.png", bbox_inches="tight")

with open("results/debug_vcirc.txt","w") as f:
    f.write(f"Vc(8.2 kpc) = {vc_solar:.3f} km/s\n")
