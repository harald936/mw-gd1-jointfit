
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0,".")
from src.models.vphi_axisym import vcirc

Path("results").mkdir(exist_ok=True)

def pick(df, cols, dtype=float):
    for c in cols:
        if c in df:
            return df[c].to_numpy(dtype=dtype)
    return df.iloc[:,0].to_numpy(dtype=dtype)

def read_rc_mle(path="results/joint_mle_params.txt"):
    vals={}
    for ln in Path(path).read_text().splitlines():
        if "=" in ln:
            k,v = ln.split("=",1); vals[k.strip().lower()] = v.strip().split()[0]
    f = lambda k, d: float(vals.get(k, d))
    return {"vh": f("vh","230"), "rh": f("rh","18.0"), "gamma": f("gamma","1.2"),
            "qz": f("qz","0.9"), "sd": f("disk_scale","1.0"), "a": f("disk_a","6.5"),
            "sb": f("bulge_scale","1.0")}

rc = read_rc_mle()
bulge_base = {"Mb": rc["sb"]*1.0e8, "b": 0.5}
disk_base  = {"Md": rc["sd"]*6.2e10, "a": rc["a"], "b": 0.26}
halo_base  = {"vh": rc["vh"], "rh": rc["rh"], "gamma": rc["gamma"], "qz": rc["qz"]}

def params_with_qz(qz):
    h = dict(halo_base); h["qz"] = float(qz)
    return {"bulge": bulge_base, "disk": disk_base, "halo": h}

R0, V0 = 8.2, 230.0

g = pd.read_csv("data/gd1/gd1_lite.csv")
phi1 = pick(g, ["phi1_deg","phi1"], dtype=float)
mu1  = pick(g, ["mu_phi1_masyr","pm_phi1","pm_phi1_masyr"], dtype=float)
mask = np.isfinite(phi1) & np.isfinite(mu1)
phi1, mu1 = phi1[mask], mu1[mask]

nbin = 24
edges = np.linspace(np.nanpercentile(phi1,5), np.nanpercentile(phi1,95), nbin+1)
cent  = 0.5*(edges[:-1]+edges[1:])
med   = np.full(nbin, np.nan)
sig   = np.full(nbin, np.nan)
for i in range(nbin):
    m = (phi1>=edges[i]) & (phi1<edges[i+1])
    if np.count_nonzero(m) >= 20:
        vals = mu1[m]
        med[i] = np.nanmedian(vals)
        sig[i] = 1.4826*np.nanmedian(np.abs(vals - med[i]))/np.sqrt(np.count_nonzero(m))
ok = np.isfinite(med) & np.isfinite(sig)
phi_b, mu_b, err_b = cent[ok], med[ok], np.clip(sig[ok], 0.02, None)
phi_mid = np.nanmedian(phi_b)

def rpath_from_phi1(ph):
    return 14.0 + 2.0*np.sin(np.deg2rad(ph))
def zpath_from_phi1(ph):
    return 0.8*np.sin(np.deg2rad(ph))

def S_ref_for(phi):
    return vcirc(rpath_from_phi1(phi), zpath_from_phi1(phi)*0.0, params_with_qz(1.0))

def gd1_model(phi, qz, a0, a1, c):
    pars = params_with_qz(qz)
    Vc = vcirc(rpath_from_phi1(phi), zpath_from_phi1(phi), pars)
    S_ref = S_ref_for(phi)
    Sc = (Vc / np.where(np.isfinite(S_ref), S_ref, np.nanmedian(S_ref))) - 1.0
    return a0 + a1*(phi - phi_mid) + c*Sc

def loss(theta):
    qz, a0, a1, c = theta
    mu_mod = gd1_model(phi_b, qz, a0, a1, c)
    r_gd1 = (mu_mod - mu_b)/err_b
    pars = params_with_qz(qz)
    r_rc  = (vcirc(R0, 0.0, pars) - V0)/3.0
    r_qz  = (qz - 0.9)/0.25
    return np.concatenate([r_gd1, np.atleast_1d(r_rc), np.atleast_1d(r_qz)])

theta0 = np.array([rc["qz"], np.nanmedian(mu_b), 0.0, 1.0])
lb = np.array([0.6, -50.0, -2.0, -10.0])
ub = np.array([1.3,  50.0,  2.0,  10.0])
sol = least_squares(loss, theta0, bounds=(lb,ub), loss="soft_l1", f_scale=2.0, max_nfev=5000)
qz_fit, a0_fit, a1_fit, c_fit = sol.x

# RC panel
df = pd.read_csv("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
R = (df["R_kpc"] if "R_kpc" in df else df.iloc[:,0]).to_numpy(float)
V = (df["Vphi_kms"] if "Vphi_kms" in df else df.iloc[:,1]).to_numpy(float)
e = df["eVphi_kms"].to_numpy(float) if "eVphi_kms" in df else np.full_like(V,5.0)
Rgrid = np.linspace(max(5.0, np.nanmin(R)), min(19.0, np.nanmax(R)), 500)
pars_qz = params_with_qz(qz_fit)
Vgrid = vcirc(Rgrid, 0.0, pars_qz)

fig, ax = plt.subplots(figsize=(7.4,4.8), dpi=150)
ax.scatter(R, V, s=8, alpha=0.35, label="data")
ax.plot(Rgrid, Vgrid, lw=2.6, label=f"model (vh={halo_base['vh']:.0f}, rh={halo_base['rh']:.1f}, γ={halo_base['gamma']:.2f}, qz={qz_fit:.2f})")
ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"$V_\phi$ [km/s]")
ax.set_title("RC + GD-1 joint (qz from stream)")
ax.legend()
fig.savefig("results/joint_rc_fit.png", bbox_inches="tight")

# Stream panel
phi_dense = np.linspace(phi_b.min(), phi_b.max(), 600)
mu_fit = gd1_model(phi_dense, qz_fit, a0_fit, a1_fit, c_fit)

fig2, ax2 = plt.subplots(1,2, figsize=(11,4), dpi=150)
ax2[0].scatter(phi1, mu1, s=2, alpha=0.3, label="GD-1 (DR3)")
ax2[0].errorbar(phi_b, mu_b, yerr=err_b, fmt="o", ms=3.5, alpha=0.9, label="binned median")
ax2[0].plot(phi_dense, mu_fit, lw=2.2, label=f"fit (qz={qz_fit:.2f})")
ax2[0].set_xlabel(r"$\phi_1$ [deg]"); ax2[0].set_ylabel(r"$\mu_{\phi_1}$ [mas/yr]")
ax2[0].set_title("GD-1 proper-motion trend (preview)")
ax2[0].legend(loc="best", markerscale=2)

Vc_q = vcirc(rpath_from_phi1(phi_dense), zpath_from_phi1(phi_dense), params_with_qz(qz_fit))
Vc_s = vcirc(rpath_from_phi1(phi_dense), zpath_from_phi1(phi_dense), params_with_qz(1.0))
ax2[1].plot(phi_dense, Vc_s, lw=1.5, label="spherical (qz=1.0)")
ax2[1].plot(phi_dense, Vc_q, lw=2.2, label=f"fitted (qz={qz_fit:.2f})")
ax2[1].set_xlabel(r"$\phi_1$ [deg]"); ax2[1].set_ylabel(r"$V_c$ [km/s]")
ax2[1].set_title("Vc along GD-1 path (proxy)")
ax2[1].legend()
fig2.savefig("results/joint_stream_fit.png", bbox_inches="tight")

with open("results/joint_qz.txt","w") as f:
    f.write(f"qz_fit={qz_fit:.4f}\n")
    f.write(f"a0={a0_fit:.4f}\n")
    f.write(f"a1={a1_fit:.4f}\n")
    f.write(f"c={c_fit:.4f}\n")
print("Wrote results/joint_rc_fit.png, results/joint_stream_fit.png, results/joint_qz.txt")
