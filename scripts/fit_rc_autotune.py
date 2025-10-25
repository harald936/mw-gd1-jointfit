
import numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
from pathlib import Path
from scipy.optimize import differential_evolution, least_squares
sys.path.insert(0,".")
from src.models.vphi_axisym import vcirc

rng = np.random.default_rng(42)
Path("results").mkdir(exist_ok=True)

def pick(df, cols, fallback=None, dtype=float):
    for c in cols:
        if c in df: 
            return df[c].to_numpy(dtype=dtype)
    if fallback is None: 
        raise KeyError(f"Missing columns {cols}")
    return np.array(fallback, dtype=dtype)

df = pd.read_csv("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
R = pick(df, ["R_kpc","R","Rkpc"])
V = pick(df, ["Vphi_kms","V_kms","Vphi"])
e = df["eVphi_kms"].to_numpy(float) if "eVphi_kms" in df else np.full_like(V, 5.0)

if np.nanmedian(R) > 300:  R = R/1000.0
if np.nanmedian(V) > 900:  V = V/1000.0

Mb0 = 1.0e8
Md0 = 6.2e10
a0, b0 = 6.5, 0.26

R0, V0 = 8.2, 230.0

# theta = [vh, rh, gamma, sd, a, sb]
def make_params(theta):
    vh, rh, gamma, sd, a, sb = theta
    bulge = {"Mb": sb*Mb0, "b": 0.5}
    disk  = {"Md": sd*Md0, "a": a, "b": b0}
    halo  = {"vh": vh, "rh": rh, "gamma": gamma, "qz": 0.9}
    return {"bulge": bulge, "disk": disk, "halo": halo}

# Gaussian priors (soft). sigmas chosen to be informative but not tight.
def prior_penalty(theta):
    vh, rh, gamma, sd, a, sb = theta
    p  = ((sd - 1.0)/0.25)**2          # disk scale ~ N(1.0, 0.25)
    p += ((a  - a0)/0.8 )**2           # disk a   ~ N(6.5, 0.8 kpc)
    p += ((sb - 1.0)/0.5 )**2          # bulge scale ~ N(1.0, 0.5)
    p += ((rh - 18.0)/6.0)**2          # rh ~ N(18, 6 kpc)
    p += ((gamma - 1.2)/0.5)**2        # gamma ~ N(1.2, 0.5)
    return p

def loss_rc(theta):
    pars = make_params(theta)
    Vmod = vcirc(R, 0.0, pars)
    w = 1.0/np.clip(e,3.0,None)
    r = (Vmod - V)*w
    r0 = (vcirc(R0,0.0,pars) - V0)/3.0
    return np.sum(r*r) + r0*r0 + prior_penalty(theta)

bounds = [
    (140, 420),   # vh
    (6.0,  40.0), # rh (avoid unphysical tiny values now that priors help)
    (0.2,  1.6),  # gamma
    (0.6,  1.6),  # sd
    (4.5,  7.5),  # a
    (0.5,  2.0),  # sb
]

res_de = differential_evolution(loss_rc, bounds=bounds, seed=42, polish=False, maxiter=180, popsize=28, updating='deferred', workers=1)
theta0 = res_de.x

def resid(theta):
    pars = make_params(theta)
    Vmod = vcirc(R,0.0,pars)
    w = 1.0/np.clip(e,3.0,None)
    r = (Vmod - V)*w
    r0 = (vcirc(R0,0.0,pars) - V0)/3.0
    # convert prior into residuals by splitting the scalar penalty into few pseudo-observations
    p = prior_penalty(theta)
    return np.concatenate([r, np.atleast_1d(r0), np.full(5, np.sqrt(p/5.0))])

lb = np.array([b[0] for b in bounds])
ub = np.array([b[1] for b in bounds])
ls = least_squares(resid, theta0, bounds=(lb,ub), loss="soft_l1", f_scale=4.0, max_nfev=8000)
vh, rh, gamma, sd, a, sb = ls.x
pars_final = make_params(ls.x)

Rgrid = np.linspace(max(3.0,np.nanmin(R)), min(25.0,np.nanmax(R)), 600)
Vgrid = vcirc(Rgrid,0.0,pars_final)
Vsun  = float(vcirc(R0,0.0,pars_final))

fig, ax = plt.subplots(figsize=(7.5,4.8), dpi=150)
ax.scatter(R, V, s=10, alpha=0.75, label="data")
ax.plot(Rgrid, Vgrid, lw=2.2, label=f"model (vh={vh:.0f}, rh={rh:.1f}, gamma={gamma:.2f}, sd={sd:.2f}, a={a:.2f}, sb={sb:.2f})")
ax.set_title("RC fit (priors on baryons+halo; Solar-circle regularized)")
ax.set_xlabel("R [kpc]")
ax.set_ylabel(r"$V_\phi$ [km/s]")
ax.legend()
fig.savefig("results/joint_rc_fit.png", bbox_inches="tight")

with open("results/joint_mle_params.txt","w") as f:
    f.write(f"vh={vh:.3f} km/s\n")
    f.write(f"rh={rh:.3f} kpc\n")
    f.write(f"gamma={gamma:.3f}\n")
    f.write(f"qz={0.900:.3f}\n")
    f.write(f"disk_scale={sd:.3f}\n")
    f.write(f"disk_a={a:.3f} kpc\n")
    f.write(f"bulge_scale={sb:.3f}\n")

with open("results/debug_vcirc.txt","w") as f:
    f.write(f"Vc(8.2 kpc) = {Vsun:.3f} km/s\n")
    f.write(f"theta=[vh={vh:.3f}, rh={rh:.3f}, gamma={gamma:.3f}, sd={sd:.3f}, a={a:.3f}, sb={sb:.3f}]\n")

# Stream preview with same params
g = pd.read_csv("data/gd1/gd1_lite.csv")
phi1 = pick(g, ["phi1_deg","phi1"], dtype=float)
mu1  = pick(g, ["mu_phi1_masyr","pm_phi1","pm_phi1_masyr"], dtype=float)
if np.nanmedian(np.abs(mu1)) > 50: mu1 = mu1 - np.nanmedian(mu1)

phi1_clean = np.linspace(np.nanpercentile(phi1,5), np.nanpercentile(phi1,95), 300)
R_path = 14.0 + 2.0*np.sin(np.deg2rad(phi1_clean))
V_path = vcirc(R_path, 0.0, pars_final)

fig2, ax2 = plt.subplots(1,2, figsize=(11,4), dpi=150)
ax2[0].scatter(phi1, mu1, s=2, alpha=0.4, label="GD-1 (Gaia DR3)")
ax2[0].set_xlabel(r"$\phi_1$ [deg]")
ax2[0].set_ylabel(r"$\mu_{\phi_1}$ [mas/yr]")
ax2[0].set_title("GD-1 sky-plane (preview)")
ax2[0].legend(loc="best", markerscale=3)
ax2[1].plot(phi1_clean, V_path, lw=2, label=r"$V_c(R(\phi_1))$")
ax2[1].set_xlabel(r"$\phi_1$ [deg]")
ax2[1].set_ylabel(r"$V_c$ [km/s]")
ax2[1].set_title("Stream-driven shape sensitivity (preview)")
ax2[1].legend()
fig2.savefig("results/joint_stream_fit.png", bbox_inches="tight")
print("Wrote results/joint_rc_fit.png, joint_stream_fit.png, joint_mle_params.txt, debug_vcirc.txt")
