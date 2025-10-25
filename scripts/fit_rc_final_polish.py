
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution, least_squares
sys.path.insert(0,".")
from src.models.vphi_axisym import vcirc

Path("results").mkdir(exist_ok=True)

def pick(df, cols, fallback=None, dtype=float):
    for c in cols:
        if c in df: return df[c].to_numpy(dtype=dtype)
    if fallback is None: raise KeyError(f"Missing columns {cols}")
    return np.array(fallback, dtype=dtype)

df = pd.read_csv("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
Rraw = pick(df, ["R_kpc","R","Rkpc"])
Vraw = pick(df, ["Vphi_kms","V_kms","Vphi"])
eraw = df["eVphi_kms"].to_numpy(float) if "eVphi_kms" in df else np.full_like(Vraw, 5.0)
if np.nanmedian(Rraw) > 300: Rraw = Rraw/1000.0
if np.nanmedian(Vraw) > 900: Vraw = Vraw/1000.0

mask = (Rraw>=6.0) & (Rraw<=18.0) & np.isfinite(Vraw)
R, V, e = Rraw[mask], Vraw[mask], eraw[mask]
order = np.argsort(R); R, V, e = R[order], V[order], e[order]

win = max(7, int(0.10*len(R)))
Vmed = pd.Series(V).rolling(win, center=True, min_periods=max(3,win//3)).median().to_numpy()
Vtrend = np.where(np.isfinite(Vmed), Vmed, V)

gmu, gsig = 1.29, 0.40
try:
    summ = pd.read_csv("results/rc_nuts_summary.csv")
    cols = {c.lower(): c for c in summ.columns}
    if "param" in cols and ("mean" in cols or "median" in cols):
        row = summ[summ[cols["param"]].str.lower().str.contains("gamma")].iloc[0]
        gmu = float(row.get("mean", row.get("median", gmu)))
        gsig = float(row.get("sd", gsig))
except Exception:
    pass
gsig = max(gsig, 0.25)

Mb0 = 1.0e8
Md0 = 6.2e10
a0, b0 = 6.5, 0.26
R0, V0 = 8.2, 230.0
Rplateau, Vplateau = 13.0, 227.0

def make_params(theta):
    vh, rh, gamma, sd, a, sb = theta
    bulge = {"Mb": sb*Mb0, "b": 0.5}
    disk  = {"Md": sd*Md0, "a": a, "b": b0}
    halo  = {"vh": vh, "rh": rh, "gamma": gamma, "qz": 0.9}
    return {"bulge": bulge, "disk": disk, "halo": halo}

def reg_slope(pars):
    Rg = np.linspace(10.0, 15.0, 60)
    Vg = vcirc(Rg, 0.0, pars)
    dVdR = np.gradient(Vg, Rg)
    return np.mean(dVdR**2)

def prior_penalty(theta):
    vh, rh, gamma, sd, a, sb = theta
    p  = ((sd - 1.0)/0.18)**2
    p += ((a  - a0)/0.6 )**2
    p += ((sb - 1.0)/0.5 )**2
    p += ((rh - 18.0)/6.0)**2
    p += ((gamma - gmu)/gsig)**2
    return p

def loss_rc(theta):
    pars = make_params(theta)
    Vmod = vcirc(R, 0.0, pars)
    w = 1.0/np.clip(e, 4.0, None)
    r = (Vmod - Vtrend)*w
    r0 = (vcirc(R0,0.0,pars) - V0)/3.0
    rp = (vcirc(Rplateau,0.0,pars) - Vplateau)/3.0
    slope = reg_slope(pars)          # km/s/kpc squared
    return np.sum(r*r) + r0*r0 + rp*rp + 4.0*slope + prior_penalty(theta)

bounds = [
    (170, 300),   # vh
    (10.0, 28.0), # rh
    (0.7,  1.6),  # gamma
    (0.9,  1.2),  # sd
    (5.6,  7.2),  # a
    (0.8,  1.3),  # sb
]

res_de = differential_evolution(loss_rc, bounds=bounds, seed=42, polish=False, maxiter=160, popsize=28, updating='deferred', workers=1)
theta0 = res_de.x

from scipy.optimize import least_squares
def resid(theta):
    pars = make_params(theta)
    Vmod = vcirc(R,0.0,pars)
    w = 1.0/np.clip(e, 4.0, None)
    r = (Vmod - Vtrend)*w
    r0 = (vcirc(R0,0.0,pars) - V0)/3.0
    rp = (vcirc(Rplateau,0.0,pars) - Vplateau)/3.0
    p  = prior_penalty(theta)
    s  = reg_slope(pars)
    return np.concatenate([r, np.atleast_1d(r0), np.atleast_1d(rp), np.full(4, np.sqrt(p/4.0)), np.full(2, np.sqrt(4.0*s/2.0))])

lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
ls = least_squares(resid, theta0, bounds=(lb,ub), loss="soft_l1", f_scale=3.2, max_nfev=9000)
vh, rh, gamma, sd, a, sb = ls.x
pars_final = make_params(ls.x)

Rgrid = np.linspace(5.0, 19.0, 600)
Vgrid = vcirc(Rgrid, 0.0, pars_final)
Vsun  = float(vcirc(R0, 0.0, pars_final))

fig, ax = plt.subplots(figsize=(7.6,4.9), dpi=150)
ax.scatter(Rraw, Vraw, s=9, alpha=0.26, label="data")
ax.plot(R, Vtrend, lw=2.0, alpha=0.65, label="running median")
ax.plot(Rgrid, Vgrid, lw=3.0, color="#d9730d", label=f"model (vh={vh:.0f}, rh={rh:.1f}, γ={gamma:.2f}, sd={sd:.2f}, a={a:.2f}, sb={sb:.2f})")
ax.set_title("RC fit (final polish: fit to running-median 6–18 kpc)")
ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"$V_\phi$ [km/s]")
ax.legend()
fig.savefig("results/joint_rc_fit.png", bbox_inches="tight")

with open("results/joint_mle_params.txt","w") as f:
    f.write(f"vh={vh:.3f} km/s\nrh={rh:.3f} kpc\ngamma={gamma:.3f}\nqz={0.900:.3f}\n")
    f.write(f"disk_scale={sd:.3f}\ndisk_a={a:.3f} kpc\nbulge_scale={sb:.3f}\n")

with open("results/debug_vcirc.txt","w") as f:
    f.write(f"Vc(8.2 kpc) = {Vsun:.3f} km/s\n")
    f.write(f"theta=[vh={vh:.3f}, rh={rh:.3f}, gamma={gamma:.3f}, sd={sd:.3f}, a={a:.3f}, sb={sb:.3f}]\n")

g = pd.read_csv("data/gd1/gd1_lite.csv")
phi1 = pick(g, ["phi1_deg","phi1"], dtype=float)
mu1  = pick(g, ["mu_phi1_masyr","pm_phi1","pm_phi1_masyr"], dtype=float)
if np.nanmedian(np.abs(mu1)) > 50: mu1 = mu1 - np.nanmedian(mu1)
phi1_clean = np.linspace(np.nanpercentile(phi1,5), np.nanpercentile(phi1,95), 300)
R_path = 14.0 + 2.0*np.sin(np.deg2rad(phi1_clean))
V_path = vcirc(R_path, 0.0, pars_final)
fig2, ax2 = plt.subplots(1,2, figsize=(11,4), dpi=150)
ax2[0].scatter(phi1, mu1, s=2, alpha=0.35, label="GD-1 (Gaia DR3)")
ax2[0].set_xlabel(r"$\phi_1$ [deg]"); ax2[0].set_ylabel(r"$\mu_{\phi_1}$ [mas/yr]")
ax2[0].set_title("GD-1 sky-plane (preview)"); ax2[0].legend(loc="best", markerscale=3)
ax2[1].plot(phi1_clean, V_path, lw=2, label=r"$V_c(R(\phi_1))$")
ax2[1].set_xlabel(r"$\phi_1$ [deg]"); ax2[1].set_ylabel(r"$V_c$ [km/s]")
ax2[1].set_title("Stream-driven shape sensitivity (preview)"); ax2[1].legend()
fig2.savefig("results/joint_stream_fit.png", bbox_inches="tight")
print("Wrote results/joint_rc_fit.png, joint_stream_fit.png, joint_mle_params.txt, debug_vcirc.txt")
