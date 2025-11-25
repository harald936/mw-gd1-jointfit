
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from gala.units import galactic as GAL_UNITS
from gala.potential import CompositePotential, HernquistPotential, MiyamotoNagaiPotential, LogarithmicPotential
from gala.dynamics import PhaseSpacePosition
from gala.coordinates import GD1Koposov10

SIGMA_FLOOR = 0.20
SIGMA_CAP = 0.60
MIN_PER_BIN = 30

def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        k = n.lower()
        if k in lower:
            return lower[k]
    return None

def load_gd1(repo):
    f = repo/"data/gd1/gd1_lite.csv"
    if not f.exists():
        f = repo/"data/gd1/gd1_raw_dr3.csv"
    df = pd.read_csv(f)
    c_phi1 = pick_col(df, ["phi1_deg","phi1","phi_1","stream_phi1","gd1_phi1"])
    c_pm1 = pick_col(df, ["pm_phi1","pmphi1","mu_phi1","mu_phi_1"])
    c_phi2 = pick_col(df, ["phi2_deg","phi2","phi_2","stream_phi2","gd1_phi2"])
    if c_phi1 is None or c_pm1 is None:
        raise RuntimeError("phi1/pm_phi1 column not found")
    mask = np.isfinite(df[c_phi1]) & np.isfinite(df[c_pm1])
    if c_phi2 is not None:
        mask &= np.isfinite(df[c_phi2]) & (np.abs(df[c_phi2]) < 1.0)
    if "p_member" in df.columns:
        mask &= df["p_member"] >= 0.8
    sub = df.loc[mask, [c_phi1, c_pm1]].copy()
    sub = sub.rename(columns={c_phi1: "phi1", c_pm1: "pm1"})
    sub = sub.sort_values("phi1")
    return sub

def bin_stream(phi, pm, nbins):
    lo, hi = np.percentile(phi, 2), np.percentile(phi, 98)
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.zeros(nbins)
    sig = np.zeros(nbins)
    ok = np.zeros(nbins, dtype=bool)
    for i in range(nbins):
        m = (phi >= edges[i]) & (phi < edges[i+1])
        n = np.count_nonzero(m)
        if n >= MIN_PER_BIN:
            v = pm[m]
            mmed = np.median(v)
            med[i] = mmed
            mad = np.median(np.abs(v - mmed))
            s = 1.4826 * mad
            if not np.isfinite(s) or s <= 0:
                s = SIGMA_FLOOR
            if s < SIGMA_FLOOR:
                s = SIGMA_FLOOR
            if s > SIGMA_CAP:
                s = SIGMA_CAP
            sig[i] = s
            ok[i] = True
    return centers[ok], med[ok], sig[ok]

def make_potential(qz):
    bulge = HernquistPotential(m=0.5e10*u.Msun, c=0.6*u.kpc, units=GAL_UNITS)
    disk = MiyamotoNagaiPotential(m=6.0e10*u.Msun, a=6.5*u.kpc, b=0.26*u.kpc, units=GAL_UNITS)
    halo = LogarithmicPotential(v_c=220*u.km/u.s, r_h=12*u.kpc, q1=1.0, q2=1.0, q3=float(qz), units=GAL_UNITS)
    return CompositePotential(bulge=bulge, disk=disk, halo=halo)

def orbit_linear_pm(qz, w0, phi_min, phi_max):
    pot = make_potential(qz)
    orb = pot.integrate_orbit(w0, dt=0.5*u.Myr, n_steps=8000)
    gal = orb.to_coord_frame(Galactocentric())
    icrs = gal.transform_to(ICRS())
    gd = icrs.transform_to(GD1Koposov10())
    phi = gd.phi1.to_value(u.deg)
    pm1 = gd.pm_phi1_cosphi2.to_value(u.mas/u.yr)
    mask = np.isfinite(phi) & np.isfinite(pm1) & (phi > phi_min - 5.0) & (phi < phi_max + 5.0)
    phi = phi[mask]
    pm1 = pm1[mask]
    if phi.size < 50:
        return None, None
    A = np.vstack([np.ones_like(phi), phi]).T
    coef, _, _, _ = np.linalg.lstsq(A, pm1, rcond=None)
    b0 = float(coef[0])
    b1 = float(coef[1])
    return b0, b1

def chi2_for_q(qz, phi_bin, pm_bin, sig_bin, w0):
    b0, b1 = orbit_linear_pm(qz, w0, phi_bin.min(), phi_bin.max())
    if b0 is None:
        return np.inf
    model = b0 + b1 * phi_bin
    r = (pm_bin - model) / sig_bin
    return float(np.sum(r*r))

def run():
    repo = Path.home()/"mw-gd1-jointfit"
    out = repo/"results"
    out.mkdir(parents=True, exist_ok=True)
    df = load_gd1(repo)
    phi = df["phi1"].to_numpy(dtype=float)
    pm1 = df["pm1"].to_numpy(dtype=float)
    nbins = 30
    phi_bin, pm_bin, sig_bin = bin_stream(phi, pm1, nbins)
    gd = GD1Koposov10()
    c0 = SkyCoord(phi1=np.median(phi)*u.deg, phi2=0.0*u.deg, distance=9.5*u.kpc,
                  pm_phi1_cosphi2=-8.0*u.mas/u.yr, pm_phi2=0.2*u.mas/u.yr,
                  radial_velocity=-130.0*u.km/u.s, frame=gd)
    c0g = c0.transform_to(Galactocentric())
    w0 = PhaseSpacePosition(c0g.cartesian.xyz, c0g.velocity.d_xyz)
    q_grid = np.linspace(0.80, 1.15, 23)
    chi = []
    qq = []
    for qz in q_grid:
        val = chi2_for_q(qz, phi_bin, pm_bin, sig_bin, w0)
        if np.isfinite(val):
            qq.append(qz)
            chi.append(val)
    qq = np.array(qq)
    chi = np.array(chi)
    chi = chi - chi.min()
    if qq.size < 5:
        raise RuntimeError("too few qz points")
    order = np.argsort(chi)
    k = max(5, int(0.8*order.size))
    sel = order[:k]
    p = np.poly1d(np.polyfit(qq[sel], chi[sel], 2))
    q_fine = np.linspace(qq.min(), qq.max(), 400)
    chi_fit = p(q_fine)
    if np.any(~np.isfinite(chi_fit)):
        q_plot = qq
        dchi = chi - chi.min()
    else:
        chi_fit = chi_fit - chi_fit.min()
        q_plot = q_fine
        dchi = chi_fit
    best_idx = int(np.argmin(dchi))
    q_best = float(q_plot[best_idx])
    dchi = dchi - dchi.min()
    mask68 = dchi <= 1.0 + 1e-6
    mask95 = dchi <= 3.84 + 1e-6
    if np.any(mask68):
        q68_min = float(q_plot[mask68].min())
        q68_max = float(q_plot[mask68].max())
    else:
        q68_min = float(q_plot.min())
        q68_max = float(q_plot.max())
    if np.any(mask95):
        q95_min = float(q_plot[mask95].min())
        q95_max = float(q_plot[mask95].max())
    else:
        q95_min = float(q_plot.min())
        q95_max = float(q_plot.max())
    sigma_q = 0.5 * (q68_max - q68_min)
    fig = plt.figure(figsize=(7.2,4.4), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(q_plot, dchi, marker="o", lw=1.5, label=r"$\Delta\chi^2$")
    ax.axhline(1.0, ls="--", lw=1.0, color="k")
    ax.axhline(3.84, ls=":", lw=1.0, color="k")
    ax.axvspan(q68_min, q68_max, alpha=0.15, label="68%")
    ax.axvspan(q95_min, q95_max, alpha=0.08, label="95%")
    ax.axvline(q_best, ls="-.", lw=1.0, color="k")
    ax.set_xlim(q_plot.min(), q_plot.max())
    ax.set_ylim(0.0, max(6.0, float(dchi.max())*1.1))
    ax.set_xlabel(r"$q_z$")
    ax.set_ylabel(r"$\Delta\chi^2(q_z)$")
    ax.set_title(r"GD-1 orbit-based proxy for $q_z$ (offset+slope, robust)")
    ax.legend()
    fig.tight_layout()
    out_png = out/"joint_qz_profile.png"
    out_pdf = out/"joint_qz_profile.pdf"
    out_txt = out/"joint_qz.txt"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    with open(out_txt, "w") as f:
        f.write(f"qz_fit {q_best:.4f}\n")
        f.write(f"sigma_qz {sigma_q:.4f}\n")
        f.write(f"ci68 {q68_min:.4f} {q68_max:.4f}\n")
        f.write(f"ci95 {q95_min:.4f} {q95_max:.4f}\n")
        f.write("method orbit_proxy_pm_linear_fit_quad_smooth\n")
    print(str(out_png))
    print(str(out_pdf))
    print(str(out_txt))

if __name__ == "__main__":
    run()
