from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Galactocentric, ICRS
from gala.potential import MiyamotoNagaiPotential, HernquistPotential, LogarithmicPotential, CompositePotential
from gala.dynamics import PhaseSpacePosition
import gala.coordinates as gc
from gala.units import galactic

def weighted_profile_chi2(x, resid, sigma):
    x = np.asarray(x, float)
    r = np.asarray(resid, float)
    s = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(r) & np.isfinite(s) & (s > 0)
    if m.sum() < 3:
        return 1e9
    x = x[m]
    r = r[m]
    w = 1.0 / (s[m] ** 2)
    X = np.vstack([np.ones_like(x), x]).T
    XT_W = (X.T * w)
    M = XT_W @ X
    b = XT_W @ r
    try:
        beta = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(M, b, rcond=None)[0]
    rprof = r - X @ beta
    return float(np.sum((rprof**2) * w))

def load_inputs(repo: Path | str = "."):
    repo = Path(repo).resolve()
    rc_csv = repo / "data" / "rotation_curve" / "beordo2024" / "beordo2024_allstars_rc_ready.csv"
    gd1_sum = repo / "data" / "gd1" / "gd1_summary.csv"

    rc = pd.read_csv(rc_csv).query("5.0 <= R_kpc <= 20.0").copy()
    R_rc = rc["R_kpc"].values * u.kpc
    Vphi = rc["Vphi_kms"].values * (u.km/u.s)
    sigmaV = rc["sigma_obs_kms"].values * (u.km/u.s)

    gd = pd.read_csv(gd1_sum)
    phi1 = gd["phi1_deg"].values * u.deg
    phi2m = gd["phi2_med"].values * u.deg
    phi2e = np.clip(gd["phi2_se"].values, 1e-3, None) * u.deg
    pm1m = gd["pm_phi1_med"].values * (u.mas/u.yr)
    pm1e = np.clip(gd["pm_phi1_se"].values, 1e-3, None) * (u.mas/u.yr)
    pm2m = gd["pm_phi2_med"].values * (u.mas/u.yr)
    pm2e = np.clip(gd["pm_phi2_se"].values, 1e-3, None) * (u.mas/u.yr)

    Md = 6.2e10 * u.Msun
    ad = 2.0 * u.kpc
    bd = 1.19 * u.kpc
    Mb = 1.0e8 * u.Msun
    ab = 2.0 * u.kpc

    R0 = 8.2 * u.kpc
    gc_frame = Galactocentric(galcen_distance=R0, galcen_v_sun=[11.1, 244.0, 7.25] * u.km/u.s)

    i_anchor = len(phi1) // 2
    d_anc = 10.0 * u.kpc
    rv_anc = 0.0 * u.km/u.s

    return dict(
        repo=repo,
        rc_csv=rc_csv,
        gd1_sum=gd1_sum,
        R_rc=R_rc,
        Vphi=Vphi,
        sigmaV=sigmaV,
        phi1=phi1,
        phi2m=phi2m,
        phi2e=phi2e,
        pm1m=pm1m,
        pm1e=pm1e,
        pm2m=pm2m,
        pm2e=pm2e,
        Md=Md, ad=ad, bd=bd, Mb=Mb, ab=ab,
        gc_frame=gc_frame,
        i_anchor=i_anchor,
        d_anc=d_anc,
        rv_anc=rv_anc,
    )

def build_potential(vh_kms=180.0, rh_kpc=15.0, qz=0.9, inputs=None):
    if inputs is None:
        inputs = load_inputs(".")
    D = MiyamotoNagaiPotential(m=inputs["Md"], a=inputs["ad"], b=inputs["bd"], units=galactic)
    B = HernquistPotential(m=inputs["Mb"], c=inputs["ab"], units=galactic)
    H = LogarithmicPotential(
        v_c=vh_kms * u.km/u.s,
        r_h=rh_kpc * u.kpc,
        q1=1.0, q2=1.0, q3=qz,
        units=galactic,
    )
    return CompositePotential(disk=D, bulge=B, halo=H)

def vcirc(pot, R_kpc):
    Vc = []
    for r in R_kpc:
        pos = np.array([r.to_value(u.kpc), 0.0, 0.0]) * u.kpc
        acc = pot.acceleration(pos)
        aR = abs(acc[0]).to(u.kpc / u.Myr**2)
        v = (r * aR) ** 0.5
        Vc.append(v.to_value(u.km / u.s))
    return np.array(Vc) * (u.km / u.s)

def anchor_phase_space(theta, inputs):
    phi1 = inputs["phi1"]
    phi2m = inputs["phi2m"]
    pm1m = inputs["pm1m"]
    pm2m = inputs["pm2m"]
    i_anchor = inputs["i_anchor"]
    gd1_c = gc.GD1Koposov10(
        phi1=phi1[i_anchor],
        phi2=phi2m[i_anchor],
        distance=inputs["d_anc"],
        pm_phi1_cosphi2=pm1m[i_anchor],
        pm_phi2=pm2m[i_anchor],
        radial_velocity=inputs["rv_anc"],
    )
    icrs = gd1_c.transform_to(ICRS())
    gcen = icrs.transform_to(inputs["gc_frame"])
    pos = np.array([gcen.x.to_value(u.kpc), gcen.y.to_value(u.kpc), gcen.z.to_value(u.kpc)]) * u.kpc
    try:
        vx, vy, vz = gcen.v_x, gcen.v_y, gcen.v_z
    except Exception:
        vx, vy, vz = gcen.velocity.d_x, gcen.velocity.d_y, gcen.velocity.d_z
    vel = np.array([vx.to_value(u.km/u.s), vy.to_value(u.km/u.s), vz.to_value(u.km/u.s)]) * (u.km/u.s)
    return PhaseSpacePosition(pos=pos, vel=vel)

def integrate_orbit(pot, w0):
    return pot.integrate_orbit(w0, dt=0.8 * u.Myr, n_steps=1200)

def sample_orbit_to_track(orb, inputs):
    icrs = orb.to_coord_frame(ICRS(), galactocentric_frame=inputs["gc_frame"])
    gd1c = icrs.transform_to(gc.GD1Koposov10())
    p1 = gd1c.phi1.wrap_at(180 * u.deg).to_value(u.deg)
    p2 = gd1c.phi2.to_value(u.deg)
    mu1 = gd1c.pm_phi1_cosphi2.to_value(u.mas/u.yr)
    mu2 = gd1c.pm_phi2.to_value(u.mas/u.yr)
    s = np.argsort(p1)
    p1, p2, mu1, mu2 = p1[s], p2[s], mu1[s], mu1[s]*0 + mu2[s]

    def interp(x, y, xq):
        return np.interp(xq, x, y, left=np.nan, right=np.nan)

    phi1 = inputs["phi1"].to_value(u.deg)
    return (
        interp(p1, p2, phi1),
        interp(p1, mu1, phi1),
        interp(p1, mu2, phi1),
    )

def make_likelihood(repo: Path | str = "."):
    inputs = load_inputs(repo)

    vh_lo, vh_hi = 80.0, 450.0
    rh_lo, rh_hi = 1.0, 80.0
    qz_lo, qz_hi = 0.4, 1.8

    def chi2_rc(theta):
        Vc = vcirc(build_potential(*theta, inputs=inputs), inputs["R_rc"])
        return float(np.nansum(((inputs["Vphi"] - Vc) / inputs["sigmaV"]).to_value(u.one) ** 2))

    def chi2_stream(theta):
        try:
            pot = build_potential(*theta, inputs=inputs)
            w0 = anchor_phase_space(theta, inputs)
            orb = integrate_orbit(pot, w0)
            p2_mod, mu1_mod, mu2_mod = sample_orbit_to_track(orb, inputs)
            x = inputs["phi1"].to_value(u.deg)
            chi = 0.0
            chi += weighted_profile_chi2(x, (inputs["phi2m"].to_value(u.deg) - p2_mod), inputs["phi2e"].to_value(u.deg))
            chi += weighted_profile_chi2(x, (inputs["pm1m"].to_value(u.mas/u.yr) - mu1_mod), inputs["pm1e"].to_value(u.mas/u.yr))
            chi += weighted_profile_chi2(x, (inputs["pm2m"].to_value(u.mas/u.yr) - mu2_mod), inputs["pm2e"].to_value(u.mas/u.yr))
            return float(chi)
        except Exception:
            return 1e9

    def log_prior(theta):
        vh, rh, qz = theta
        if not (vh_lo <= vh <= vh_hi and rh_lo <= rh <= rh_hi and qz_lo <= qz <= qz_hi):
            return -np.inf
        return 0.0

    def log_like(theta):
        return -0.5 * (chi2_rc(theta) + chi2_stream(theta))

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_like(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    return dict(
        inputs=inputs,
        chi2_rc=chi2_rc,
        chi2_stream=chi2_stream,
        log_like=log_like,
        log_prob=log_prob,
        build_potential=lambda *a, **k: build_potential(*a, **k, inputs=inputs),
        vcirc=vcirc,
    )
