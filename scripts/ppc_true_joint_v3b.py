#!/usr/bin/env python3
from __future__ import annotations

import os, sys, argparse, time, json
from typing import Any, Dict, List, Tuple, Optional

# BLAS thread pinning so multiprocessing scales
for _k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import astropy.units as u
from astropy.coordinates import ICRS
import gala.coordinates as gc


def repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def import_v3b_api():
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts import v3b_api  # type: ignore
    return v3b_api

    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts import v3b_api  # type: ignore

def _orbit_to_gd1_arrays(orb, inputs):
    icrs = orb.to_coord_frame(ICRS(), galactocentric_frame=inputs["gc_frame"])
    gd1c = icrs.transform_to(gc.GD1Koposov10())
    p1 = gd1c.phi1.wrap_at(180 * u.deg).to_value(u.deg)
    p2 = gd1c.phi2.to_value(u.deg)
    mu1 = gd1c.pm_phi1_cosphi2.to_value(u.mas / u.yr)
    mu2 = gd1c.pm_phi2.to_value(u.mas / u.yr)
    return p1, p2, mu1, mu2

def _branchsafe_track_dp(orb, inputs, n_cand=40, lam_jump=0.02, phi_tol_deg=1.0):
    import numpy as np

    phi1_dat = np.asarray(inputs["phi1"].to_value(u.deg), float)
    phi2_obs = np.asarray(inputs["phi2m"].to_value(u.deg), float)
    pm1_obs  = np.asarray(inputs["pm1m"].to_value(u.mas/u.yr), float)
    pm2_obs  = np.asarray(inputs["pm2m"].to_value(u.mas/u.yr), float)

    phi2_err = np.asarray(inputs["phi2e"].to_value(u.deg), float)
    pm1_err  = np.asarray(inputs["pm1e"].to_value(u.mas/u.yr), float)
    pm2_err  = np.asarray(inputs["pm2e"].to_value(u.mas/u.yr), float)

    p1, p2, mu1, mu2 = _orbit_to_gd1_arrays(orb, inputs)

    ok = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(mu1) & np.isfinite(mu2)
    if not np.any(ok):
        nan = np.full_like(phi1_dat, np.nan, dtype=float)
        return nan, nan, nan

    p1 = p1[ok]; p2 = p2[ok]; mu1 = mu1[ok]; mu2 = mu2[ok]

    Ndat = phi1_dat.size
    idx_lists = []
    cost_lists = []

    for i in range(Ndat):
        d = np.abs(p1 - phi1_dat[i])
        cand = np.where(d <= float(phi_tol_deg))[0]
        if cand.size == 0:
            cand = np.argsort(d)[:max(5, int(n_cand))]
        else:
            cand = cand[np.argsort(d[cand])[:int(n_cand)]]

        idx_lists.append(cand)

        dz2 = ((phi2_obs[i] - p2[cand]) / phi2_err[i])**2
        dz1 = ((pm1_obs[i]  - mu1[cand]) / pm1_err[i])**2
        dz0 = ((pm2_obs[i]  - mu2[cand]) / pm2_err[i])**2
        cost_lists.append(dz2 + dz1 + dz0)

    dp = []
    back = []

    dp0 = cost_lists[0].astype(float)
    dp.append(dp0)
    back.append(np.full(dp0.shape, -1, dtype=int))

    for i in range(1, Ndat):
        prev_idx = idx_lists[i-1]
        cur_idx  = idx_lists[i]
        prev_dp  = dp[i-1]
        cur_cost = cost_lists[i].astype(float)

        Mprev = prev_idx.size
        Mcur  = cur_idx.size

        best_prev = np.empty(Mcur, dtype=int)
        best_val  = np.empty(Mcur, dtype=float)

        for j in range(Mcur):
            jump = (cur_idx[j] - prev_idx).astype(float)
            pen = float(lam_jump) * (jump*jump)
            vals = prev_dp + pen
            k = int(np.argmin(vals))
            best_prev[j] = k
            best_val[j]  = vals[k]

        dp_i = cur_cost + best_val
        dp.append(dp_i)
        back.append(best_prev)

    last = int(np.argmin(dp[-1]))
    path = [last]
    for i in range(Ndat-1, 0, -1):
        last = int(back[i][last])
        path.append(last)
    path = path[::-1]

    choose = np.array([idx_lists[i][path[i]] for i in range(Ndat)], dtype=int)

    return p2[choose], mu1[choose], mu2[choose]
    return v3b_api

def _track_timewindow_best(orb, inputs, nwin=60, win_half=260):
    """
    Select a time-contiguous orbit segment whose phi1 range covers the data,
    and that minimizes total chi^2 over (phi2, pm1, pm2). Then interpolate.
    """
    import numpy as np

    phi1_dat = np.asarray(inputs["phi1"].to_value(u.deg), float)
    phi2_obs = np.asarray(inputs["phi2m"].to_value(u.deg), float)
    pm1_obs  = np.asarray(inputs["pm1m"].to_value(u.mas/u.yr), float)
    pm2_obs  = np.asarray(inputs["pm2m"].to_value(u.mas/u.yr), float)

    phi2_err = np.asarray(inputs["phi2e"].to_value(u.deg), float)
    pm1_err  = np.asarray(inputs["pm1e"].to_value(u.mas/u.yr), float)
    pm2_err  = np.asarray(inputs["pm2e"].to_value(u.mas/u.yr), float)

    # IMPORTANT: keep time order (do NOT sort yet)
    p1, p2, mu1, mu2 = _orbit_to_gd1_arrays(orb, inputs)

    ok = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(mu1) & np.isfinite(mu2)
    if not np.any(ok):
        nan = np.full_like(phi1_dat, np.nan, dtype=float)
        return nan, nan, nan

    p1 = p1[ok]; p2 = p2[ok]; mu1 = mu1[ok]; mu2 = mu2[ok]
    N = p1.size

    phi1_min = float(np.nanmin(phi1_dat))
    phi1_max = float(np.nanmax(phi1_dat))

    # candidate centers along time
    centers = np.linspace(0, N-1, int(nwin)).astype(int)

    best_cost = np.inf
    best_mod = None

    for c in centers:
        lo = max(0, c - int(win_half))
        hi = min(N, c + int(win_half) + 1)
        p1w = p1[lo:hi]; p2w = p2[lo:hi]; m1w = mu1[lo:hi]; m2w = mu2[lo:hi]

        if p1w.size < 10:
            continue

        # does this window cover the data phi1 range?
        if (np.nanmin(p1w) > phi1_min) or (np.nanmax(p1w) < phi1_max):
            continue

        # Sort by phi1 for interpolation
        s = np.argsort(p1w)
        x = p1w[s]
        y2 = p2w[s]
        y1 = m1w[s]
        y0 = m2w[s]

        # Compress duplicate phi1 values (np.interp hates repeats)
        xu, inv = np.unique(x, return_inverse=True)
        if xu.size < 10:
            continue

        def mean_by_inv(vals):
            num = np.bincount(inv, weights=vals, minlength=xu.size)
            den = np.bincount(inv, minlength=xu.size)
            return num / np.where(den > 0, den, 1)

        y2u = mean_by_inv(y2)
        y1u = mean_by_inv(y1)
        y0u = mean_by_inv(y0)

        # Interpolate onto data phi1 grid
        p2_mod = np.interp(phi1_dat, xu, y2u, left=np.nan, right=np.nan)
        m1_mod = np.interp(phi1_dat, xu, y1u, left=np.nan, right=np.nan)
        m2_mod = np.interp(phi1_dat, xu, y0u, left=np.nan, right=np.nan)

        m = np.isfinite(p2_mod) & np.isfinite(m1_mod) & np.isfinite(m2_mod)
        if np.sum(m) < 0.8 * phi1_dat.size:
            continue

        dz2 = ((phi2_obs[m] - p2_mod[m]) / phi2_err[m])**2
        dz1 = ((pm1_obs[m]  - m1_mod[m]) / pm1_err[m])**2
        dz0 = ((pm2_obs[m]  - m2_mod[m]) / pm2_err[m])**2
        cost = float(np.nansum(dz2 + dz1 + dz0))

        if cost < best_cost:
            best_cost = cost
            best_mod = (p2_mod, m1_mod, m2_mod)

    if best_mod is None:
        # fallback: keep your DP (if it exists), else NaNs
        try:
            return _branchsafe_track_dp(orb, inputs, n_cand=40, lam_jump=0.02, phi_tol_deg=1.0)
        except Exception:
            nan = np.full_like(phi1_dat, np.nan, dtype=float)
            return nan, nan, nan

    return best_mod

def safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def squeeze1d(a: Any) -> np.ndarray:
    return np.asarray(a).reshape(-1)

def qtiles(x: np.ndarray, qs: Tuple[float, ...]) -> List[np.ndarray]:
    return [np.quantile(x, q, axis=0) for q in qs]

def format_theta(theta: np.ndarray) -> str:
    return f"vh={theta[0]:.3f} km/s, rh={theta[1]:.3f} kpc, qz={theta[2]:.5f}"

def stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, float)
    return {"min": float(np.min(a)), "median": float(np.median(a)), "max": float(np.max(a))}

def load_chain_npz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    chain = np.asarray(z["chain"])
    log_prob = np.asarray(z["log_prob"]) if "log_prob" in z else None
    mle = np.asarray(z["mle"]) if "mle" in z else None
    return chain, log_prob, mle

def flatten_chain(chain: np.ndarray, thin: int) -> np.ndarray:
    if chain.ndim != 3:
        raise ValueError(f"Expected chain ndim=3, got shape={chain.shape}")
    th = chain[::thin, :, :]
    return th.reshape(-1, th.shape[-1])

def draw_thetas_from_chain(path: str, ndraw: int, thin: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    chain, log_prob, mle = load_chain_npz(path)
    flat = flatten_chain(chain, thin)
    nd = min(ndraw, flat.shape[0])
    idx = rng.choice(flat.shape[0], size=nd, replace=False)
    thetas = flat[idx, :]
    meta = {
        "path": path,
        "chain_shape": list(chain.shape),
        "flat_shape": list(flat.shape),
        "thin": thin,
        "ndraw": nd,
        "mle_in_file": (mle is not None),
    }
    return thetas, meta


def worker_eval(theta: np.ndarray) -> Dict[str, Any]:
    v3b_api = import_v3b_api()
    L = v3b_api.make_likelihood()
    inputs = L["inputs"]

    # Observed RC arrays (1D floats)
    V_rc = squeeze1d(inputs["Vphi"].to_value(u.km/u.s))
    S_rc = squeeze1d(inputs["sigmaV"].to_value(u.km/u.s))

    out: Dict[str, Any] = {"theta": theta}

    # Potential + RC prediction
    pot = v3b_api.build_potential(*theta, inputs=inputs)
    VcQ = v3b_api.vcirc(pot, inputs["R_rc"])
    Vc = squeeze1d(VcQ.to_value(u.km/u.s))
    out["rc_vpred_kms"] = Vc

    # RC chi2 (shape safe)
    out["chi2_rc"] = float(np.nansum(((V_rc - Vc) / S_rc) ** 2))

    # Stream prediction (matches chi2_stream internals)
    w0 = v3b_api.anchor_phase_space(theta, inputs)
    orb = v3b_api.integrate_orbit(pot, w0)
    p2_mod, mu1_mod, mu2_mod = _track_timewindow_best(orb, inputs, nwin=60, win_half=260)
    out["p2_mod_deg"] = squeeze1d(np.asarray(p2_mod, float))
    out["mu1_mod_masyr"] = squeeze1d(np.asarray(mu1_mod, float))
    out["mu2_mod_masyr"] = squeeze1d(np.asarray(mu2_mod, float))

    out["chi2_stream"] = float(L["chi2_stream"](theta))
    return out


def evaluate_chain(thetas: np.ndarray, ncores: int) -> List[Dict[str, Any]]:
    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=ncores, mp_context=ctx) as ex:
        futs = [ex.submit(worker_eval, np.asarray(th, float)) for th in thetas]
        for i, fut in enumerate(as_completed(futs), start=1):
            results.append(fut.result())
            if i % max(1, len(thetas)//10) == 0:
                print(f"[INFO] Completed {i}/{len(thetas)}")
    return results


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--ndraw", type=int, default=400)
    ap.add_argument("--thin", type=int, default=50)
    ap.add_argument("--ncores", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--mode_low", default="results/true_joint_emcee_orbitfit_v3b_mode_lowqz_chain.npz")
    ap.add_argument("--mode_high", default="results/true_joint_emcee_orbitfit_v3b_mode_highqz_chain.npz")
    ap.add_argument("--ndraw_mode", type=int, default=250)
    ap.add_argument("--thin_mode", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    safe_makedirs(args.outdir)
    rng = np.random.default_rng(args.seed)

    v3b_api = import_v3b_api()
    L = v3b_api.make_likelihood()
    inputs = L["inputs"]

    # Observed RC
    R_rc = squeeze1d(inputs["R_rc"].to_value(u.kpc))
    V_rc = squeeze1d(inputs["Vphi"].to_value(u.km/u.s))
    S_rc = squeeze1d(inputs["sigmaV"].to_value(u.km/u.s))

    # Observed stream
    x_phi1 = squeeze1d(inputs["phi1"].to_value(u.deg))
    phi2_obs = squeeze1d(inputs["phi2m"].to_value(u.deg))
    phi2_err = squeeze1d(inputs["phi2e"].to_value(u.deg))
    pm1_obs = squeeze1d(inputs["pm1m"].to_value(u.mas/u.yr))
    pm1_err = squeeze1d(inputs["pm1e"].to_value(u.mas/u.yr))
    pm2_obs = squeeze1d(inputs["pm2m"].to_value(u.mas/u.yr))
    pm2_err = squeeze1d(inputs["pm2e"].to_value(u.mas/u.yr))

    # ---- MAIN chain
    thetas_main, meta_main = draw_thetas_from_chain(args.chain, args.ndraw, args.thin, args.seed)
    print(f"[INFO] Chain: {args.chain}")
    print(f"[INFO] chain_shape={meta_main['chain_shape']} thin={args.thin} flat={meta_main['flat_shape']}")
    print(f"[INFO] Using ndraw={meta_main['ndraw']}, ncores={args.ncores}")

    res_main = evaluate_chain(thetas_main, args.ncores)

    rc_vpred = np.stack([r["rc_vpred_kms"] for r in res_main], axis=0)
    p2_mod = np.stack([r["p2_mod_deg"] for r in res_main], axis=0)
    mu1_mod = np.stack([r["mu1_mod_masyr"] for r in res_main], axis=0)
    mu2_mod = np.stack([r["mu2_mod_masyr"] for r in res_main], axis=0)

    chi2_rc = np.array([r["chi2_rc"] for r in res_main], float)
    chi2_stream = np.array([r["chi2_stream"] for r in res_main], float)
    chi2_tot = chi2_rc + chi2_stream

    best = int(np.nanargmin(chi2_tot))
    theta_best = res_main[best]["theta"]

    # ---- RC plots
    q05, q16, q50, q84, q95 = qtiles(rc_vpred, (0.05,0.16,0.50,0.84,0.95))
    z_rc = (V_rc[None,:] - rc_vpred) / S_rc[None,:]
    rz05, rz16, rz50, rz84, rz95 = qtiles(z_rc, (0.05,0.16,0.50,0.84,0.95))
    best_rc_z = (V_rc - rc_vpred[best]) / S_rc

    # RC discrepancy p-value
    Tobs_rc = chi2_rc.copy()
    yrep_rc = rc_vpred + rng.normal(0.0, S_rc[None,:], size=rc_vpred.shape)
    Trep_rc = np.sum(((yrep_rc - rc_vpred) / S_rc[None,:])**2, axis=1)
    bayes_p_rc = float(np.mean(Trep_rc > Tobs_rc))

    plt.figure(figsize=(8.7,5.6))
    plt.errorbar(R_rc, V_rc, yerr=S_rc, fmt="o", ms=3.5, lw=1, label="Observed RC")
    plt.fill_between(R_rc, q05, q95, alpha=0.20, label="PP band 5–95% (model)")
    plt.fill_between(R_rc, q16, q84, alpha=0.35, label="PP band 16–84% (model)")
    for j in rng.choice(len(res_main), size=min(25, len(res_main)), replace=False):
        plt.plot(R_rc, rc_vpred[j], lw=0.8, alpha=0.20)
    plt.plot(R_rc, rc_vpred[best], lw=2.0, label="Best (min χ²)")
    plt.xlabel("R [kpc]")
    plt.ylabel("V_circ [km/s]")
    plt.title("Rotation Curve PPC (v3b true joint)")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ppc_rc_overlay.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8.7,5.2))
    plt.axhline(0.0, lw=1)
    plt.fill_between(R_rc, rz05, rz95, alpha=0.20, label="z band 5–95%")
    plt.fill_between(R_rc, rz16, rz84, alpha=0.35, label="z band 16–84%")
    plt.plot(R_rc, rz50, lw=1.5, label="z median")
    plt.plot(R_rc, best_rc_z, lw=1.5, label="Best z")
    plt.xlabel("R [kpc]")
    plt.ylabel("(V_obs - V_model)/σ")
    plt.title("RC residual PPC (standardized)")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ppc_rc_residuals.png"), dpi=220)
    plt.close()

    # ---- Stream plots
    p2_05, p2_16, p2_50, p2_84, p2_95 = qtiles(p2_mod, (0.05,0.16,0.50,0.84,0.95))
    m1_05, m1_16, m1_50, m1_84, m1_95 = qtiles(mu1_mod, (0.05,0.16,0.50,0.84,0.95))
    m2_05, m2_16, m2_50, m2_84, m2_95 = qtiles(mu2_mod, (0.05,0.16,0.50,0.84,0.95))

    z_p2 = (phi2_obs[None,:] - p2_mod) / phi2_err[None,:]
    z_m1 = (pm1_obs[None,:] - mu1_mod) / pm1_err[None,:]
    z_m2 = (pm2_obs[None,:] - mu2_mod) / pm2_err[None,:]
    zp2_05, zp2_16, zp2_50, zp2_84, zp2_95 = qtiles(z_p2, (0.05,0.16,0.50,0.84,0.95))
    zm1_05, zm1_16, zm1_50, zm1_84, zm1_95 = qtiles(z_m1, (0.05,0.16,0.50,0.84,0.95))
    zm2_05, zm2_16, zm2_50, zm2_84, zm2_95 = qtiles(z_m2, (0.05,0.16,0.50,0.84,0.95))
    best_zp2 = (phi2_obs - p2_mod[best]) / phi2_err
    best_zm1 = (pm1_obs - mu1_mod[best]) / pm1_err
    best_zm2 = (pm2_obs - mu2_mod[best]) / pm2_err

    # Stream discrepancy p-value (uses exact helper)
    Tobs_stream = chi2_stream.copy()
    Trep_stream = np.zeros(len(res_main), float)
    for i in range(len(res_main)):
        p2_rep = p2_mod[i] + rng.normal(0.0, phi2_err, size=phi2_err.shape)
        m1_rep = mu1_mod[i] + rng.normal(0.0, pm1_err, size=pm1_err.shape)
        m2_rep = mu2_mod[i] + rng.normal(0.0, pm2_err, size=pm2_err.shape)
        chi = 0.0
        chi += float(v3b_api.weighted_profile_chi2(x_phi1, (p2_rep - p2_mod[i]), phi2_err))
        chi += float(v3b_api.weighted_profile_chi2(x_phi1, (m1_rep - mu1_mod[i]), pm1_err))
        chi += float(v3b_api.weighted_profile_chi2(x_phi1, (m2_rep - mu2_mod[i]), pm2_err))
        Trep_stream[i] = chi
    bayes_p_stream = float(np.mean(Trep_stream > Tobs_stream))

    fig, axs = plt.subplots(3, 1, figsize=(9.2, 10.0), sharex=True)
    axs[0].errorbar(x_phi1, phi2_obs, yerr=phi2_err, fmt="o", ms=3, lw=1, label="Observed")
    axs[0].fill_between(x_phi1, p2_05, p2_95, alpha=0.20, label="PP 5–95% (model)")
    axs[0].fill_between(x_phi1, p2_16, p2_84, alpha=0.35, label="PP 16–84% (model)")
    for j in rng.choice(len(res_main), size=min(25, len(res_main)), replace=False):
        axs[0].plot(x_phi1, p2_mod[j], lw=0.8, alpha=0.20)
    axs[0].plot(x_phi1, p2_mod[best], lw=2.0, label="Best (min χ²)")
    axs[0].set_ylabel("phi2 [deg]")
    axs[0].set_title("GD-1 Stream PPC (v3b true joint)")
    axs[0].legend(frameon=False, fontsize=9)

    axs[1].errorbar(x_phi1, pm1_obs, yerr=pm1_err, fmt="o", ms=3, lw=1)
    axs[1].fill_between(x_phi1, m1_05, m1_95, alpha=0.20)
    axs[1].fill_between(x_phi1, m1_16, m1_84, alpha=0.35)
    for j in rng.choice(len(res_main), size=min(25, len(res_main)), replace=False):
        axs[1].plot(x_phi1, mu1_mod[j], lw=0.8, alpha=0.20)
    axs[1].plot(x_phi1, mu1_mod[best], lw=2.0)
    axs[1].set_ylabel("pm1 [mas/yr]")

    axs[2].errorbar(x_phi1, pm2_obs, yerr=pm2_err, fmt="o", ms=3, lw=1)
    axs[2].fill_between(x_phi1, m2_05, m2_95, alpha=0.20)
    axs[2].fill_between(x_phi1, m2_16, m2_84, alpha=0.35)
    for j in rng.choice(len(res_main), size=min(25, len(res_main)), replace=False):
        axs[2].plot(x_phi1, mu2_mod[j], lw=0.8, alpha=0.20)
    axs[2].plot(x_phi1, mu2_mod[best], lw=2.0)
    axs[2].set_ylabel("pm2 [mas/yr]")
    axs[2].set_xlabel("phi1 [deg]")

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ppc_stream_overlay.png"), dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
    axs[0].axhline(0.0, lw=1)
    axs[0].fill_between(x_phi1, zp2_05, zp2_95, alpha=0.20, label="z 5–95%")
    axs[0].fill_between(x_phi1, zp2_16, zp2_84, alpha=0.35, label="z 16–84%")
    axs[0].plot(x_phi1, zp2_50, lw=1.5, label="z median")
    axs[0].plot(x_phi1, best_zp2, lw=1.5, label="Best z")
    axs[0].set_ylabel("(phi2_obs-phi2_mod)/σ")
    axs[0].legend(frameon=False, fontsize=9)

    axs[1].axhline(0.0, lw=1)
    axs[1].fill_between(x_phi1, zm1_05, zm1_95, alpha=0.20)
    axs[1].fill_between(x_phi1, zm1_16, zm1_84, alpha=0.35)
    axs[1].plot(x_phi1, zm1_50, lw=1.5)
    axs[1].plot(x_phi1, best_zm1, lw=1.5)
    axs[1].set_ylabel("(pm1_obs-pm1_mod)/σ")

    axs[2].axhline(0.0, lw=1)
    axs[2].fill_between(x_phi1, zm2_05, zm2_95, alpha=0.20)
    axs[2].fill_between(x_phi1, zm2_16, zm2_84, alpha=0.35)
    axs[2].plot(x_phi1, zm2_50, lw=1.5)
    axs[2].plot(x_phi1, best_zm2, lw=1.5)
    axs[2].set_ylabel("(pm2_obs-pm2_mod)/σ")
    axs[2].set_xlabel("phi1 [deg]")

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ppc_stream_residuals.png"), dpi=220)
    plt.close(fig)

    # ---- Mode chain chi2 overlays (3 panels)
    modes = []
    for label, path in [("mode_low", args.mode_low), ("mode_high", args.mode_high)]:
        if path and os.path.exists(path):
            thetas_m, meta_m = draw_thetas_from_chain(path, args.ndraw_mode, args.thin_mode, args.seed + (1 if label=="mode_low" else 2))
            print(f"[INFO] Evaluating {label}: {path} ndraw={meta_m['ndraw']}")
            res_m = evaluate_chain(thetas_m, args.ncores)
            c_rc = np.array([r["chi2_rc"] for r in res_m], float)
            c_st = np.array([r["chi2_stream"] for r in res_m], float)
            modes.append((label, path, c_rc, c_st, c_rc+c_st))
        else:
            print(f"[INFO] {label} not found, skipping: {path}")

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    bins = 40

    def plot_hist(ax, arr, label, alpha):
        ax.hist(arr, bins=bins, alpha=alpha, label=label)

    # RC
    plot_hist(axs[0], chi2_rc, "main", 0.45)
    for label, path, c_rc, c_st, c_tot in modes:
        plot_hist(axs[0], c_rc, label, 0.35)
    axs[0].axvline(float(np.median(chi2_rc)), lw=2, label="main median")
    axs[0].axvline(float(np.min(chi2_rc)), lw=2, label="main best")
    axs[0].set_title("chi2_rc")
    axs[0].set_xlabel("chi2")

    # Stream
    plot_hist(axs[1], chi2_stream, "main", 0.45)
    for label, path, c_rc, c_st, c_tot in modes:
        plot_hist(axs[1], c_st, label, 0.35)
    axs[1].axvline(float(np.median(chi2_stream)), lw=2)
    axs[1].axvline(float(np.min(chi2_stream)), lw=2)
    axs[1].set_title("chi2_stream")
    axs[1].set_xlabel("chi2")

    # Total
    plot_hist(axs[2], chi2_tot, "main", 0.45)
    for label, path, c_rc, c_st, c_tot in modes:
        plot_hist(axs[2], c_tot, label, 0.35)
    axs[2].axvline(float(np.median(chi2_tot)), lw=2)
    axs[2].axvline(float(np.min(chi2_tot)), lw=2)
    axs[2].set_title("chi2_total")
    axs[2].set_xlabel("chi2")

    for ax in axs:
        ax.set_ylabel("count")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=4)
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(os.path.join(args.outdir, "ppc_chi2_distributions.png"), dpi=220)
    plt.close(fig)

    # ---- Extra scalar discrepancy stats (science-fair add)
    rc_rmsz = float(np.sqrt(np.mean(best_rc_z**2)))
    rc_maxabsz = float(np.max(np.abs(best_rc_z)))
    stream_maxabs = {
        "phi2": float(np.max(np.abs(best_zp2))),
        "pm1": float(np.max(np.abs(best_zm1))),
        "pm2": float(np.max(np.abs(best_zm2))),
    }
    stream_rms = {
        "phi2": float(np.sqrt(np.mean(best_zp2**2))),
        "pm1": float(np.sqrt(np.mean(best_zm1**2))),
        "pm2": float(np.sqrt(np.mean(best_zm2**2))),
    }

    summary = {
        "chain_main": meta_main,
        "ndraw_main": len(res_main),
        "ncores": args.ncores,
        "seed": args.seed,
        "best_theta": theta_best.tolist(),
        "best_theta_str": format_theta(theta_best),
        "median_theta_main": np.median(thetas_main, axis=0).tolist(),
        "bayes_p_rc": bayes_p_rc,
        "bayes_p_stream": bayes_p_stream,
        "chi2_rc_stats": stats(chi2_rc),
        "chi2_stream_stats": stats(chi2_stream),
        "chi2_total_stats": stats(chi2_tot),
        "Tobs_rc_stats": stats(Tobs_rc),
        "Trep_rc_stats": stats(Trep_rc),
        "Tobs_stream_stats": stats(Tobs_stream),
        "Trep_stream_stats": stats(Trep_stream),
        "rc_best_rms_z": rc_rmsz,
        "rc_best_maxabs_z": rc_maxabsz,
        "stream_best_maxabs_z": stream_maxabs,
        "stream_best_rms_z": stream_rms,
        "modes_included": [{"label": m[0], "path": m[1]} for m in modes],
        "runtime_sec": float(time.time() - t0),
    }

    with open(os.path.join(args.outdir, "ppc_summary.txt"), "w") as f:
        f.write("PPC summary (v3b true joint)\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    print("[INFO] Wrote results/ppc_*.png and results/ppc_summary.txt")


if __name__ == "__main__":
    main()
