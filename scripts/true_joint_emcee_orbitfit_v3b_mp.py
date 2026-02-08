#!/usr/bin/env python3
from pathlib import Path
import time
import numpy as np
import multiprocessing as mp
import importlib.util
import emcee
from emcee.moves import StretchMove, DEMove
from scipy.optimize import minimize

def load_module(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

GLOBAL = {}

def _init_worker(repo_str):
    repo = Path(repo_str).resolve()
    api = load_module("v3b_api", repo / "scripts" / "v3b_api.py")
    GLOBAL["L"] = api.make_likelihood(str(repo))

def log_prob_mp(theta):
    return float(GLOBAL["L"]["log_prob"](theta))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=64)
    ap.add_argument("--burn", type=int, default=200)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mle_maxiter", type=int, default=800)
    ap.add_argument("--ncores", type=int, default=12)
    args = ap.parse_args()

    repo = Path(".").resolve()
    outdir = repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    api = load_module("v3b_api", repo / "scripts" / "v3b_api.py")
    L = api.make_likelihood(str(repo))

    vh_lo, vh_hi = 80.0, 450.0
    rh_lo, rh_hi = 1.0, 80.0
    qz_lo, qz_hi = 0.4, 1.8

    t0 = time.time()
    x0 = np.array([198.3, 12.2, 0.92], float)
    res = minimize(lambda t: -(float(L["log_prob"](t))), x0, method="Nelder-Mead",
                   options=dict(maxiter=int(args.mle_maxiter), xatol=1e-3, fatol=1e-2))
    mle = np.array(res.x, float)
    t_mle = time.time() - t0

    rng = np.random.default_rng(int(args.seed))
    nwalk = int(args.walkers)
    ndim = 3

    sig = np.array([2.0, 0.3, 0.005], float)
    p0 = mle[None, :] + rng.normal(size=(nwalk, ndim)) * sig[None, :]
    p0[:, 0] = np.clip(p0[:, 0], vh_lo + 1e-6, vh_hi - 1e-6)
    p0[:, 1] = np.clip(p0[:, 1], rh_lo + 1e-6, rh_hi - 1e-6)
    p0[:, 2] = np.clip(p0[:, 2], qz_lo + 1e-6, qz_hi - 1e-6)

    moves = [(StretchMove(), 0.7), (DEMove(), 0.3)]

    pool = None
    if int(args.ncores) > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=int(args.ncores), initializer=_init_worker, initargs=(str(repo),))
        sampler = emcee.EnsembleSampler(nwalk, ndim, log_prob_mp, moves=moves, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalk, ndim, lambda th: float(L["log_prob"](th)), moves=moves)

    t1 = time.time()
    sampler.run_mcmc(p0, int(args.burn), progress=True)
    state = sampler.get_last_sample()
    sampler.reset()
    sampler.run_mcmc(state, int(args.steps), progress=True)
    runtime = time.time() - t1

    if pool is not None:
        pool.close()
        pool.join()

    chain = sampler.get_chain()
    lnp = sampler.get_log_prob()

    out_chain = outdir / "true_joint_emcee_orbitfit_v3b_mp_chain.npz"
    np.savez(out_chain, chain=chain, log_prob=lnp, mle=mle)

    flat = chain.reshape(-1, ndim)
    pcts = {}
    for j, name in enumerate(["vh_kms", "rh_kpc", "qz"]):
        p16, p50, p84 = np.percentile(flat[:, j], [16, 50, 84])
        pcts[name] = (float(p16), float(p50), float(p84))

    out_sum = outdir / "true_joint_emcee_orbitfit_v3b_mp_summary.txt"
    lines = []
    lines.append("true joint sampler (v3b_mp), multiprocessing via v3b_api")
    lines.append(f"walkers {nwalk} burn {int(args.burn)} steps {int(args.steps)} seed {int(args.seed)} ncores {int(args.ncores)}")
    lines.append(f"mle {mle.tolist()}")
    lines.append(f"mle_time_s {t_mle:.2f}")
    lines.append(f"mean_acceptance {float(np.mean(sampler.acceptance_fraction)):.6f}")
    for k,v in pcts.items():
        lines.append(f"{k} p16 {v[0]:.6f} p50 {v[1]:.6f} p84 {v[2]:.6f}")
    lines.append(f"runtime_s {runtime:.2f}")
    out_sum.write_text("\\n".join(lines) + "\\n")

    print(out_chain)
    print(out_sum)

if __name__ == "__main__":
    main()
