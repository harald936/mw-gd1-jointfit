#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import time
import multiprocessing as mp
import importlib.util
import emcee
from emcee.moves import StretchMove, DEMove

def load_api(repo):
    spec = importlib.util.spec_from_file_location("v3b_api", str(repo/"scripts"/"v3b_api.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

GLOBAL = {}

def _init_worker(repo_str):
    repo = Path(repo_str).resolve()
    api = load_api(repo)
    GLOBAL["L"] = api.make_likelihood(str(repo))

def log_prob_mp(theta):
    return float(GLOBAL["L"]["log_prob"](theta))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=64)
    ap.add_argument("--burn", type=int, default=80)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ncores", type=int, default=12)
    args = ap.parse_args()

    repo = Path(".").resolve()
    outdir = repo/"results"
    outdir.mkdir(exist_ok=True)

    api = load_api(repo)
    L0 = api.make_likelihood(str(repo))

    center = np.array([199.7568521196927, 6.0852638620993185, 0.9276933659403341], float)
    sig = np.array([40.0, 8.0, 0.25], float)

    vh_lo, vh_hi = 80.0, 450.0
    rh_lo, rh_hi = 1.0, 80.0
    qz_lo, qz_hi = 0.4, 1.8

    rng = np.random.default_rng(int(args.seed))
    p0 = center[None,:] + rng.normal(size=(int(args.walkers),3))*sig[None,:]
    p0[:,0] = np.clip(p0[:,0], vh_lo+1e-6, vh_hi-1e-6)
    p0[:,1] = np.clip(p0[:,1], rh_lo+1e-6, rh_hi-1e-6)
    p0[:,2] = np.clip(p0[:,2], qz_lo+1e-6, qz_hi-1e-6)

    moves = [(StretchMove(), 0.7), (DEMove(), 0.3)]

    t0 = time.time()
    pool = None
    if int(args.ncores) > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=int(args.ncores), initializer=_init_worker, initargs=(str(repo),))
        sampler = emcee.EnsembleSampler(int(args.walkers), 3, log_prob_mp, moves=moves, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(int(args.walkers), 3, lambda th: float(L0["log_prob"](th)), moves=moves)

    sampler.run_mcmc(p0, int(args.burn), progress=True)
    state = sampler.get_last_sample()
    sampler.reset()
    sampler.run_mcmc(state, int(args.steps), progress=True)

    if pool is not None:
        pool.close(); pool.join()

    chain = sampler.get_chain()
    lp = sampler.get_log_prob()

    out_npz = outdir/"true_joint_emcee_orbitfit_v3b_mp_wideinit_chain.npz"
    np.savez(out_npz, chain=chain, log_prob=lp)

    flat = chain.reshape(-1,3)
    flat_lp = lp.reshape(-1)
    med = np.nanmedian(flat, axis=0)
    p16 = np.nanpercentile(flat, 16, axis=0)
    p84 = np.nanpercentile(flat, 84, axis=0)
    i = int(np.nanargmax(flat_lp))
    best = flat[i]

    print("WROTE", out_npz)
    print("posterior p16/50/84 vh", float(p16[0]), float(med[0]), float(p84[0]))
    print("posterior p16/50/84 rh", float(p16[1]), float(med[1]), float(p84[1]))
    print("posterior p16/50/84 qz", float(p16[2]), float(med[2]), float(p84[2]))
    print("best theta", best.tolist(), "best_logp", float(flat_lp[i]))

    c1 = float(L0["chi2_rc"](med))
    c2 = float(L0["chi2_stream"](med))
    print("chi2 at median", "chi2_rc", c1, "chi2_stream", c2, "total", c1+c2, "frac_rc", c1/(c1+c2))
    print("runtime_s", time.time()-t0)

if __name__ == "__main__":
    main()
