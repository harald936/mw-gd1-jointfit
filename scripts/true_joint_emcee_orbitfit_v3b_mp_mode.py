#!/usr/bin/env python3
from pathlib import Path
import time
import numpy as np
import multiprocessing as mp
import importlib.util
import emcee
from emcee.moves import StretchMove, DEMove
import matplotlib.pyplot as plt

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
    ap.add_argument("--mode", type=str, default="highqz", choices=["highqz","lowqz"])
    ap.add_argument("--walkers", type=int, default=64)
    ap.add_argument("--burn", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ncores", type=int, default=12)
    args = ap.parse_args()

    repo = Path(".").resolve()
    outdir = repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    api = load_api(repo)
    L0 = api.make_likelihood(str(repo))

    if args.mode == "highqz":
        center = np.array([199.76, 6.085, 0.9277], float)
        sig    = np.array([2.0, 0.25, 0.005], float)
        tag = "mode_highqz"
    else:
        center = np.array([204.87217219300274, 6.65760764382037, 0.6033044214635424], float)
        sig    = np.array([2.0, 0.30, 0.01], float)
        tag = "mode_lowqz"

    vh_lo, vh_hi = 80.0, 450.0
    rh_lo, rh_hi = 1.0, 80.0
    qz_lo, qz_hi = 0.4, 1.8

    rng = np.random.default_rng(int(args.seed))
    nwalk = int(args.walkers)
    p0 = center[None,:] + rng.normal(size=(nwalk,3))*sig[None,:]
    p0[:,0] = np.clip(p0[:,0], vh_lo+1e-6, vh_hi-1e-6)
    p0[:,1] = np.clip(p0[:,1], rh_lo+1e-6, rh_hi-1e-6)
    p0[:,2] = np.clip(p0[:,2], qz_lo+1e-6, qz_hi-1e-6)

    moves = [(StretchMove(), 0.7), (DEMove(), 0.3)]

    t0 = time.time()
    pool = None
    if int(args.ncores) > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=int(args.ncores), initializer=_init_worker, initargs=(str(repo),))
        sampler = emcee.EnsembleSampler(nwalk, 3, log_prob_mp, moves=moves, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalk, 3, lambda th: float(L0["log_prob"](th)), moves=moves)

    sampler.run_mcmc(p0, int(args.burn), progress=True)
    state = sampler.get_last_sample()
    sampler.reset()
    sampler.run_mcmc(state, int(args.steps), progress=True)

    if pool is not None:
        pool.close(); pool.join()

    chain = sampler.get_chain()
    lp = sampler.get_log_prob()
    flat = chain.reshape(-1,3)
    flat_lp = lp.reshape(-1)

    i = int(np.nanargmax(flat_lp))
    best = flat[i]
    med = np.nanmedian(flat, axis=0)
    p16 = np.nanpercentile(flat, 16, axis=0)
    p84 = np.nanpercentile(flat, 84, axis=0)

    c1_best = float(L0["chi2_rc"](best))
    c2_best = float(L0["chi2_stream"](best))
    c1_med  = float(L0["chi2_rc"](med))
    c2_med  = float(L0["chi2_stream"](med))

    out_npz = outdir / f"true_joint_emcee_orbitfit_v3b_{tag}_chain.npz"
    np.savez(out_npz, chain=chain, log_prob=lp)

    out_sum = outdir / f"true_joint_emcee_orbitfit_v3b_{tag}_summary.txt"
    lines = []
    lines.append(tag)
    lines.append(f"walkers {nwalk} burn {int(args.burn)} steps {int(args.steps)} seed {int(args.seed)} ncores {int(args.ncores)}")
    lines.append(f"mean_acceptance {float(np.mean(sampler.acceptance_fraction)):.6f}")
    lines.append(f"best_theta {best.tolist()} best_logp {float(flat_lp[i])}")
    lines.append(f"best_chi2_rc {c1_best} best_chi2_stream {c2_best} best_total {c1_best+c2_best}")
    lines.append(f"median_theta {med.tolist()}")
    lines.append(f"median_chi2_rc {c1_med} median_chi2_stream {c2_med} median_total {c1_med+c2_med}")
    lines.append(f"vh p16/50/84 {float(p16[0])} {float(med[0])} {float(p84[0])}")
    lines.append(f"rh p16/50/84 {float(p16[1])} {float(med[1])} {float(p84[1])}")
    lines.append(f"qz p16/50/84 {float(p16[2])} {float(med[2])} {float(p84[2])}")
    lines.append(f"runtime_s {time.time()-t0:.2f}")
    out_sum.write_text("\n".join(lines) + "\n")

    print(out_npz)
    print(out_sum)

    names = ["vh_kms","rh_kpc","qz"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for j in range(3):
        axes[j].plot(chain[:,:,j], alpha=0.2, linewidth=0.5)
        axes[j].set_ylabel(names[j])
    axes[-1].plot(lp, alpha=0.2, linewidth=0.5)
    axes[-1].set_ylabel("log_prob")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    trace_path = outdir / f"true_joint_emcee_orbitfit_v3b_{tag}_traces.png"
    fig.savefig(trace_path, dpi=200)
    plt.close(fig)
    print(trace_path)

    try:
        import corner
        fig = corner.corner(flat, labels=names, quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt=".6f")
        corner_path = outdir / f"true_joint_emcee_orbitfit_v3b_{tag}_corner.png"
        fig.savefig(corner_path, dpi=200)
        plt.close(fig)
        print(corner_path)
    except Exception as e:
        print("corner_not_made", repr(e))

if __name__ == "__main__":
    main()
