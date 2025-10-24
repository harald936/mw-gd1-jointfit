#!/usr/bin/env python3
"""
Compare cuspiness:
- Einasto: gamma_eff(R) = 2 * (R/rs)**alpha  (varies with radius)
- Your gNFW fit: constant inner slope gamma_gNFW (from Step 2)
Saves a plot to results/einasto_vs_gnfw.png and prints a small table.
"""
from pathlib import Path
import argparse, numpy as np, matplotlib.pyplot as plt

def gamma_eff_einasto(R, rs, alpha):
    R = np.asarray(R, float)
    return 2.0 * (R/rs)**alpha

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True, help="Einasto shape parameter from the paper")
    p.add_argument("--rs",    type=float, required=True, help="Einasto scale radius [kpc] from the paper")
    p.add_argument("--gamma_gnfw", type=float, default=1.34, help="Your gNFW inner slope (from Step 2)")
    p.add_argument("--rmin",  type=float, default=5.0, help="min radius [kpc]")
    p.add_argument("--rmax",  type=float, default=20.0, help="max radius [kpc]")
    p.add_argument("--out",   type=str,   default="results/einasto_vs_gnfw.png", help="output figure path")
    args = p.parse_args()

    R = np.linspace(args.rmin, args.rmax, 300)
    ge = gamma_eff_einasto(R, args.rs, args.alpha)

    # Print a quick table at a few radii
    sample_R = np.array([5, 8, 10, 12, 15, 18], float)
    sample_ge = gamma_eff_einasto(sample_R, args.rs, args.alpha)
    print("Einasto cuspiness gamma_eff(R) with alpha=%.3f, rs=%.2f kpc:" % (args.alpha, args.rs))
    for r, g in zip(sample_R, sample_ge):
        print("  R=%4.1f kpc  ->  gamma_eff=%.3f" % (r, g))
    print("\nYour gNFW gamma (constant): %.3f\n" % args.gamma_gnfw)

    # Plot
    Path("results").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4,3.8))
    plt.plot(R, ge, label=r"Einasto  $\gamma_{\rm eff}(R)=2\,(R/r_s)^\alpha$")
    plt.axhline(args.gamma_gnfw, ls="--", label=fr"Your gNFW  $\gamma={args.gamma_gnfw:.2f}$")
    plt.scatter(sample_R, sample_ge, s=18, zorder=3)
    plt.xlabel("R [kpc]"); plt.ylabel(r"Cuspiness  $\gamma_{\rm eff}$")
    plt.title("Cuspiness: Einasto (paper) vs gNFW (yours)")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print("Saved figure:", args.out)

if __name__ == "__main__":
    main()
