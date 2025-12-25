from pathlib import Path
import hashlib, re
import numpy as np

repo = Path(".").resolve()

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def summarize_npz(p: Path) -> None:
    print("\n===", p, "===")
    if not p.exists():
        print("MISSING")
        return
    print("size_bytes", p.stat().st_size)
    print("sha256", sha256(p))

    z = np.load(p)
    keys = sorted(list(z.keys()))
    print("keys", keys)

    if "gamma" not in z or "posterior" not in z:
        print("ERROR: expected keys gamma + posterior")
        return

    g = np.asarray(z["gamma"], float)
    P = np.asarray(z["posterior"], float)
    print("gamma_grid_min", float(g.min()), "gamma_grid_max", float(g.max()), "Ng", g.size)
    print("posterior_shape", tuple(P.shape), "posterior_sum", float(P.sum()))

    pg = P.sum(axis=1)
    s = float(pg.sum())
    if s <= 0:
        print("ERROR: marginal sum <= 0")
        return
    pg = pg / s

    # gridpoint masses at exact values if present
    for val in [0.5, 1.0, 1.5]:
        idx = np.where(np.isclose(g, val))[0]
        if len(idx) == 0:
            print(f"mass_at_gamma_{val}", "NONE")
        else:
            print(f"mass_at_gamma_{val}", float(pg[int(idx[0])]))

    # half-open convention (your chosen one)
    def mass(mask):
        return float(pg[mask].sum())

    P_core   = mass(g < 0.5)
    P_shal   = mass((g >= 0.5) & (g < 1.0))
    P_nfw    = mass((g >= 1.0) & (g < 1.5))
    P_steep  = mass(g >= 1.5)
    print("binning_convention: core <0.5, shallow [0.5,1.0), nfw [1.0,1.5), steep >=1.5 (half-open)")
    print("P_core_<0.5", P_core)
    print("P_shallow_[0.5,1.0)", P_shal)
    print("P_nfw_[1.0,1.5)", P_nfw)
    print("P_steep_>=1.5", P_steep)
    print("P_sum_check", P_core + P_shal + P_nfw + P_steep)

    # gamma quantiles from the marginal (grid-based)
    c = np.cumsum(pg)
    def q(qv):
        return float(np.interp(qv, c, g))
    print("gamma_q03_q16_q50_q84_q97", q(0.03), q(0.16), q(0.50), q(0.84), q(0.97))

def scan_script(fp: Path) -> None:
    txt = fp.read_text(errors="replace")
    hits = sorted(set(re.findall(r'rc_gammaA_[A-Za-z0-9_]+\.npz', txt)))
    if hits:
        print(fp, "mentions", ", ".join(hits))

cand_from = repo / "results" / "rc_gammaA_from_nuts_samples.npz"
cand_grid = repo / "results" / "rc_gammaA_posterior_grid.npz"

print("CANDIDATES")
print("from_nuts", cand_from)
print("posterior_grid", cand_grid)

summarize_npz(cand_from)
summarize_npz(cand_grid)

print("\nSCRIPT SCAN (which NPZ filenames are referenced)")
for fp in [
    repo/"scripts/rc_gamma_posterior_core_vs_cusp.py",
    repo/"scripts/final_summary_qz_gamma.py",
    repo/"scripts/rc_gammaA_from_nuts_samples_build.py",
]:
    if fp.exists():
        scan_script(fp)
    else:
        print(fp, "MISSING")
