from pathlib import Path
import importlib.util
import numpy as np

def load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def report(L, name, theta):
    c1 = L["chi2_rc"](theta)
    c2 = L["chi2_stream"](theta)
    ct = c1 + c2
    ll = -0.5 * ct
    print(name)
    print("theta", float(theta[0]), float(theta[1]), float(theta[2]))
    print("chi2_rc", c1)
    print("chi2_stream", c2)
    print("chi2_total", ct)
    print("logL", ll)

def main():
    repo = Path(__file__).resolve().parents[1]
    api_path = repo / "scripts" / "v3b_api.py"
    api = load_module("v3b_api", api_path)
    L = api.make_likelihood(str(repo))

    chain_npz = repo / "results" / "true_joint_emcee_orbitfit_v3b_chain.npz"
    d = np.load(chain_npz)
    chain = d["chain"]
    lp = d["log_prob"]

    flat_chain = chain.reshape(-1, 3)
    flat_lp = lp.reshape(-1)
    i = int(np.nanargmax(flat_lp))
    best = flat_chain[i]
    med = np.nanmedian(flat_chain, axis=0)

    report(L, "posterior_best", best)
    report(L, "posterior_median", med)

if __name__ == "__main__":
    main()
