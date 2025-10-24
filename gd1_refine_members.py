#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

in_csv  = Path("data/gd1/gd1_lite.csv")
out_csv = Path("data/gd1/gd1_catalog.csv")
meta_js = Path("data/gd1/catalog_meta.json")
Path("results").mkdir(exist_ok=True)

df = pd.read_csv(in_csv)

# ----- 1) get a clean core to learn the distance/G-ridge vs phi1 -----
core = df[(df["p_member"]>0.8) & (df["phi2_deg"].abs()<3) & np.isfinite(df["bp_rp"])].copy()
if len(core) < 2000:
    print("Warning: few core stars; consider p_member>0.7 or |phi2|<5.")
# robust binning: median G per phi1 bin
bins = np.linspace(core["phi1_deg"].min(), core["phi1_deg"].max(), 40)
ix   = np.clip(np.digitize(core["phi1_deg"], bins)-1, 0, len(bins)-2)
g_med = core.groupby(ix)["phot_g_mean_mag"].median().to_numpy()
phi_mid = 0.5*(bins[:-1]+bins[1:])
mask_valid = np.isfinite(g_med)
phi_mid, g_med = phi_mid[mask_valid], g_med[mask_valid]
# linear fit: G_ridge(phi1) ~ a*phi1 + b (captures distance gradient)
A = np.vstack([phi_mid, np.ones_like(phi_mid)]).T
a, b = np.linalg.lstsq(A, g_med, rcond=None)[0]

# ----- 2) CMD matched filter -----
# allow the ridge to vary in brightness with phi1; accept stars close to it
G   = df["phot_g_mean_mag"].values
phi = df["phi1_deg"].values
G_ridge = a*phi + b
# broad color lane typical for old, metal-poor pop; you can tighten later
color = df["bp_rp"].values
cmd_keep = (np.abs(G - G_ridge) < 0.6) & (color > 0.2) & (color < 1.4)

# ----- 3) spatial + PM cut (loose), then 3-D GMM on filtered sample -----
pre = df[cmd_keep & (df["phi2_deg"].abs()<10)].copy()
X = np.vstack([pre["pm_phi1"].values, pre["pm_phi2"].values, pre["phi2_deg"].values]).T
finite = np.isfinite(X).all(axis=1)
pre, X = pre.loc[finite], X[finite]

n = len(pre)
if n < 200:
    print("Too few stars after filter; loosen CMD or |phi2|.")
    pre["p_member_refined"] = 0.0
    df["p_member_refined"]  = 0.0
else:
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
    P   = gmm.predict_proba(X)
    # identify narrower component as the stream
    cov_dets = [np.linalg.det(gmm.covariances_[k]) for k in range(2)]
    k_stream = int(np.argmin(cov_dets))
    p_ref = np.zeros(n)
    p_ref[:] = P[:, k_stream]
    pre["p_member_refined"] = p_ref
    df["p_member_refined"]  = 0.0
    df.loc[pre.index, "p_member_refined"] = pre["p_member_refined"].values

# ----- 4) save catalog + metadata -----
df.to_csv(out_csv, index=False)

meta = {
  "source": "Gaia DR3 (gaiadr3.gaia_source)",
  "steps": {
    "ridge_fit": {"model": "G ~ a*phi1 + b", "a": float(a), "b": float(b), "band_mag": 0.6},
    "cmd_filter": {"bp_rp_range": [0.2, 1.4]},
    "gmm_space": ["pm_phi1","pm_phi2","phi2_deg"], "gmm_k": 2
  },
  "notes": "This is a simple, reproducible refinement. You can tighten bands or add RVs later."
}
with open(meta_js, "w") as f: json.dump(meta, f, indent=2)

# ----- 5) quick plots -----
import matplotlib
matplotlib.use("Agg")

# PM plane colored by refined probability
plt.figure(figsize=(5.5,4.5))
plt.scatter(df["pm_phi1"], df["pm_phi2"], s=1, c=df["p_member_refined"], cmap="viridis", alpha=0.4)
plt.colorbar(label="p_member_refined")
plt.xlabel("pm_phi1 [mas/yr]"); plt.ylabel("pm_phi2 [mas/yr]")
plt.title("GD-1 refined: PM plane")
plt.tight_layout(); plt.savefig("results/gd1_pm_refined.png", dpi=140)

# Sky track highlighting refined members
plt.figure(figsize=(6,4.5))
mask = df["p_member_refined"]>0.6
plt.scatter(df["phi1_deg"], df["phi2_deg"], s=1, c="lightgray", alpha=0.25, label="all")
plt.scatter(df.loc[mask,"phi1_deg"], df.loc[mask,"phi2_deg"], s=2, c="crimson", alpha=0.6, label="p_ref>0.6")
plt.xlabel("phi1 [deg]"); plt.ylabel("phi2 [deg]")
plt.legend(markerscale=4); plt.title("GD-1 refined in stream frame")
plt.tight_layout(); plt.savefig("results/gd1_sky_refined.png", dpi=140)

# CMD with ridge
plt.figure(figsize=(5.5,4.5))
plt.scatter(color, G, s=1, c="lightgray", alpha=0.25)
plt.scatter(df.loc[mask,"bp_rp"], df.loc[mask,"phot_g_mean_mag"], s=2, c="green", alpha=0.6, label="p_ref>0.6")
# ridge line
phi_grid = np.linspace(df["phi1_deg"].min(), df["phi1_deg"].max(), 100)
G_grid   = a*phi_grid + b
# project ridge into apparent CMD using median color of core as a guide
plt.plot(np.full_like(phi_grid, np.nanmedian(core["bp_rp"])), G_grid, color="k", lw=2, label="G ridge(φ1)")
plt.gca().invert_yaxis()
plt.xlabel("BP-RP"); plt.ylabel("G")
plt.legend(); plt.title("CMD + distance ridge (empirical)")
plt.tight_layout(); plt.savefig("results/gd1_cmd_ridge.png", dpi=140)

print("Wrote:")
print(" ", out_csv)
print(" ", meta_js)
print(" results/gd1_pm_refined.png, gd1_sky_refined.png, gd1_cmd_ridge.png")
