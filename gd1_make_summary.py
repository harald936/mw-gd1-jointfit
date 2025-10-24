#!/usr/bin/env python3
"""
Make a binned, weighted GD-1 'stream track' summary from data/gd1/gd1_catalog.csv:
- bins in phi1
- weighted medians (p_member_refined as weights)
- standard-error estimates (weighted)
Writes: data/gd1/gd1_summary.csv and quick-look plot.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

in_path  = Path("data/gd1/gd1_catalog.csv")
out_csv  = Path("data/gd1/gd1_summary.csv")
out_png  = Path("results/gd1_summary_track.png")

df = pd.read_csv(in_path)
# keep reasonably clean members (you can tune these)
keep = (df["p_member_refined"] >= 0.6) & (np.isfinite(df["phi1_deg"])) \
     & np.isfinite(df["phi2_deg"]) & np.isfinite(df["pm_phi1"]) & np.isfinite(df["pm_phi2"])
df = df.loc[keep].copy()
w  = df["p_member_refined"].values

phi1 = df["phi1_deg"].values
phi2 = df["phi2_deg"].values
pm1  = df["pm_phi1"].values
pm2  = df["pm_phi2"].values

# bin edges & centers
nbin = 60
edges = np.linspace(phi1.min(), phi1.max(), nbin+1)
cent  = 0.5*(edges[:-1] + edges[1:])

def wmedian(x, w):
    # weighted median
    s = np.argsort(x)
    x, w = x[s], w[s]
    c = np.cumsum(w) / np.sum(w)
    i = np.searchsorted(c, 0.5)
    return x[min(i, len(x)-1)]

def wstderr(x, w):
    # weighted standard error of the mean as a rough error bar
    mu = np.average(x, weights=w)
    var = np.average((x-mu)**2, weights=w)
    n_eff = (np.sum(w)**2) / np.sum(w**2)
    return np.sqrt(var / max(n_eff, 1.0))

rows = []
for i in range(nbin):
    m = (phi1 >= edges[i]) & (phi1 < edges[i+1])
    if m.sum() < 20:
        continue
    wi = w[m]
    rows.append({
        "phi1_deg": cent[i],
        "phi2_med": wmedian(phi2[m], wi),
        "phi2_se" : wstderr(phi2[m], wi),
        "pm_phi1_med": wmedian(pm1[m], wi),
        "pm_phi1_se" : wstderr(pm1[m], wi),
        "pm_phi2_med": wmedian(pm2[m], wi),
        "pm_phi2_se" : wstderr(pm2[m], wi),
        "n_eff" : (wi.sum()**2)/np.sum(wi**2)
    })

track = pd.DataFrame(rows)
track.to_csv(out_csv, index=False)
print("Wrote:", out_csv, "rows:", len(track))

# quick-look plot
plt.figure(figsize=(6.5,4))
plt.subplot(2,1,1)
plt.errorbar(track["phi1_deg"], track["phi2_med"], yerr=track["phi2_se"], fmt=".", capsize=2)
plt.ylabel(r"$\phi_2$ [deg]"); plt.title("GD-1 binned stream track (weighted)")
plt.subplot(2,1,2)
plt.errorbar(track["phi1_deg"], track["pm_phi1_med"], yerr=track["pm_phi1_se"], fmt=".", capsize=2, label="pm_phi1")
plt.errorbar(track["phi1_deg"], track["pm_phi2_med"], yerr=track["pm_phi2_se"], fmt=".", capsize=2, label="pm_phi2")
plt.xlabel(r"$\phi_1$ [deg]"); plt.ylabel("pm [mas/yr]"); plt.legend()
plt.tight_layout(); plt.savefig(out_png, dpi=140)
print("Saved:", out_png)
