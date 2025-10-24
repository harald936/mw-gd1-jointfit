#!/usr/bin/env python3
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("data/gd1/gd1_lite.csv")

# 1) Proper-motion plane, colored by p_member
plt.figure(figsize=(5.5,4.5))
plt.scatter(df.pm_phi1, df.pm_phi2, s=2, c=df.p_member, cmap="viridis", alpha=0.4)
plt.colorbar(label="p_member")
plt.xlabel("pm_phi1 [mas/yr]"); plt.ylabel("pm_phi2 [mas/yr]")
plt.title("GD-1 field: PM plane")
plt.tight_layout(); plt.savefig("results/gd1_pm_plane.png", dpi=140)

# 2) Sky track (phi1, phi2), highlight p_member>0.5
plt.figure(figsize=(6,4.5))
mask = df.p_member > 0.5
plt.scatter(df.phi1_deg, df.phi2_deg, s=1, c="lightgray", alpha=0.2, label="all")
plt.scatter(df.loc[mask,"phi1_deg"], df.loc[mask,"phi2_deg"], s=2, c="crimson", alpha=0.5, label="p>0.5")
plt.xlabel("phi1 [deg]"); plt.ylabel("phi2 [deg]")
plt.legend(markerscale=4)
plt.title("GD-1 field in stream-aligned sky")
plt.tight_layout(); plt.savefig("results/gd1_sky_phi.png", dpi=140)

# 3) p_member histogram
plt.figure(figsize=(5.0,3.6))
plt.hist(df.p_member, bins=50, color="steelblue", alpha=0.8)
plt.xlabel("p_member"); plt.ylabel("count"); plt.title("p_member distribution")
plt.tight_layout(); plt.savefig("results/gd1_p_hist.png", dpi=140)

print("Saved: results/gd1_pm_plane.png, results/gd1_sky_phi.png, results/gd1_p_hist.png")
