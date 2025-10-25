#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.coordinates import SkyCoord
import gala.coordinates as gc
from sklearn.mixture import GaussianMixture

# --- paths ---
data_dir = Path("data/gd1"); data_dir.mkdir(parents=True, exist_ok=True)
raw_csv  = data_dir / "gd1_raw_dr3.csv"
lite_csv = data_dir / "gd1_lite.csv"

# --- Explicitly use DR3 at ESA TAP (latest full Gaia release) ---
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1
Gaia.TAP_URL   = "https://gea.esac.esa.int/tap-server/tap"

# --- region (tune later if needed) ---
ra_min, ra_max = 130, 200
dec_min, dec_max = 10, 50

adql = f"""
SELECT TOP 300000
  source_id, ra, dec, parallax, ruwe,
  pmra, pmdec, phot_g_mean_mag, bp_rp
FROM {Gaia.MAIN_GAIA_TABLE}
WHERE ra BETWEEN {ra_min} AND {ra_max}
  AND dec BETWEEN {dec_min} AND {dec_max}
  AND ruwe < 1.4
  AND parallax BETWEEN -0.5 AND 1.0
  AND phot_g_mean_mag < 20.5
"""

print("Submitting Gaia DR3 ADQL to ESA TAP…")
print(Gaia.TAP_URL, "| table:", Gaia.MAIN_GAIA_TABLE)

# NOTE: keep arguments minimal for broad astroquery compatibility
job = Gaia.launch_job_async(adql)
tab = job.get_results()
print("Server returned rows:", len(tab))

# write raw CSV
df_raw = tab.to_pandas()
df_raw.to_csv(raw_csv, index=False)
print("Wrote raw:", raw_csv, "| rows:", len(df_raw))

if len(df_raw) == 0:
    print("No rows returned. Widen box or relax cuts; then re-run.")
    raise SystemExit(0)

# --- transform to GD-1 coordinates ---
sc = SkyCoord(ra=df_raw["ra"].values*u.deg, dec=df_raw["dec"].values*u.deg, distance=10*u.kpc, frame="icrs")
gd1_frame = gc.GD1Koposov10()
gd1 = sc.transform_to(gd1_frame)
df_raw["phi1_deg"] = gd1.phi1.to(u.deg).value
df_raw["phi2_deg"] = gd1.phi2.to(u.deg).value
df_raw["pm_phi1"]  = gd1.pm_phi1_cosphi2.to(u.mas/u.yr).value
df_raw["pm_phi2"]  = gd1.pm_phi2.to(u.mas/u.yr).value

# --- focus subset ---
mask_focus = (np.abs(df_raw["phi2_deg"]) < 10) & (df_raw["phot_g_mean_mag"] < 20.0)
df = df_raw.loc[mask_focus].copy()
print("Focus subset rows:", len(df))

# --- simple 2-GMM on (pm_phi1, pm_phi2) for p_member ---
if len(df) >= 20:
    X = np.vstack([df["pm_phi1"].values, df["pm_phi2"].values]).T
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
    P   = gmm.predict_proba(X)
    cov_dets = [np.linalg.det(gmm.covariances_[k]) for k in range(2)]
    stream_k = int(np.argmin(cov_dets))
    df["p_member"] = P[:, stream_k]
else:
    df["p_member"] = 0.0

# --- write lite CSV + selection / pmzp placeholders ---
df.to_csv(lite_csv, index=False)
print("Wrote lite:", lite_csv, "| rows:", len(df))

sel = {
  "sky_box": {"ra_min": ra_min, "ra_max": ra_max, "dec_min": dec_min, "dec_max": dec_max},
  "pm_window_note": "p_member from 2-GMM on (pm_phi1, pm_phi2); no hard PM window.",
  "notes": "Gaia DR3 (gaiadr3.gaia_source) at ESA TAP."
}
with open(data_dir/"selection.json","w") as f: json.dump(sel, f, indent=2)

pmzp = {"grid": "uniform", "pmra_zp_masyr": 0.0, "pmdec_zp_masyr": 0.0}
with open(data_dir/"pm_zero_point.json","w") as f: json.dump(pmzp, f, indent=2)

print("Also wrote selection.json and pm_zero_point.json")
print("Done.")
