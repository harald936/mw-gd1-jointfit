from pathlib import Path
import pandas as pd

def load_beordo_allstars_ready(repo_root: str | Path = ".") -> pd.DataFrame:
    p = Path(repo_root) / "data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv"
    df = pd.read_csv(p)
    cols = ["R_kpc","Vphi_kms","eVphi_plus_kms","eVphi_minus_kms","sigma_obs_kms"]
    return df[cols].sort_values("R_kpc").reset_index(drop=True)
