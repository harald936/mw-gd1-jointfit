set -e
mkdir -p src/models

cat > src/models/vphi_axisym.py << 'PY'
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
G = 4.300917270e-6
@dataclass
class DiskMN:
    M: float; a: float; b: float
    def vc2(self, R: np.ndarray) -> np.ndarray:
        B = self.a + self.b
        return G*self.M * (R*R) / (R*R + B*B)**1.5
@dataclass
class BulgeHernquist:
    M: float; a: float
    def vc2(self, R: np.ndarray) -> np.ndarray:
        return G*self.M * R / (R + self.a)**2
@dataclass
class Halo_gNFW:
    rho0: float; rs: float; gamma: float; q: float = 1.0
    def rho(self, r: np.ndarray) -> np.ndarray:
        x = r / self.rs
        return self.rho0 * (x**(-self.gamma)) * (1.0 + x)**(self.gamma - 3.0)
    def _mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        r = np.atleast_1d(r); m = np.empty_like(r)
        for i, ri in enumerate(r):
            if ri <= 0: m[i] = 0.0; continue
            x = np.linspace(0.0, ri, 2048); x[0] = 1e-9
            y = self.rho(x) * x*x
            m[i] = 4*np.pi * np.trapz(y, x)
        return m
    def vc2(self, R: np.ndarray) -> np.ndarray:
        M_enc = self._mass_enclosed(R)
        return G * M_enc / np.clip(R, 1e-6, None)
def vphi_total(R: np.ndarray, disk: DiskMN, bulge: BulgeHernquist, halo: Halo_gNFW) -> np.ndarray:
    return np.sqrt(disk.vc2(R) + bulge.vc2(R) + halo.vc2(R))
PY

cat > rc_gp_fit_physical.py << 'PY'
#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from src.models.vphi_axisym import DiskMN, BulgeHernquist, Halo_gNFW, vphi_total

csv = Path("data/rotation_curve/beordo2024/beordo2024_allstars_rc_ready.csv")
df = pd.read_csv(csv).sort_values("R_kpc").reset_index(drop=True)
R = df["R_kpc"].to_numpy(float); y = df["Vphi_kms"].to_numpy(float); sigma = df["sigma_obs_kms"].to_numpy(float)

def matern32(x,y,amp,scale):
    dx = np.abs(x[:,None]-y[None,:]); z = np.sqrt(3.0)*dx/scale
    return (amp**2)*(1.0+z)*np.exp(-z)

def unpack(th):
    logMd,a_d,b_d, logMb,a_b, logrho0,rs,gamma, log_amp,log_scale,log_jitter = th
    disk  = DiskMN(M=np.exp(logMd), a=a_d, b=b_d)
    bulge = BulgeHernquist(M=np.exp(logMb), a=a_b)
    halo  = Halo_gNFW(rho0=np.exp(logrho0), rs=rs, gamma=gamma, q=1.0)
    return disk, bulge, halo, np.exp(log_amp), np.exp(log_scale), np.exp(log_jitter)

def nll(th):
    disk, bulge, halo, amp, scale, jitter = unpack(th)
    mu = vphi_total(R, disk, bulge, halo)
    K = matern32(R,R,amp,scale); K[np.diag_indices_from(K)] += (sigma**2 + jitter**2)
    try:
        c,l = cho_factor(K, check_finite=False)
        a = cho_solve((c,l), (y-mu), check_finite=False)
        logdet = 2*np.sum(np.log(np.diag(c)))
        return 0.5*((y-mu)@a + logdet + len(R)*np.log(2*np.pi))
    except Exception:
        return 1e25

theta0 = np.array([np.log(6e10),6.0,0.3, np.log(7e9),0.6, np.log(1e7),20.0,1.0, np.log(10.0),np.log(3.0),np.log(2.0)], float)
bounds = [(np.log(1e9),np.log(3e11)),(2.0,10.0),(0.05,1.5),(np.log(1e8),np.log(5e10)),(0.1,2.0),(np.log(1e5),np.log(1e9)),(5.0,50.0),(0.2,1.8),(np.log(0.5),np.log(50.0)),(np.log(0.3),np.log(8.0)),(np.log(0.1),np.log(20.0))]
res = minimize(lambda th: float(nll(th)), x0=theta0, method="L-BFGS-B", bounds=bounds)
print("MLE success:", res.success, "| message:", res.message)
th = res.x; disk, bulge, halo, amp, scale, jitter = unpack(th)
print(f"Disk M={disk.M:.3e} a={disk.a:.2f} b={disk.b:.2f}; Bulge M={bulge.M:.3e} a={bulge.a:.2f}; Halo rho0={halo.rho0:.3e} rs={halo.rs:.2f} gamma={halo.gamma:.2f}; GP amp={amp:.2f} scale={scale:.2f} jitter={jitter:.2f}")

mu = vphi_total(R, disk, bulge, halo)
K = matern32(R,R,amp,scale); K[np.diag_indices_from(K)] += (sigma**2 + jitter**2)
c,l = cho_factor(K, check_finite=False)
a = cho_solve((c,l), (y-mu), check_finite=False)
pred_mu = mu + K @ a
v = cho_solve((c,l), K, check_finite=False)
pred_std = np.sqrt(np.clip(np.diag(K - K @ v), 0.0, np.inf))

ii = np.argsort(R)
plt.figure(figsize=(7,4))
plt.errorbar(R,y,yerr=sigma,fmt=".",capsize=2,label="data")
plt.plot(R[ii],mu[ii],lw=2,label="Vphi model")
plt.plot(R[ii],pred_mu[ii],lw=1.5,ls="--",label="model+GP")
plt.fill_between(R[ii],(pred_mu-pred_std)[ii],(pred_mu+pred_std)[ii],alpha=0.2,label="±1σ latent")
plt.xlabel("R [kpc]"); plt.ylabel("Vφ [km/s]"); plt.title("Axisymmetric mean + residual GP")
plt.legend(); plt.tight_layout()
out = Path("data/rotation_curve/beordo2024/rc_gp_fit_physical.png")
plt.savefig(out,dpi=160); print("Saved plot:", out)
