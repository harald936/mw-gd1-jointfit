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

def vcirc(R_kpc, z_kpc=0.0, params=None, return_components=False):
    """
    Circular speed (km/s) at cylindrical (R,z) in kpc for the axisymmetric model.
    Inputs: R_kpc, z_kpc in kpc. Returns: km/s.
    Model parts:
      - Bulge: Plummer (Mb [Msun], b [kpc])
      - Disk : Miyamoto–Nagai (Md [Msun], a,b [kpc])
      - Halo : gNFW-like with flattening (vh [km/s], rh [kpc], gamma, qz)
    """
    import numpy as _np
    G = 4.30091727003628e-6  # kpc (km/s)^2 Msun^-1

    R = _np.asarray(R_kpc, dtype=float)
    z = _np.asarray(z_kpc, dtype=float)

    # Bulge: Plummer
    Mb = float(params['bulge']['Mb'])
    bb = float(params['bulge']['b'])
    r2 = R*R + z*z
    Vb2 = G * Mb * R*R / (r2 + bb*bb)**1.5

    # Disk: Miyamoto–Nagai
    Md = float(params['disk']['Md'])
    a  = float(params['disk']['a'])
    b  = float(params['disk']['b'])
    B  = _np.sqrt(z*z + b*b)
    denom_d = (R*R + (a + B)*(a + B))**1.5
    Vd2 = G * Md * R*R / denom_d

    # Halo: gNFW-like, flattened
    vh    = float(params['halo']['vh'])
    rh    = float(params['halo']['rh'])
    gamma = float(params['halo'].get('gamma', 1.0))
    qz    = float(params['halo'].get('qz', 1.0))
    eps   = 1e-12
    qz_eff = qz if abs(qz) > 1e-3 else 1e-3

    m = _np.sqrt(R*R + (z/qz_eff)**2)
    m_eff  = _np.maximum(m, eps)
    rh_eff = max(rh, eps)
    alpha = 2.0 - gamma
    Vh2 = (vh*vh) * (m_eff / (m_eff + rh_eff))**alpha * (R*R / (m_eff*m_eff + eps))

    V2 = _np.clip(Vb2, 0, _np.inf) + _np.clip(Vd2, 0, _np.inf) + _np.clip(Vh2, 0, _np.inf)
    V  = _np.sqrt(V2)
    if return_components:
        return V, {
            'bulge': _np.sqrt(_np.clip(Vb2, 0, _np.inf)),
            'disk' : _np.sqrt(_np.clip(Vd2, 0, _np.inf)),
            'halo' : _np.sqrt(_np.clip(Vh2, 0, _np.inf)),
        }
    return V

