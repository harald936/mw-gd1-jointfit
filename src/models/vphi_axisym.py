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
