
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from pathlib import Path

def pick(df, names):
    for n in names:
        if n in df.columns: return df[n].to_numpy()
    raise KeyError(f"None of {names} in columns: {df.columns.tolist()}")

df = pd.read_csv("data/gd1/gd1_lite.csv")
phi1 = pick(df, ["phi1","phi1_deg","phi1_gd1"])
mu   = pick(df, ["pm_phi1","pmphi1","mu_phi1"])
err  = pick(df, ["pm_phi1_err","pmphi1_err","mu_phi1_err"]) if any(c in df.columns for c in ["pm_phi1_err","pmphi1_err","mu_phi1_err"]) else np.full_like(mu, 0.15)

m = np.isfinite(phi1) & np.isfinite(mu) & np.isfinite(err)
phi1, mu, err = phi1[m], mu[m], err[m]

bins = np.linspace(-70,-50,25)
bc = 0.5*(bins[1:]+bins[:-1])
idx = np.digitize(phi1, bins) - 1
good = (idx>=0) & (idx<len(bc))
phi_b = np.array([bc[i] for i in idx[good]])
mu_b  = mu[good]
w_b   = 1/np.clip(err[good],0.05,5.0)**2
phi0  = np.average(phi_b, weights=w_b)
x     = phi_b - phi0

def template(phi, qz):
    t = (phi - phi0)**2
    t = t - np.average(t, weights=w_b)
    amp = (qz - 1.0) + 0.6*(qz - 1.0)**2
    return amp * t

def chi2_of_q(q):
    T = template(phi_b, q)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w_b)
    beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ (mu_b - T))
    res = mu_b - (A @ beta + T)
    return float((res*res*w_b).sum())

grid = np.linspace(0.70, 1.30, 121)
vals = np.array([chi2_of_q(q) for q in grid], dtype=float)
cmin = float(vals.min())
qbest = float(grid[vals.argmin()])

import numpy as np
def brack(delta):
    y = vals - cmin - delta
    s = np.sign(y)
    edges=[]
    for i in range(len(grid)-1):
        if s[i]==0: edges.append(grid[i])
        if s[i]*s[i+1] < 0:
            a,b = grid[i], grid[i+1]; ya,yb=y[i],y[i+1]
            t = a - ya*(b-a)/(yb-ya)
            edges.append(float(t))
    return (edges[0], edges[-1]) if len(edges)>=2 else (np.nan, np.nan)

L68,R68 = brack(1.0)
L95,R95 = brack(3.84)
sigma = 0.5*(R68 - L68) if np.isfinite(L68) and np.isfinite(R68) else np.nan

Path("results").mkdir(exist_ok=True)
png = Path("results/joint_qz_profile.png")
plt.figure(figsize=(7.2,4.4), dpi=180)
plt.plot(grid, vals - cmin, lw=2.2, label=r'$\Delta\chi^2$')
plt.axhline(1.0, ls='--'); plt.axhline(3.84, ls=':')
plt.axvline(qbest, ls='-')
plt.xlabel(r'$q_z$'); plt.ylabel(r'$\Delta\chi^2$')
plt.title('GD-1 profile likelihood for $q_z$ (offset+slope profiled)')
plt.legend(); plt.tight_layout()
plt.savefig(png, bbox_inches="tight")
plt.close()

txt = f"qz_fit={qbest:.3f}\nsigma_qz={sigma:.3f}\nci68=[{L68:.3f},{R68:.3f}]\nci95=[{L95:.3f},{R95:.3f}]\nspan={float((vals-cmin).ptp()):.3f}\nmethod=proxy(offset+slope, quadratic response)\n"
Path("results/joint_qz.txt").write_text(txt)
print('WROTE', png, os.path.getsize(png), 'bytes')
print(txt)
