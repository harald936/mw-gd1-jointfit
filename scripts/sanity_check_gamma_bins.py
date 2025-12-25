import numpy as np
from pathlib import Path

z = np.load("results/rc_gammaA_from_nuts_samples.npz")
g = np.asarray(z["gamma"], float)
P = np.asarray(z["posterior"], float)

pg = P.sum(axis=1)
pg = pg / pg.sum()

def mass(mask):
    return float(pg[mask].sum())

print("npz", Path("results/rc_gammaA_from_nuts_samples.npz").resolve())
print("Ng", g.size, "gmin", float(g.min()), "gmax", float(g.max()))

for val in [0.5, 1.0, 1.5]:
    idx = np.where(np.isclose(g, val))[0]
    if len(idx)==0:
        print("no_gridpoint_at", val)
    else:
        i = int(idx[0])
        print(f"mass_at_gamma_{val}", float(pg[i]))

A = (mass(g<=0.5), mass((g>0.5)&(g<=1.0)), mass((g>1.0)&(g<=1.5)), mass(g>1.5))
B = (mass(g<0.5),  mass((g>=0.5)&(g<1.0)), mass((g>=1.0)&(g<1.5)), mass(g>=1.5))

print("convention_A <=0.5, (0.5,1], (1,1.5], >1.5", *A)
print("convention_B <0.5, [0.5,1), [1,1.5), >=1.5", *B)
print("A_minus_B", *(np.array(A)-np.array(B)))

