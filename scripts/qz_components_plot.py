import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

results = Path("results")
txt = results / "joint_qz_components.txt"

qz_joint = None
sig_joint = None
qz_stream = None
sig_stream = None

with open(txt) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        if parts[0] == "qz_fit":
            qz_joint = float(parts[1])
        elif parts[0] == "sigma_qz":
            sig_joint = float(parts[1])
        elif parts[0] == "qz_stream":
            qz_stream = float(parts[1])
        elif parts[0] == "sigma_stream":
            sig_stream = float(parts[1])

if None in (qz_joint, sig_joint, qz_stream, sig_stream):
    raise SystemExit("Missing values in joint_qz_components.txt")

qmin = min(qz_joint - 5*sig_joint, qz_stream - 5*sig_stream, 0.8)
qmax = max(qz_joint + 5*sig_joint, qz_stream + 5*sig_stream, 1.2)
q = np.linspace(qmin, qmax, 400)

chi2_joint = (q - qz_joint)**2 / sig_joint**2
chi2_stream = (q - qz_stream)**2 / sig_stream**2

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(q, chi2_joint, label="joint (stream+RC)")
ax.plot(q, chi2_stream, label="stream only", linestyle="--")
ax.axvline(1.0, color="k", linestyle=":", linewidth=1)
ax.set_xlabel(r"$q_z$")
ax.set_ylabel(r"$\Delta\chi^2$")
ax.set_title(r"q$_z$ constraints: stream vs joint")
ax.legend()
fig.tight_layout()

pdf = results / "joint_qz_components_plot.pdf"
png = results / "joint_qz_components_plot.png"
fig.savefig(pdf)
fig.savefig(png)
print(pdf)
print(png)
