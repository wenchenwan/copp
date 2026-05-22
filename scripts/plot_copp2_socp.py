"""
COPP2-SOCP results compared with TOPP2-RA baseline.
Run from the project root:  python scripts/plot_copp2_socp.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ra   = pd.read_csv("output/data/topp2_ra.csv")
socp = pd.read_csv("output/data/copp2_socp.csv")
traj_ra   = pd.read_csv("output/data/topp2_ra_traj.csv")
traj_socp = pd.read_csv("output/data/copp2_socp_traj.csv")

s        = ra["s"].values
a_ra     = ra["a"].values
t_s_ra   = ra["t_s"].values
a_socp   = socp["a"].values
t_s_socp = socp["t_s"].values

tf_ra   = t_s_ra[-1]
tf_socp = t_s_socp[-1]

t_ra   = traj_ra["t"].values;    st_ra   = traj_ra["s_t"].values
t_socp = traj_socp["t"].values;  st_socp = traj_socp["s_t"].values

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"COPP2-SOCP vs TOPP2-RA\n"
    f"t_final:  TOPP2-RA = {tf_ra:.4f} s   COPP2-SOCP = {tf_socp:.4f} s",
    fontsize=13, fontweight="bold",
)

# (0,0) a(s) comparison
ax = axes[0, 0]
ax.plot(s, a_ra,   label=f"TOPP2-RA   tf={tf_ra:.3f}s",   color="#1f77b4", lw=1.5)
ax.plot(s, a_socp, label=f"COPP2-SOCP tf={tf_socp:.3f}s", color="#d62728", lw=1.5, linestyle="--")
ax.set_xlabel("s"); ax.set_ylabel("a(s) = ṡ²")
ax.set_title("Velocity-Squared Profile a(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (0,1) ṡ(s) comparison
ax = axes[0, 1]
ax.plot(s, np.sqrt(a_ra),   color="#1f77b4", lw=1.5, label="TOPP2-RA")
ax.plot(s, np.sqrt(a_socp), color="#d62728", lw=1.5, linestyle="--", label="COPP2-SOCP")
ax.set_xlabel("s"); ax.set_ylabel("ṡ(s) = √a(s)")
ax.set_title("Path Speed Profile ṡ(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (1,0) s(t) comparison
ax = axes[1, 0]
t_max = max(tf_ra, tf_socp)
ax.plot(t_ra,   st_ra,   color="#1f77b4", lw=1.5, label="TOPP2-RA")
ax.plot(t_socp, st_socp, color="#d62728", lw=1.5, linestyle="--", label="COPP2-SOCP")
ax.set_xlabel("t  (s)"); ax.set_ylabel("s(t)")
ax.set_title("Trajectory s(t)")
ax.set_xlim(0, t_max * 1.01); ax.set_ylim(0, 1)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (1,1) timing map t(s) comparison
ax = axes[1, 1]
ax.plot(s, t_s_ra,   color="#1f77b4", lw=1.5, label="TOPP2-RA")
ax.plot(s, t_s_socp, color="#d62728", lw=1.5, linestyle="--", label="COPP2-SOCP")
ax.set_xlabel("s"); ax.set_ylabel("t(s)  (s)")
ax.set_title("Timing Map t(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
out = "output/plot_copp2_socp.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
