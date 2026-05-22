"""
TOPP2-RA results: speed profile, timing map, trajectory, and phase portrait.
Run from the project root:  python scripts/plot_topp2_ra.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

profile = pd.read_csv("output/data/topp2_ra.csv")
traj    = pd.read_csv("output/data/topp2_ra_traj.csv")

s    = profile["s"].values
a    = profile["a"].values      # a(s) = ṡ²
t_s  = profile["t_s"].values    # timing map t(s)
t    = traj["t"].values
s_t  = traj["s_t"].values
t_final = t_s[-1]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"TOPP2-RA  —  t_final = {t_final:.4f} s", fontsize=14, fontweight="bold")

# (0,0) a(s) = ṡ²
ax = axes[0, 0]
ax.plot(s, a, color="#1f77b4", linewidth=1.5)
ax.set_xlabel("s  (path parameter)")
ax.set_ylabel("a(s) = ṡ²")
ax.set_title("Velocity-Squared Profile a(s)")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# (0,1) ṡ(s) = √a(s)
ax = axes[0, 1]
ax.plot(s, np.sqrt(a), color="#ff7f0e", linewidth=1.5)
ax.set_xlabel("s  (path parameter)")
ax.set_ylabel("ṡ(s) = √a(s)  (path speed)")
ax.set_title("Path Speed Profile ṡ(s)")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# (1,0) s(t) trajectory
ax = axes[1, 0]
ax.plot(t, s_t, color="#2ca02c", linewidth=1.5)
ax.set_xlabel("t  (s)")
ax.set_ylabel("s(t)")
ax.set_title("Trajectory s(t)")
ax.set_xlim(0, t_final)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# (1,1) timing map t(s)
ax = axes[1, 1]
ax.plot(s, t_s, color="#9467bd", linewidth=1.5)
ax.set_xlabel("s  (path parameter)")
ax.set_ylabel("t(s)  (s)")
ax.set_title("Timing Map t(s)")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
out = "output/plot_topp2_ra.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
