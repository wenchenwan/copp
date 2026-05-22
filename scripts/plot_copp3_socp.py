"""
COPP3-SOCP results: a(s), b(s)=s̈, and s(t) for SCP iterations 1 and 2.
Objective: time + 0.1 × thermal energy.
Run from the project root:  python scripts/plot_copp3_socp.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

d1 = pd.read_csv("output/data/copp3_socp_iter1.csv")
d2 = pd.read_csv("output/data/copp3_socp_iter2.csv")
t1 = pd.read_csv("output/data/copp3_socp_iter1_traj.csv")
t2 = pd.read_csv("output/data/copp3_socp_iter2_traj.csv")

s      = d1["s"].values
a1, b1 = d1["a"].values, d1["b"].values
a2, b2 = d2["a"].values, d2["b"].values
tf1, tf2 = d1["t_s"].values[-1], d2["t_s"].values[-1]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"COPP3-SOCP  (objective: time + 0.1 × thermal energy, SCP iterations)\n"
    f"t_final:  iter1 = {tf1:.4f} s   →   iter2 = {tf2:.4f} s",
    fontsize=13, fontweight="bold",
)

C1, C2 = "#8c564b", "#e377c2"

# (0,0) a(s)
ax = axes[0, 0]
ax.plot(s, a1, color=C1, lw=1.8, label=f"Iter 1  tf={tf1:.3f}s")
ax.plot(s, a2, color=C2, lw=1.8, linestyle="--", label=f"Iter 2  tf={tf2:.3f}s")
ax.set_xlabel("s"); ax.set_ylabel("a(s) = ṡ²")
ax.set_title("Velocity-Squared Profile a(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(True, alpha=0.3)

# (0,1) path speed
ax = axes[0, 1]
ax.plot(s, np.sqrt(a1), color=C1, lw=1.8, label="Iter 1")
ax.plot(s, np.sqrt(a2), color=C2, lw=1.8, linestyle="--", label="Iter 2")
ax.set_xlabel("s"); ax.set_ylabel("ṡ(s) = √a(s)")
ax.set_title("Path Speed Profile ṡ(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(); ax.grid(True, alpha=0.3)

# (1,0) b(s)
ax = axes[1, 0]
ax.plot(s, b1, color=C1, lw=1.5, label="Iter 1")
ax.plot(s, b2, color=C2, lw=1.5, linestyle="--", label="Iter 2")
ax.axhline(0, color="k", lw=0.6, linestyle=":")
ax.set_xlabel("s"); ax.set_ylabel("b(s) = s̈")
ax.set_title("Path Acceleration Profile b(s) = s̈")
ax.set_xlim(0, 1)
ax.legend(); ax.grid(True, alpha=0.3)

# (1,1) trajectory
ax = axes[1, 1]
ax.plot(t1["t"], t1["s_t"], color=C1, lw=1.8, label=f"Iter 1  tf={tf1:.3f}s")
ax.plot(t2["t"], t2["s_t"], color=C2, lw=1.8, linestyle="--", label=f"Iter 2  tf={tf2:.3f}s")
ax.set_xlabel("t  (s)"); ax.set_ylabel("s(t)")
ax.set_title("Trajectory s(t)")
ax.set_xlim(0, max(tf1, tf2) * 1.01); ax.set_ylim(0, 1)
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
out = "output/plot_copp3_socp.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
