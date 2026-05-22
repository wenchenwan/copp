"""
Reconstruct and plot original joint-space trajectories p(t), v(t), a(t).

Usage:
    python scripts/plot_joint_trajectory.py [method]

method is one of: topp2_ra (default), copp2_socp,
                  topp3_lp_iter1, topp3_lp_iter2,
                  topp3_socp_iter1, topp3_socp_iter2,
                  copp3_socp_iter1, copp3_socp_iter2

Formulas:
    q(t)   = q(s(t))                           position
    q'(t)  = q'(s(t)) · ṡ(t)                   velocity      ṡ = √a(s)
    q''(t) = q''(s(t)) · a(s(t)) + q'(s(t)) · b(t)   acceleration
where b(t) = s̈(t) is derived from a(s(t)) via: b = da/dt / (2ṡ)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

method = sys.argv[1] if len(sys.argv) > 1 else "copp3_socp_iter2"

# ---------- load data ----------
derivs  = pd.read_csv("output/data/path_derivs.csv")
profile = pd.read_csv(f"output/data/{method}.csv")
traj    = pd.read_csv(f"output/data/{method}_traj.csv")

s_grid = derivs["s"].values
DIM = sum(1 for c in derivs.columns if c.startswith("q") and not c.startswith("dq"))

q_grid   = np.column_stack([derivs[f"q{i}"].values  for i in range(DIM)])   # (n, DIM)
dq_grid  = np.column_stack([derivs[f"dq{i}"].values for i in range(DIM)])   # (n, DIM)
ddq_grid = np.column_stack([derivs[f"ddq{i}"].values for i in range(DIM)])  # (n, DIM)

s_traj = traj["s_t"].values
t      = traj["t"].values
dt     = t[1] - t[0] if len(t) > 1 else 1e-3
t_final = t[-1]

a_grid = profile["a"].values   # a(s) = ṡ² on s-grid

# ---------- reconstruct kinematics ----------
# a(s(t)) and ṡ(t)
a_t  = np.interp(s_traj, s_grid, a_grid)
ds_t = np.sqrt(np.maximum(a_t, 0.0))          # ṡ(t) = √a(s(t))

# b(t) = s̈(t): from 3rd-order profile if available, else from da/dt
if "b" in profile.columns:
    b_grid = profile["b"].values
    b_t = np.interp(s_traj, s_grid, b_grid)
else:
    da_dt = np.gradient(a_t, t)
    b_t   = da_dt / (2.0 * np.where(ds_t > 1e-9, ds_t, 1e-9))

# Joint-space kinematics (NT × DIM)
p_t   = np.column_stack([np.interp(s_traj, s_grid, q_grid[:, i])   for i in range(DIM)])
dq_t  = np.column_stack([np.interp(s_traj, s_grid, dq_grid[:, i])  for i in range(DIM)])
ddq_t = np.column_stack([np.interp(s_traj, s_grid, ddq_grid[:, i]) for i in range(DIM)])

v_t   = dq_t  * ds_t[:, None]                          # q̇_i = q'_i · ṡ
acc_t = ddq_t * a_t[:, None] + dq_t * b_t[:, None]    # q̈_i = q''_i·a + q'_i·b

# ---------- plot ----------
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
labels = [f"Joint {i}" for i in range(DIM)]

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle(
    f"Joint-Space Trajectories  [{method}]\n"
    f"Path: 3-axis Lissajous,  t_final = {t_final:.4f} s",
    fontsize=13, fontweight="bold",
)

# Position p(t)
ax = axes[0]
for i in range(DIM):
    ax.plot(t, p_t[:, i], color=colors[i], lw=1.5, label=labels[i])
ax.set_ylabel("q(t)  (rad)")
ax.set_title("Position p(t) = q(s(t))")
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

# Velocity v(t)
ax = axes[1]
for i in range(DIM):
    ax.plot(t, v_t[:, i], color=colors[i], lw=1.5, label=labels[i])
ax.axhline( 1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6, label="±vel limit")
ax.axhline(-1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
ax.set_ylabel("q̇(t)  (rad/s)")
ax.set_title("Velocity v(t) = q′(s)·ṡ")
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

# Acceleration a(t)
ax = axes[2]
for i in range(DIM):
    ax.plot(t, acc_t[:, i], color=colors[i], lw=1.5, label=labels[i])
ax.axhline( 1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6, label="±acc limit")
ax.axhline(-1.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("t  (s)")
ax.set_ylabel("q̈(t)  (rad/s²)")
ax.set_title("Acceleration a(t) = q′′(s)·ṡ² + q′(s)·s̈")
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
out = f"output/plot_joint_{method}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
