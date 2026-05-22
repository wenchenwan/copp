"""
TOPP2 reachable-set visualization: backward / bidirectional bounds + TOPP2-RA solution.
Run from the project root:  python scripts/plot_reach_set2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv("output/data/reach_set2.csv")

s            = df["s"].values
a_max_back   = df["a_max_back"].values
a_min_back   = df["a_min_back"].values
a_max_bidir  = df["a_max_bidir"].values
a_min_bidir  = df["a_min_bidir"].values
a_ra         = df["a_ra"].values
t_s_ra       = df["t_s_ra"].values
tf_ra        = t_s_ra[-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"TOPP2 Reachable Set  (boundary: a_start = a_end = 0)\n"
    f"TOPP2-RA solution lies within the bidirectional bounds   t_final = {tf_ra:.4f} s",
    fontsize=13, fontweight="bold",
)

# Left: a(s) = ṡ² view
ax = axes[0]
ax.fill_between(s, a_min_bidir, a_max_bidir, alpha=0.18, color="#1f77b4", label="Bidir feasible region")
ax.fill_between(s, a_min_back,  a_max_back,  alpha=0.10, color="#ff7f0e", label="Backward-only region (looser)")
ax.plot(s, a_max_back,  color="#ff7f0e", lw=1.0, linestyle=":",  label="a_max  backward-only")
ax.plot(s, a_max_bidir, color="#1f77b4", lw=1.2, linestyle="--", label="a_max  bidirectional")
ax.plot(s, a_min_bidir, color="#2ca02c", lw=1.2, linestyle="--", label="a_min  bidirectional")
ax.plot(s, a_ra,        color="#d62728", lw=2.0, label="TOPP2-RA  a(s)  [optimal]")
ax.set_xlabel("s  (path parameter)")
ax.set_ylabel("a(s) = ṡ²")
ax.set_title("Reachable Set: a(s) = ṡ²")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# Right: path-speed view √a(s)
ax = axes[1]
ax.fill_between(s, np.sqrt(np.clip(a_min_bidir, 0, None)), np.sqrt(a_max_bidir),
                alpha=0.18, color="#1f77b4", label="Bidir feasible region")
ax.plot(s, np.sqrt(a_max_back),  color="#ff7f0e", lw=1.0, linestyle=":",  label="√a_max  backward-only")
ax.plot(s, np.sqrt(a_max_bidir), color="#1f77b4", lw=1.2, linestyle="--", label="√a_max  bidirectional")
ax.plot(s, np.sqrt(np.clip(a_min_bidir, 0, None)), color="#2ca02c", lw=1.2, linestyle="--", label="√a_min  bidirectional")
ax.plot(s, np.sqrt(a_ra),        color="#d62728", lw=2.0, label="TOPP2-RA  ṡ(s)")
ax.set_xlabel("s  (path parameter)")
ax.set_ylabel("ṡ(s) = √a(s)  (path speed)")
ax.set_title("Reachable Set: Path Speed ṡ(s)")
ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("output", exist_ok=True)
out = "output/plot_reach_set2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
