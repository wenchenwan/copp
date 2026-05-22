# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build
cargo build --release

# Run a minimal solver example (console output only)
cargo run --example topp2_ra
cargo run --example copp2_socp
cargo run --example topp3_lp
cargo run --example topp3_socp
cargo run --example copp3_socp
cargo run --example reach_set2

# Run benchmark tests (slow, ignored by default)
cargo test --release -- --ignored --test-threads=1

# Generate and open API docs (includes MathJax equations)
cargo doc --no-deps
```

## Demo workflow (Rust → CSV → Python plots)

All six solvers share the same 3-axis Lissajous path and robot constraints. Run once to produce all CSV data, then use individual Python scripts to plot.

### Step 1 — export solver data

```bash
cargo run --example save_all --release
```

Writes to `output/data/`:

| File | Contents |
|------|----------|
| `path_derivs.csv` | `s, q0–q2, dq0–dq2, ddq0–ddq2` — path geometry for joint-space recovery |
| `topp2_ra.csv` / `_traj.csv` | `s, a, t_s` profile + `t, s_t` uniform trajectory |
| `copp2_socp.csv` / `_traj.csv` | same structure (time-optimal objective) |
| `reach_set2.csv` | `s, a_max_back, a_min_back, a_max_bidir, a_min_bidir, a_ra, t_s_ra` |
| `topp3_lp_iter{1,2}.csv` / `_traj.csv` | `s, a, b, t_s` profile (b = s̈) + trajectory |
| `topp3_socp_iter{1,2}.csv` / `_traj.csv` | same structure |
| `copp3_socp_iter{1,2}.csv` / `_traj.csv` | same structure (time + thermal-energy objective) |

### Step 2 — per-method result plots

```bash
python scripts/plot_topp2_ra.py       # a(s), ṡ(s), s(t), t(s)  → output/plot_topp2_ra.png
python scripts/plot_copp2_socp.py     # comparison with TOPP2-RA → output/plot_copp2_socp.png
python scripts/plot_reach_set2.py     # reachable-set bounds      → output/plot_reach_set2.png
python scripts/plot_topp3_lp.py       # iter 1 vs iter 2          → output/plot_topp3_lp.png
python scripts/plot_topp3_socp.py     # iter 1 vs iter 2          → output/plot_topp3_socp.png
python scripts/plot_copp3_socp.py     # iter 1 vs iter 2          → output/plot_copp3_socp.png
```

### Step 3 — joint-space trajectories p(t), v(t), a(t)

```bash
# Default: copp3_socp_iter2
python scripts/plot_joint_trajectory.py

# Choose any method explicitly
python scripts/plot_joint_trajectory.py topp2_ra
python scripts/plot_joint_trajectory.py copp2_socp
python scripts/plot_joint_trajectory.py topp3_lp_iter2
python scripts/plot_joint_trajectory.py topp3_socp_iter2
python scripts/plot_joint_trajectory.py copp3_socp_iter2
```

Output saved to `output/plot_joint_{method}.png`.

**Reconstruction formulas** (implemented in `scripts/plot_joint_trajectory.py`):

| Signal | Formula |
|--------|---------|
| Position `q(t)` | `q(s(t))` — interpolated from path grid |
| Velocity `q̇(t)` | `q'(s(t)) · ṡ(t)`,  where `ṡ = √a(s(t))` |
| Acceleration `q̈(t)` | `q''(s(t)) · a(s(t)) + q'(s(t)) · b(t)` |

For 2nd-order methods `b = s̈` is estimated as `da/dt / (2ṡ)`; for 3rd-order methods `b(s)` is read directly from the profile CSV.

## Architecture

This library solves **Optimal Path Parameterization (OPP)**: given a geometric path `q(s)`, find a time law `s(t)` that satisfies robot constraints while optimizing a convex objective.

### Solver taxonomy

| Order | Time-optimal | General convex objective |
|-------|-------------|--------------------------|
| 2nd (velocity/acceleration/torque) | TOPP2-RA | COPP2-SOCP |
| 3rd (+ jerk) | TOPP3-LP, TOPP3-SOCP | COPP3-SOCP |

### Public API surface

All user-facing entry points are in `src/lib.rs`. The implementation under `src/copp/` is `pub(crate)` only.

- **`copp::solver::{topp2_ra, copp2_socp, topp3_lp, topp3_socp, copp3_socp, reach_set2}`** — solver submodules; each re-exports the relevant builder types and solver functions.
- **`copp::prelude::*`** — convenience re-export of all common types; recommended for application code.
- **`copp::robot::Robot`** — high-level constraint ingestion (call `with_axial_velocity`, `with_axial_acceleration`, `with_axial_jerk`, etc.).
- **`copp::constraints::Constraints`** — lower-level constraint container; use when `Robot` abstractions are insufficient.
- **`copp::path::{Path, Jet3}`** — path definition with automatic differentiation. `Path::from_parametric` accepts a closure over `Jet3`; `Path::from_waypoints` fits a spline.

### Solver output conventions

- **2nd-order solvers** return `Vec<f64>` representing the `a(s) = ṡ²` profile at each grid point.
- **3rd-order solvers** return `(Vec<f64>, Vec<f64>, (usize, usize))` — `(a, b, num_stationary)`, where `b = s̈` and `num_stationary` counts stationary boundary segments.
- Post-processing helpers `s_to_t_topp2`, `t_to_s_topp2` (2nd-order) and `s_to_t_topp3`, `t_to_s_topp3` (3rd-order) convert solver output to timing maps and trajectory samples.

### 3rd-order linearization dependency

TOPP3 and COPP3 problem builders require a linearization point. The standard pattern is:

```rust
// 1. Run TOPP2-RA to get a(s) as a linearization reference
let a_topp2_ra = topp2_ra(&topp2_problem, &options)?;

// 2. Build 3rd-order problem with linearization
let problem = Topp3ProblemBuilder::new(&mut robot, idx_s_start, &a_topp2_ra, a_boundary, b_boundary)
    .build_with_linearization()?;
```

Note that `Topp3ProblemBuilder` and `Copp3ProblemBuilder` take `&mut robot` (mutation required for linearization), unlike 2nd-order builders which take `&robot`.

### Objective specification (COPP only)

COPP objectives are built as a slice of `CoppObjective` enum variants and composed linearly:

```rust
let objectives = [
    CoppObjective::Time(1.0),
    CoppObjective::ThermalEnergy(0.1, &weights),
];
```

Available variants: `Time`, `ThermalEnergy`, `TotalVariationTorque`, `Linear`.

### Clarabel solver options

The `ClarabelOptionsBuilder` controls tolerance for near-solved states. For benchmark use, enable:

```rust
ClarabelOptionsBuilder::new()
    .allow_almost_solved(true)
    .allow_max_time(true)
    .allow_max_iterations(true)
    .allow_insufficient_progress(true)
    .build()?
```

### Internal module layout

```
src/copp/
  copp2/          # 2nd-order: DP reachability (dp2/) and SOCP optimization (opt2/)
  copp3/          # 3rd-order: LP and SOCP optimization (opt3/)
  clarabel_backend.rs  # Clarabel wrapper types and solution conversion
  constraints.rs  # Station-indexed circular constraint storage
  objectives.rs   # CoppObjective enum
  general.rs      # InterpolationMode and shared helpers
src/path/         # Path, Jet3 AD type, spline fitting
src/robot/        # Robot constraint ingestion; RobotBasic / RobotTorque traits
src/math/         # Numerical utilities (LP helpers, general math)
src/diag/         # CoppError, Verbosity, diagnostics
```
