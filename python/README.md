# COPP-Python (Design Stage)

This directory holds the design for a Python trajectory-planning framework built on top of the
algorithms documented in [`docs/code_reading_guide.md`](../docs/code_reading_guide.md).

It currently contains **design documentation only** — no implementation yet.

See [`DESIGN.md`](./DESIGN.md) for:

- Environment / dependency choices (numerics, LP/SOCP solvers, robot kinematics & dynamics libraries)
- Overall package layout (`commands/`, `planning/`, `copp/`, `robot/`, `solver_backend/`, ...)
- The three motion-instruction types: joint move, linear move, circular (arc) move
- The trajectory-planning module design (`planning/`) and its data flow
- The corresponding test-case plan (`tests/unit`, `tests/integration`, `tests/benchmark`)
