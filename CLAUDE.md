# quadrace

Batched quadrotor racing simulator in Rust (PyO3 bindings) plus hand-written PPO in PyTorch. A policy learns to fly gate courses (figure-8, then the Swift paper's Split-S) in simulation. State-based only: no camera, no hardware.

Stack: Rust stable (MSVC), PyO3 + rust-numpy + maturin, rayon, criterion; Python 3.11+, PyTorch (CUDA), NumPy, pytest, TensorBoard, Rerun. Pinned (milestone 0, full table in `README.md`): Python 3.12.10, maturin 1.16.0, pyo3 0.30.1 + rust-numpy 0.30.0 (built and tested together), Rust 1.100.0 stable edition 2024, VS Build Tools 2022. On Windows a username with a space breaks the VS C++ installer unless TEMP/TMP point at a space-free folder (see `docs/STATUS.md`).

Full spec: `AGENT_HANDOFF.md` (source of truth; this file is the short version). Read only the sections you need:
section 5 architecture and API, 6 conventions, 7 dynamics and required tests, 8 environment (race and hover), 9 training and evaluation, 10 milestones and stop conditions, 13 unverified values.

## Session protocol

1. Read `docs/STATUS.md` first (current milestone, last test results, open questions). Create it in milestone 0.
2. Read the handoff sections for the current milestone only.
3. Work on that milestone and nothing beyond it.
4. Before ending a session, update `docs/STATUS.md`.
5. When a milestone's exit test passes: write `docs/milestone-N.md`, report results, and stop. Do not start the next milestone unprompted.
6. Commit each milestone's work as it is finished: small commits, one task each, and the milestone's docs in the last one. Commit only; push only when the owner asks. Commit messages carry no `Co-Authored-By` or other attribution lines (owner's instruction).

## Owner context

- Native Windows, PowerShell, RTX 4050 (6 GB). Use PowerShell syntax, not bash.
- The owner has not written Rust and must be able to explain this code in interviews. Prefer the plainest code that works, and explain choices in the milestone docs.

## Commands

These are the intended commands. If milestone 0 lands them differently, correct this section.

```
.venv\Scripts\Activate.ps1                         # activate the environment
maturin develop --release                          # rebuild the Rust extension
cargo test -p quadsim                              # all Rust tests
cargo test -p quadsim <test_name>                  # one Rust test
pytest -q                                          # all Python tests
pytest tests/test_parity.py -k <pattern> -q        # one Python test
cargo clippy --all-targets -- -D warnings          # Rust lint
cargo fmt --check                                  # Rust format check
cargo bench -p quadsim                             # simulator benchmark
python -m quadrace.train --config configs/ppo.yaml # train (ask first if over 1 hour)
python -m quadrace.evaluate --checkpoint <path>    # 200-episode evaluation
python -m quadrace.ppo_crosscheck --env CartPole-v1 --seeds 1 2 3   # PPO vs Stable-Baselines3 (also Pendulum-v1)
python -m quadrace.viz --checkpoint <path> --rerun # watch a hover policy fly: GIF + PNG, and the Rerun viewer
tensorboard --logdir runs                          # training metrics
```

Run the single-test form while iterating. Run the full suites before reporting a task done.

## Key folders

- `crates/quadsim/src/`: `math.rs` (Vec3, Quat), `params.rs` (drone constants), `dynamics.rs` (rigid body, RK4), `motors.rs`, `controller.rs` (rate loop), `track.rs` (gates, crossing test), `env.rs` (obs, reward, resets), `batch.rs` (rayon), `lib.rs` (PyO3 API).
- `python/quadrace/`: `ref_sim.py` (NumPy reference, the oracle for parity), `vec_env.py`, `ppo.py`, `train.py`, `evaluate.py`, `viz.py`, `replay_progress.py`, `baseline.py`.
- `configs/`: every physical constant and hyperparameter, as YAML.
- `tests/`: parity and environment tests. `docs/`: status, milestone pages, tuning log.
- `data/external/`: owner-supplied files. Never committed.

## Conventions (do not change without asking)

- SI units, radians. World frame z up, gravity `(0, 0, -9.81)`. Body frame x forward, y left, z up.
- Quaternion: scalar first `(w, x, y, z)`, Hamilton product, body to world. Renormalize every physics substep.
- Angular velocity is always in the body frame: `q_dot = 0.5 * q (x) (0, omega)`.
- State is 17 values: position, quaternion, velocity, body rates, 4 motor thrusts.
- Physics 500 Hz, policy 50 Hz. Physics in f64; observations cast to f32 at the Python boundary.
- Motor numbering and signs: the table in handoff section 6.

## Code style

- Rust: no `unsafe`; no macros beyond `derive` and PyO3 attributes; no clever generics. Hand-written `Vec3` and `Quat`, no nalgebra.
- Name every vector with its frame: `vel_w`, `omega_b`. Comment each physics equation with its name and frames.
- No heap allocation inside `step()`. Output arrays are preallocated.
- Validate shapes and dtypes at the PyO3 boundary and return a Python error. Do not let a Rust panic cross into Python.
- `debug_assert!` that state is finite and the quaternion is unit length.
- Python: type hints, explicit seeds, no module-level state. `ref_sim.py` uses float64 throughout.
- One behaviour per test, named after the spec item it checks.

## Guardrails

- Never weaken, skip or delete a test to make it pass. If an exit test cannot be met, stop and report.
- Never invent a physical value. Handoff section 13 lists what is unverified and where to verify it.
- Do not add a crate or pip package without asking. Approved: pyo3, rust-numpy, rayon, rand plus one seedable generator crate, serde plus a maintained YAML crate, criterion; numpy, torch, pyyaml, pytest, tensorboard, rerun-sdk, matplotlib, and for the PPO cross-check only, gymnasium and stable-baselines3.
- Ask before: a training run over 1 hour, a download over 100 MB, an install needing admin rights, choosing a licence, making the repo public.
- Do not edit `AGENT_HANDOFF.md` silently. Propose the change; if approved, edit it and bump its version.
- Do not build anything on the handoff's out-of-scope list.
- Do not commit `runs/`, checkpoints, videos, `.rrd` recordings or `data/external/`.
- Tuning changes (reward terms, PPO settings, gains) go in `docs/tuning-log.md` with the reason.
- Keep this file under 200 lines. Put detail in the handoff or `docs/`.

## Gotchas

- After any Rust edit, run `maturin develop --release` before `pytest`, or Python tests run the old binary.
- Never quote a benchmark from a debug build.
- SciPy's `Rotation` is scalar-last; this repo is scalar-first. Reorder when comparing.
- `q` and `-q` are the same rotation. Compare quaternions up to sign.
- Results must not depend on thread count: one seeded generator per environment.
- PPO: bootstrap the value from `final_obs` on truncation, not on termination.
- Save observation-normalization statistics with every checkpoint.
- Never render the training environments. The live view uses a separate 1 to 4 environment simulator.
- Reward rising while lap completion stays flat means an exploit. Stop and report.
- The Split-S milestone is blocked until `data/external/track.yaml` exists. Do not guess gate orientations or sizes.
