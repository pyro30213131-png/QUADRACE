# quadrace

A batched quadrotor racing simulator written in Rust (exposed to Python through PyO3), plus a hand-written PPO in PyTorch that learns to fly gate courses. State-based only: no camera, no hardware.

**Status: milestones 0 (setup) and 1 (the NumPy reference simulator) are done. Milestone 2 (the Rust simulator) is also done, with CI green on Windows and Ubuntu.** Milestone 3 (PPO and hover) passes its exit test locally, with its CI run pending a push: the hand-written PPO learns CartPole and Pendulum about as well as Stable-Baselines3 (3 seeds each), and a policy it trains to hover on the Rust simulator passes the hover evaluation on all 3 training seeds (199 to 200 of 200 episodes within 0.1 m of the target for the final 10 s; 95% is required), at about 215,000 to 260,000 environment steps per second on the RTX 4050 ([docs/milestone-3.md](docs/milestone-3.md)). Gate racing, the project's headline task, has not been trained yet. Measured so far, on one laptop (Intel i5-13450HX, release build): the Rust simulator agrees with the NumPy reference to about 1e-14 after one second of flight on 100 random states, and it steps about 4.8 million drone-steps per second on 16 threads (4,096 drones; 0.85 million on one thread). Details: [docs/milestone-2.md](docs/milestone-2.md). The full spec is in [AGENT_HANDOFF.md](AGENT_HANDOFF.md); current progress is in [docs/STATUS.md](docs/STATUS.md).

## Setup (native Windows, PowerShell)

1. Install the Visual Studio Build Tools with the "Desktop development with C++" workload, then install Rust with rustup (MSVC toolchain). Details: [docs/STATUS.md](docs/STATUS.md).
   - **If your Windows username contains a space**, run the Build Tools installer with `TEMP` and `TMP` set to a folder without spaces (for example `C:\vs-temp`). On the development machine the default install failed with error 1622 in the Visual C++ redistributable step, and the same install succeeded with a space-free temp folder.
2. Create the Python environment and build:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install maturin pytest numpy pyyaml
maturin develop --release
```

3. For training (milestone 3 on), install PyTorch, TensorBoard and gymnasium. Use `nvidia-smi` to see which CUDA version your NVIDIA driver supports. On the development machine (driver supporting CUDA 12.9) this was:

```powershell
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # about 2.4 GB
pip install tensorboard gymnasium
pip install stable-baselines3     # only for cross-checking ppo.py
```

   The CUDA 13 builds (`cu130`) need a newer driver. Without an NVIDIA GPU use `--index-url https://download.pytorch.org/whl/cpu` instead.

4. Check it works:

```powershell
cargo test -p quadsim
pytest -q
```

## Layout

```
Cargo.toml                 Cargo workspace
pyproject.toml             maturin build configuration
crates/quadsim/            the Rust simulator (physics, batching, PyO3 API, benchmark)
python/quadsim/            thin Python package that re-exports the compiled module
python/quadrace/           ref_sim.py (NumPy reference simulator) and, later, the learning code (PPO, training, evaluation)
configs/                   every physical constant and hyperparameter, as YAML
tests/                     Python tests
docs/                      status, milestone pages, tuning log
```

### Why two Python packages

The handoff asks for the Rust module to import as `quadsim` and the Python package as `quadrace`.

- maturin builds the compiled file as `quadsim._quadsim` (the leading underscore marks it as an implementation detail).
- `python/quadsim/__init__.py` re-exports it, so users write `import quadsim`.
- `python/quadrace/` is a separate, plain-Python package in the same project, listed under `python-packages` in `pyproject.toml`.

maturin requires the Python directory to exist when `python-source` is set and `module-name` is a single name, so a dotted `module-name` inside a real `quadsim` package is the layout maturin documents.

## Conventions (fixed, from handoff section 6)

- SI units, angles in radians.
- World frame: z up, gravity `(0, 0, -9.81)`. Body frame: x forward, y left, z up along the thrust axis.
- Quaternion: scalar first `(w, x, y, z)`, Hamilton product, rotates body to world; renormalized after every physics substep. `q` and `-q` are the same rotation.
- Angular velocity is always in the body frame.
- State is 17 values: position (3, world), quaternion (4), velocity (3, world), body rates (3, body), motor thrusts (4).
- Physics runs at 500 Hz, the policy at 50 Hz. Physics is `f64`; observations are cast to `f32` at the Python boundary.

## Pinned versions

| Tool | Version | How it was fixed |
| --- | --- | --- |
| Python | 3.12.10 | installed on this machine; the project requires 3.11 or newer |
| maturin | 1.16.0 | installed in `.venv`; the build requires `>=1.9.4,<2` |
| numpy | 2.5.4 | installed in `.venv` |
| pytest | 9.2.0 | installed in `.venv` |
| PyYAML | 6.0.3 | installed in `.venv` |
| Rust crates | `pyo3` 0.30.1, `numpy` 0.30.0, `ndarray` 0.17.2, `rayon` 1.12, `rand` 0.10 and `rand_pcg` 0.10, `criterion` 0.8 (benchmarks only) | resolved in `Cargo.lock`; built and tested together on 2026-11-15 |
| Rust | 1.100.0 (stable, released 2026-11-09), edition 2024 | installed with rustup; 1.86 is the stated minimum (1.85 for PyO3 and rust-numpy, 1.86 for criterion) |
| C++ build tools | Visual Studio Build Tools 2022 17.14, MSVC 14.44.35207, Windows SDK 10.0.26100.0 | installed with winget |
| GitHub Actions | `checkout` v7, `setup-python` v7 | latest releases when written; run on GitHub on 2026-11-14 and passed |
| PyTorch | 2.14.1 (`+cu126` on the development machine, `+cpu` in CI) | the newest build whose CUDA runtime the owner's driver supports; installed 2026-11-15 from `download.pytorch.org` |
| Training and check packages | tensorboard 2.21.0, gymnasium 1.4.0, stable-baselines3 2.9.2 | installed in `.venv` on 2026-11-15 (stable-baselines3 for the PPO cross-check only) |
