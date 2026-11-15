# STATUS

Last updated: 2026-11-15.

## Current milestone

**Milestone 3 (PPO and hover): in progress. Part 1 is done locally, part 2 is not started.** Milestones 0, 1 and 2 are done (CI green on both systems: runs #1 to #4). Part 1 of milestone 3 is the hand-written PPO and its checks on CartPole and Pendulum, requested by the owner on 2026-11-15; it is committed locally but not pushed, so CI has not run on it.

Milestone 3 exit test: the hover evaluation (handoff section 8): 200 episodes with a deterministic policy, pass when in at least 95% of them the drone stays within 0.1 m of the target for the final 10 s.

| Part | State |
| --- | --- |
| `ppo.py`, hand-written PPO (CleanRL style), checkpointing with observation statistics | **Done locally.** 44 tests; all 18 deliberate bugs caught. |
| Checks on CartPole and Pendulum against Stable-Baselines3 | **Done locally.** Both implementations learn both problems on all 3 seeds at about the same speed (tables in `docs/milestone-3.md`). |
| Hover task in the Rust simulator (22-value observation, reward, resets, seeded RNG, full `reset()`/`step()`) | Not started. |
| Python wrapper making `QuadSim` a `VecEnv`; per-term reward logging in TensorBoard | Not started. |
| Hover evaluation passes (exit test) | Not started. |
| CI green on the PPO commits | **Not verified.** Local only. CI now installs CPU PyTorch, TensorBoard and gymnasium. |

## Owner actions (what only you can do)

1. **Push when you are ready** (`git push`, a normal push; I only push when asked), then check the Actions page: https://github.com/pyro30213131-png/QUADRACE/actions. This is the first CI run that installs PyTorch, so it may be slower; if either job fails, send me the failing step's log.
2. **Look at one deviation from the handoff** (the first item under "Decisions and deviations"): the Python class has no `track_cfg` or `seed` yet and its step method is called `step_physics`. Part 2 of milestone 3 adds them with the full `step()`. Say if you want it otherwise.
3. **Note on the commit identity.** Commits carry your existing git identity (`git config user.name` and `user.email`). The repo is public (GitHub's API answered without a login), so that name and the commit email are visible to everyone. If you want a different email, tell me before more commits pile up: changing the commits already pushed means another history rewrite and force-push.

## Last test results

Measured on 2026-11-15 on this machine (Rust 1.100.0, Python 3.12.10, NumPy 2.5.4):

| Check | Result |
| --- | --- |
| `pytest -q` | 115 passed in 101 s (12 setup, 31 reference simulator, 23 parity, 5 determinism, 44 PPO) |
| `cargo test -p quadsim` | 38 passed, 0 failed |
| Mutation check on `ppo.py` (milestone 3) | 18 deliberate bugs, all caught (17 on the first pass; the 18th, no gradient clipping, led to a refactor and tests) |
| PPO versus Stable-Baselines3, 3 seeds each | CartPole: both reach 475 on 3 of 3 seeds (ours after 209k steps on average, theirs 195k). Pendulum: both reach -250 on 3 of 3 seeds (103k against 92k); final deterministic returns -194 +- 38 (ours) against -185 +- 30. Details in `docs/milestone-3.md`. |
| One PPO update at the drone's batch size (131,072 samples) | RTX 4050: 284 ms; CPU with 10 threads: 2,586 ms; simulating the same steps: about 27 ms |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0), including the benchmark |
| `maturin develop --release` | builds and installs (exit code 0) |
| Mutation check on `ref_sim.py` (milestone 1) | 15 deliberate bugs, all caught (one needed a new test, added) |
| Mutation check on the Rust code (milestone 2) | 14 deliberate bugs, all caught (one needed a new test, one needed the drag-enabled parity test) |
| CI runs #1 to #4 on GitHub (milestones 0 to 2) | both `windows-latest` and `ubuntu-latest` passed every step, every time |
| Fresh clone of `3efb6d2` from GitHub, built and tested here | fmt, clippy, 38 Rust tests, `maturin develop --release`, 71 Python tests: all passed |

The individual numbers (parity differences, throughput table, physics checks, PPO cross-check tables) are in `docs/milestone-1.md`, `docs/milestone-2.md` and `docs/milestone-3.md`. No test was weakened, skipped or deleted.

## Toolchain log (2026-11-14)

- **Rust: installed** with rustup: `stable-x86_64-pc-windows-msvc`, rustc and cargo 1.100.0. A shell opened before the install needs `%USERPROFILE%\.cargo\bin` on PATH or a restart.
- **Visual Studio Community 2026: installed by the rustup flow, failed, and was removed.** The C++ workload was marked incomplete: one package, `Microsoft.VisualCpp.Redist.14.Latest` (x86), failed with Windows error 1622 ("Error opening installation log file"), and the Windows SDK was never installed. I uninstalled it with Visual Studio's own uninstaller (exit code 0).
- **Cause of the failure, most likely: the space in the Windows username.** The package's log shows its path as `C:\Users\Anish Krishna\AppData\...`, and `VCRedistInstall.ps1` (line 254) builds `-ArgumentList "/q /norestart /log $LogFile"` without quotes around the path. The same install then succeeded with `TEMP` and `TMP` set to `C:\vs-temp`, which fits that explanation. It is evidence, not proof.
- **Visual Studio Build Tools 2022: installed** with `winget` (`Microsoft.VisualStudio.2022.BuildTools`, workload `Microsoft.VisualStudio.Workload.VCTools`, with recommended components), launched elevated with `TEMP` and `TMP` set to `C:\vs-temp`. Checked afterwards: `vswhere` reports the instance complete (version 17.14.37906.3) with the MSVC x64/x86 tools, `link.exe` is present (MSVC 14.44.35207), the Windows SDK is 10.0.26100.0, and none of the installer's error logs contains anything. winget showed no agreements to accept, so none were accepted on your behalf.
- **Leftover:** `C:\vs-temp` holds the installer's logs (about 350 files). It is safe to delete when you no longer want them. Free space on C: is now 12.5 GB.

## What is done

- **Git:** repo initialised on branch `main` with five small commits for the milestone 0 scaffold. Convention from now on (also in `CLAUDE.md`): commit each milestone's work as it is finished, small commits one task each, commit only and push only when asked. Commit messages carry no `Co-Authored-By` or other attribution lines (owner's instruction).
- Cargo workspace and the `quadsim` crate with a hello-world `add_arrays` (PyO3 0.30.1 + rust-numpy 0.30.0), including two Rust unit tests.
- maturin configuration in `pyproject.toml`; Python packages `quadsim` (re-export) and `quadrace` (empty for now).
- `configs/drone.yaml` with the handoff section 7 values.
- `tests/test_setup.py` (12 tests), CI workflow `.github/workflows/ci.yml`, `.gitattributes`, `.gitignore`, `README.md`, `docs/milestone-0.md`.
- Python environment: `.venv` (Python 3.12.10) with maturin 1.16.0, numpy 2.5.4, pytest 9.2.0, PyYAML 6.0.3.
- Versions checked against current sources: crates.io, the PyO3 getting-started guide, PyPI, GitHub releases. Full table in `README.md`.
- **Milestone 1:** `python/quadrace/ref_sim.py` (one quadrotor in NumPy float64: rigid body with RK4, rate controller, mixing and allocation, motor lag, action mapping, optional linear drag), `tests/test_ref_sim.py` (31 test cases) and `docs/milestone-1.md`. Pushed; CI green.
- **Milestone 2:** the Rust simulator in `crates/quadsim/src/` (`math.rs`, `params.rs`, `motors.rs`, `controller.rs`, `dynamics.rs`, `env.rs`, `batch.rs`, and the PyO3 `QuadSim` class in `lib.rs`), `tests/test_parity.py`, `tests/test_determinism.py`, 38 Rust unit tests, `benches/step.rs` (criterion) and `docs/milestone-2.md`. Dependencies added: `rayon` 1.12 and, for benchmarks only, `criterion` 0.8 (both approved in CLAUDE.md). Pushed; CI green.
- **Milestone 3, part 1 (local, not pushed):** `python/quadrace/ppo.py`, `vec_env.py`, `ppo_crosscheck.py`, `tests/test_ppo.py` (44 tests), `docs/tuning-log.md`, and an interim `docs/milestone-3.md`. Installed in `.venv` on 2026-11-15 with the owner's go-ahead: PyTorch 2.14.1 with CUDA 12.6 (a 2.42 GB download from `download.pytorch.org`; the driver supports CUDA 12.9, so the CUDA 13 builds would not work), TensorBoard 2.21.0, gymnasium 1.4.0 and Stable-Baselines3 2.9.2 (all approved in CLAUDE.md; the last for the cross-check only). Free disk space on C: is now about 12.5 GB. CI installs CPU PyTorch 2.14.1.

## Things that were unverified and now are

1. The Rust code compiled on the first build with no changes.
2. The maturin layout works: `maturin develop` writes `quadrace.pth` pointing at `python/`, so both packages import from any folder, and `python-packages = ["quadrace"]` puts `quadrace` in the wheel.
3. `cargo test` runs on Windows (the system Python is on PATH).
4. `cargo fmt --check` and `cargo clippy -- -D warnings` pass.

5. **CI works on both systems.** The `actions/checkout@v7` and `actions/setup-python@v7` tags exist, the Ubuntu steps (virtual environment, linking against the runner's Python) and the Windows steps passed, and the code builds and passes its tests on Linux as well as Windows (CI run #1).

## Still unverified

1. **PPO is checked on CartPole and Pendulum only, with 3 seeds each.** The seed-to-seed spread is large, so "equal to Stable-Baselines3" means "within noise", not identical. PPO has never trained the drone: the hover task does not exist yet. The simulator itself is verified three ways: the section 7 physics tests (milestone 1), parity with the NumPy reference, and CI on Windows and Ubuntu.
2. **CI on the PPO commits.** Local only. It is the first CI run with PyTorch (CPU build), TensorBoard and gymnasium, and with the 44 PPO tests (about 45 s here, one of them a 20 s CartPole training run). A failure I can imagine: a slow runner, or CartPole learning slightly differently on Linux (the test has a wide margin).
3. **The end-to-end training speed is not measured.** One PPO update on the GPU takes 284 ms, but the real loop also needs the per-step policy forward pass and CPU-to-GPU copies.
4. **Throughput on other machines.** The numbers are from one laptop (Intel i5-13450HX) with about 10 percent run-to-run noise. The "16,384 drones is slower on one thread because of the cache" explanation is a guess.
5. **Drone parameters are still the handoff's approximate values**, not from a verified source (handoff section 13). Fine for now; the Swift values are needed for milestone 6.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **Not installed yet:** Rerun, matplotlib, ffmpeg (milestones 4 and 8).
- **PPO choices** (reasons in `docs/milestone-3.md` and `docs/tuning-log.md`): tanh is inside the networks, not on the output, and the simulator clips actions (the handoff does not decide; tanh squashing is an alternative if actions get stuck at the limits); time-limit cut-offs are bootstrapped from `final_obs` and real endings are not; the value loss is a plain MSE with weight 0.5 and no value clipping, matching Stable-Baselines3; reward normalization exists but is off by default; checkpoints load with `weights_only=True`.
- **`docs/milestone-0.md`** is final: milestone 0 is done.
- **Milestone 1 choices that the Rust port copied** (details and reasons in `docs/milestone-1.md`): the rotation matrix is built from a normalized quaternion even inside RK4's intermediate stages; RK4 uses the thrust after this substep's motor lag; actions outside [-1, 1] are clipped (the handoff is silent, so this is my choice and is easy to change); the thrust clamp reads `thrust_min` from the config.
- **Deviation from handoff section 5 (needs the owner's eye).** The Python class is `QuadSim(num_envs, drone_cfg, num_threads=None)` with `step_physics`, `get_state`, `set_state`. It has no `track_cfg` or `seed` argument and `step_physics` returns nothing (no observations, rewards, `terminated`, `truncated`, `final_obs`, `info`). Those belong to the environment code of milestones 3 and 4, nothing is random yet, and an unused `seed` would mislead. The full `step()` and the other arguments are added then.
- **Other milestone 2 choices** (reasons in `docs/milestone-2.md`): the drone config is passed as a Python dict (no YAML crate in Rust; the Rust tests and the benchmark carry a labelled literal copy of the numbers); allocation uses a closed form that relies on the mixing-matrix rows being at right angles; each batch owns its own thread pool; the crate's minimum Rust version is now 1.86 (criterion 0.8 needs it) and it builds as both `cdylib` and `rlib`.
- **`rand` is not added yet.** Nothing is random until resets and domain randomization. When randomness arrives, each environment needs its own seeded generator and the determinism tests must be extended.
- **Finding for later:** at hover collective thrust, a full-scale yaw command (4 rad/s) asks two motors for below zero thrust, so yaw saturates in normal flight. Roll and pitch at full scale do not. No change made.

## Next

Part 2 of milestone 3: the hover task in the Rust simulator (22-value observation, 5-term reward, termination and truncation, random resets with one seeded generator per environment using the approved `rand` crate, and the full `reset()`/`step()` of handoff section 5), a Python wrapper that makes `QuadSim` a `VecEnv`, per-term reward logging in TensorBoard, then the first training runs and the hover evaluation (200 deterministic episodes, pass at 95%). Hover training run times are unmeasured; I will ask before any run over an hour. Not started; waiting for the owner's go. Read handoff sections 5 and 8 first.

## Parked ideas (not to be worked on)

`docs/future-ideas.md` holds ideas the owner wants to keep for after the project: a collaborative challenge where friends design drones (within limits enforced in Rust) and rewards, train locally, and are compared by lap time, and later racing several drones at once. **They are not part of any milestone. Do not start, prepare for or build anything from that page until milestones 0 to 8 are finished and the owner asks.**

## Open questions

- Is the `QuadSim` API subset described under "Decisions and deviations" acceptable until milestones 3 and 4?
- Should the clipping of actions outside [-1, 1] stay as it is? (Handoff is silent; I chose clipping.)
