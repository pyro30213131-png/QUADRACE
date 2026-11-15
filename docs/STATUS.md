# STATUS

Last updated: 2026-11-15.

## Current milestone

**Milestone 2 (Rust simulator): done.** Milestones 0, 1 and 2 are done: each exit test passed locally, and CI is green on both systems (run #3 for milestone 1, run #4 for milestone 2). Milestone 3 has not been started, per the session protocol; it waits for the owner's go.

Milestone 2 exit test: parity passes, 1-thread and 8-thread results match, a throughput number is recorded.

| Exit-test part | State |
| --- | --- |
| Parity (Rust agrees with `ref_sim.py` to 1e-9) | **Met.** 100 random states, 1 s: worst difference 1.7e-14 m in position and 2.4e-15 in attitude. Also passes with drag on, after one step at 1e-12, and with actions outside [-1, 1]. |
| 1-thread and 8-thread results match | **Met.** Bit-for-bit identical for 1, 2, 8 and default threads, in a Rust test and a Python test. |
| Throughput number recorded | **Met.** About 4.8 million env steps per second at 4,096 drones and 16 threads; 0.85 million on one thread (criterion, release build). Details in `docs/milestone-2.md`. |
| CI green (handoff rule 9) | **Met.** CI run #4 (commit `3efb6d2`, the milestone 2 push) passed every step on `windows-latest` (216 s) and `ubuntu-latest` (96 s). A fresh clone of that commit from GitHub also built and passed every check on this machine (fmt, clippy, 38 Rust tests, 71 Python tests). |

## Owner actions (what only you can do)

1. **Nothing is waiting on you for milestone 2.** Say go when you want milestone 3. I will ask your OK before installing PyTorch (a download well over 100 MB) and before any training run over an hour.
2. **Look at one deviation from the handoff** (the first item under "Decisions and deviations"): the Python class has no `track_cfg` or `seed` yet and its step method is called `step_physics`. Say if you want it otherwise.
3. **Note on the commit identity.** Commits carry your existing git identity (`git config user.name` and `user.email`). The repo is public (GitHub's API answered without a login), so that name and the commit email are visible to everyone. If you want a different email, tell me before more commits pile up: changing the commits already pushed means another history rewrite and force-push.

## Last test results

Measured on 2026-11-15 on this machine (Rust 1.100.0, Python 3.12.10, NumPy 2.5.4):

| Check | Result |
| --- | --- |
| `pytest -q` | 71 passed in 50 s (12 setup, 31 reference simulator, 23 parity, 5 determinism) |
| `cargo test -p quadsim` | 38 passed, 0 failed |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0), including the benchmark |
| `maturin develop --release` | builds and installs (exit code 0) |
| Mutation check on `ref_sim.py` (milestone 1) | 15 deliberate bugs, all caught (one needed a new test, added) |
| Mutation check on the Rust code (milestone 2) | 14 deliberate bugs, all caught (one needed a new test, one needed the drag-enabled parity test) |
| CI runs #1 to #4 on GitHub (milestones 0 to 2) | both `windows-latest` and `ubuntu-latest` passed every step, every time |
| Fresh clone of `3efb6d2` from GitHub, built and tested here | fmt, clippy, 38 Rust tests, `maturin develop --release`, 71 Python tests: all passed |

The individual numbers (parity differences, throughput table, physics checks) are in `docs/milestone-1.md` and `docs/milestone-2.md`. No test was weakened, skipped or deleted.

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

## Things that were unverified and now are

1. The Rust code compiled on the first build with no changes.
2. The maturin layout works: `maturin develop` writes `quadrace.pth` pointing at `python/`, so both packages import from any folder, and `python-packages = ["quadrace"]` puts `quadrace` in the wheel.
3. `cargo test` runs on Windows (the system Python is on PATH).
4. `cargo fmt --check` and `cargo clippy -- -D warnings` pass.

5. **CI works on both systems.** The `actions/checkout@v7` and `actions/setup-python@v7` tags exist, the Ubuntu steps (virtual environment, linking against the runner's Python) and the Windows steps passed, and the code builds and passes its tests on Linux as well as Windows (CI run #1).

## Still unverified

1. **No learning code exists yet,** so nothing about PPO, the hover task, gates or training is verified. The simulator itself is verified three ways: the section 7 physics tests (milestone 1), parity with the NumPy reference, and CI on Windows and Ubuntu.
2. **Throughput on other machines.** The numbers are from one laptop (Intel i5-13450HX) with about 10 percent run-to-run noise. The "16,384 drones is slower on one thread because of the cache" explanation is a guess.
3. **Drone parameters are still the handoff's approximate values**, not from a verified source (handoff section 13). Fine for now; the Swift values are needed for milestone 6.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **PyTorch is not installed.** It is a download well over 100 MB, which CLAUDE.md says to ask about, and it is not needed until milestone 3.
- **Not installed:** TensorBoard, Rerun, matplotlib, ffmpeg (all later milestones).
- **`docs/milestone-0.md`** is final: milestone 0 is done.
- **Milestone 1 choices that the Rust port copied** (details and reasons in `docs/milestone-1.md`): the rotation matrix is built from a normalized quaternion even inside RK4's intermediate stages; RK4 uses the thrust after this substep's motor lag; actions outside [-1, 1] are clipped (the handoff is silent, so this is my choice and is easy to change); the thrust clamp reads `thrust_min` from the config.
- **Deviation from handoff section 5 (needs the owner's eye).** The Python class is `QuadSim(num_envs, drone_cfg, num_threads=None)` with `step_physics`, `get_state`, `set_state`. It has no `track_cfg` or `seed` argument and `step_physics` returns nothing (no observations, rewards, `terminated`, `truncated`, `final_obs`, `info`). Those belong to the environment code of milestones 3 and 4, nothing is random yet, and an unused `seed` would mislead. The full `step()` and the other arguments are added then.
- **Other milestone 2 choices** (reasons in `docs/milestone-2.md`): the drone config is passed as a Python dict (no YAML crate in Rust; the Rust tests and the benchmark carry a labelled literal copy of the numbers); allocation uses a closed form that relies on the mixing-matrix rows being at right angles; each batch owns its own thread pool; the crate's minimum Rust version is now 1.86 (criterion 0.8 needs it) and it builds as both `cdylib` and `rlib`.
- **`rand` is not added yet.** Nothing is random until resets and domain randomization. When randomness arrives, each environment needs its own seeded generator and the determinism tests must be extended.
- **Finding for later:** at hover collective thrust, a full-scale yaw command (4 rad/s) asks two motors for below zero thrust, so yaw saturates in normal flight. Roll and pitch at full scale do not. No change made.

## Next

When you say go, milestone 3 is PPO and the hover task: `ppo.py` hand-written in PyTorch, checks on CartPole and Pendulum, the hover task, checkpointing, per-term reward logging; exit test is the hover evaluation in handoff section 8. It needs PyTorch (a download well over 100 MB: I must ask before installing it), and it brings the first `track_cfg`, `seed` and observation/reward code. I have not started it. Read handoff sections 5, 8 and 9 first.

## Parked ideas (not to be worked on)

`docs/future-ideas.md` holds ideas the owner wants to keep for after the project: a collaborative challenge where friends design drones (within limits enforced in Rust) and rewards, train locally, and are compared by lap time, and later racing several drones at once. **They are not part of any milestone. Do not start, prepare for or build anything from that page until milestones 0 to 8 are finished and the owner asks.**

## Open questions

- Is the `QuadSim` API subset described under "Decisions and deviations" acceptable until milestones 3 and 4?
- Should the clipping of actions outside [-1, 1] stay as it is? (Handoff is silent; I chose clipping.)
