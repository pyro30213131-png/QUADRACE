# STATUS

Last updated: 2026-11-15.

## Current milestone

**Milestone 3 (PPO and hover): the exit test is met locally; CI has not run on the final commits.** Milestones 0, 1 and 2 are done (CI green on both systems). Milestone 4 has not been started, per the session protocol; it waits for the owner's go.

Milestone 3 exit test: the hover evaluation (handoff section 8): 200 episodes with a deterministic policy, pass when in at least 95% of them the drone stays within 0.1 m of the target for the final 10 s.

| Part | State |
| --- | --- |
| `ppo.py`, hand-written PPO, checkpointing with observation statistics | **Done.** 47 tests; all 18 deliberate bugs caught. CI run #5 (PPO commits, with PyTorch) green on both systems. |
| Checks on CartPole and Pendulum against Stable-Baselines3 | **Done.** Both implementations learn both problems on all 3 seeds at about the same speed (tables in `docs/milestone-3.md`). |
| Hover task in the Rust simulator (22-value observation, reward, resets, seeded RNG, full `reset()`/`step()`) | **Done locally.** 71 Rust tests and 24 Python tests; all 24 deliberate bugs caught. |
| `QuadVecEnv`, per-term reward logging, `train.py`, `evaluate.py`, `configs/ppo.yaml`, `configs/task_hover.yaml` | **Done locally.** |
| Hover evaluation passes (exit test) | **Met.** Three training seeds (50 million steps, 4 to 5 minutes each) score 200, 199 and 200 of 200 (need 190). An independent check on 500 episodes of a new seed gives 497, 500 and 500. |
| CI green on the final commits (handoff rule 9) | **Not verified.** The hover-task commits are local only. |

## Owner actions (what only you can do)

1. **Push when you are ready** (`git push`, a normal push; I only push when asked), then check the Actions page: https://github.com/pyro30213131-png/QUADRACE/actions. These commits add the Rust hover task and about 100 more Python tests; if either job fails, send me the failing step's log. When both are green, milestone 3 is done and I wait for your go on milestone 4.
2. **Optional: look at the reset choices** in `docs/milestone-3.md` ("Choices worth knowing", item 5): the handoff left some details of the hover reset open (start yaw, how "up to 2 m/s" is sampled), and I chose.
3. **Note on the commit identity.** Commits carry your existing git identity (`git config user.name` and `user.email`). The repo is public (GitHub's API answered without a login), so that name and the commit email are visible to everyone. If you want a different email, tell me before more commits pile up: changing the commits already pushed means another history rewrite and force-push.

## Last test results

Measured on 2026-11-15 on this machine (Rust 1.100.0, Python 3.12.10, NumPy 2.5.4):

| Check | Result |
| --- | --- |
| `pytest -q` | 177 passed in 82 s (12 setup, 31 reference simulator, 23 parity, 5 determinism, 47 PPO, 24 hover environment, 14 training and evaluation, 21 viewer) |
| `cargo test -p quadsim` | 71 passed, 0 failed |
| Hover evaluation, handoff settings, 50 million steps, seeds 1, 2, 3 | 200, 199 and 200 of 200 episodes pass (need 95%); median worst distance in the final 10 s about 0.4 to 1 mm. Independent check (500 episodes, new seed, true positions): 497, 500, 500. |
| Hover evaluation of a hand-written controller (no learning) | 200 of 200; the same controller aimed 0.3 m off: 0 of 200 |
| Whole training loop speed | about 215,000 to 260,000 environment steps per second on the RTX 4050 (target: 100,000) |
| Mutation check on `env.rs` (hover task) | 24 deliberate bugs, all caught (five gaps found on the first pass, five tests added) |
| Mutation check on `ppo.py` | 18 deliberate bugs, all caught (17 on the first pass; the 18th, no gradient clipping, led to a refactor and tests) |
| PPO versus Stable-Baselines3, 3 seeds each | CartPole: both reach 475 on 3 of 3 seeds (ours after 209k steps on average, theirs 195k). Pendulum: both reach -250 on 3 of 3 seeds (103k against 92k); final deterministic returns -194 +- 38 (ours) against -185 +- 30. Details in `docs/milestone-3.md`. |
| One PPO update at the drone's batch size (131,072 samples) | RTX 4050: 284 ms; CPU with 10 threads: 2,586 ms; simulating the same steps: about 27 ms |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0), including the benchmark |
| `maturin develop --release` | builds and installs (exit code 0) |
| Mutation check on `ref_sim.py` (milestone 1) | 15 deliberate bugs, all caught (one needed a new test, added) |
| Mutation check on the Rust code (milestone 2) | 14 deliberate bugs, all caught (one needed a new test, one needed the drag-enabled parity test) |
| CI runs #1 to #5 on GitHub (milestones 0 to 3, up to the PPO commits) | both `windows-latest` and `ubuntu-latest` passed every step, every time (#5: Ubuntu 149 s, Windows 341 s, with PyTorch) |
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
- **Milestone 3, part 2 (local, not pushed):** the hover task in the Rust simulator (`env.rs`, `batch.rs`, `lib.rs`; the `rand` 0.10 and `rand_pcg` 0.10 crates, approved in CLAUDE.md), `configs/task_hover.yaml`, `configs/ppo.yaml`, `QuadVecEnv`, `train.py`, `evaluate.py`, and tests (`test_hover_env.py`, `test_hover_training.py`, more in `test_ppo.py`, and Rust tests). `docs/milestone-3.md` is final. Trained policies live in `runs/` (not committed).
- **Milestone 3, part 1 (pushed; CI green):** `python/quadrace/ppo.py`, `vec_env.py`, `ppo_crosscheck.py`, `tests/test_ppo.py`, `docs/tuning-log.md`. Installed in `.venv` on 2026-11-15 with the owner's go-ahead: PyTorch 2.14.1 with CUDA 12.6 (a 2.42 GB download from `download.pytorch.org`; the driver supports CUDA 12.9, so the CUDA 13 builds would not work), TensorBoard 2.21.0, gymnasium 1.4.0 and Stable-Baselines3 2.9.2 (all approved in CLAUDE.md; the last for the cross-check only). Free disk space on C: is now about 12.5 GB. CI installs CPU PyTorch 2.14.1.

## Things that were unverified and now are

1. The Rust code compiled on the first build with no changes.
2. The maturin layout works: `maturin develop` writes `quadrace.pth` pointing at `python/`, so both packages import from any folder, and `python-packages = ["quadrace"]` puts `quadrace` in the wheel.
3. `cargo test` runs on Windows (the system Python is on PATH).
4. `cargo fmt --check` and `cargo clippy -- -D warnings` pass.

5. **CI works on both systems.** The `actions/checkout@v7` and `actions/setup-python@v7` tags exist, the Ubuntu steps (virtual environment, linking against the runner's Python) and the Windows steps passed, and the code builds and passes its tests on Linux as well as Windows (CI run #1).

## Still unverified

1. **CI on the hover-task commits.** Local only. They add the Rust hover task, the `rand` crates, and about 100 more Python tests (156 in total, about 110 s here). The PPO commits already passed CI (run #5). Failures I can imagine: a slow runner, or a platform difference in the `rand` crates (the generators are specified to be identical across platforms, but nothing has run on Linux).
2. **Hover result on one set of settings.** The exit test is passed on three training seeds with the handoff's settings, but only those; other settings were not explored. A few episodes (about 0.2 to 0.4%) end early (the drone hits the ground); I suspect extreme starts (0.5 m up, falling at 2 m/s, tilted 0.5 rad) but did not check.
3. **PPO versus Stable-Baselines3 is "within noise".** Only 3 seeds each, with a large spread.
4. **Throughput on other machines.** The numbers are from one laptop (Intel i5-13450HX) with about 10 percent run-to-run noise. The "16,384 drones is slower on one thread because of the cache" explanation is a guess.
5. **Drone parameters are still the handoff's approximate values**, not from a verified source (handoff section 13). Fine for now; the Swift values are needed for milestone 6.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **The viewer was pulled forward at the owner's request (2026-11-16).** After the hover exit test passed, the owner asked to watch the policy fly. `python/quadrace/viz.py` records a rollout on its own small simulator and shows it as an interactive 3D Rerun recording and as a matplotlib GIF and PNG (`python -m quadrace.viz --checkpoint <path> --rerun`; `--hardest-of 64` for a stress demo of the most tilted starts). This does **not** replace milestone 4's Rerun work (a live view of the policy during gate-racing training, and a replay) or milestone 8's videos. Details in `docs/milestone-3.md`, "Seeing it fly".
- **New installs for it (owner approved Rerun on 2026-11-16, "go for rerun"; its wheel is 145.6 MB, over the 100 MB that CLAUDE.md says to ask about):** `rerun-sdk` 0.43.2 (plus pyarrow 29 MB, psutil, attrs) and `matplotlib` 3.11.2 (8.9 MB), both approved in CLAUDE.md; `pyproject.toml` has a `view` extra, and CI installs both. The first download crawled at 0.03 to 0.06 MB/s because a Steam download was using most of the connection (not the Wi-Fi: 96% signal, 1.2 Gbps link); after the owner paused Steam it ran at 3.3 MB/s and installed in 66 s.
- **Not installed yet:** ffmpeg (milestone 8, for videos). The GIFs need no ffmpeg.
- **PPO choices** (reasons in `docs/milestone-3.md` and `docs/tuning-log.md`): tanh is inside the networks, not on the output, and the simulator clips actions (the handoff does not decide; tanh squashing is an alternative if actions get stuck at the limits); time-limit cut-offs are bootstrapped from `final_obs` and real endings are not; the value loss is a plain MSE with weight 0.5 and no value clipping, matching Stable-Baselines3; reward normalization exists but is off by default; checkpoints load with `weights_only=True`.
- **`docs/milestone-0.md`** is final: milestone 0 is done.
- **Milestone 1 choices that the Rust port copied** (details and reasons in `docs/milestone-1.md`): the rotation matrix is built from a normalized quaternion even inside RK4's intermediate stages; RK4 uses the thrust after this substep's motor lag; actions outside [-1, 1] are clipped (the handoff is silent, so this is my choice and is easy to change); the thrust clamp reads `thrust_min` from the config.
- **The milestone 2 deviation from handoff section 5 is closed.** `QuadSim(num_envs, drone_cfg, track_cfg=None, seed=0, num_threads=None)` now has `reset(seed)`, `step()` returning `(obs, reward, terminated, truncated, final_obs, info)`, `obs_dim`, `get_state` and `set_state`, as in section 5. Without a `track_cfg` it keeps the physics-only mode (`step_physics`), so the milestone 2 tests are unchanged. Only the hover task exists; the racing task is milestone 4.
- **Hover task reset choices** (handoff left them open; details in `docs/milestone-3.md`): start yaw 0; speed and body rate are a random direction times a size uniform in [0, maximum]; position uniform in the 1.5 m ball around the target, redrawn until at least 0.5 m up.
- **Random numbers:** `rand` 0.10 and `rand_pcg` 0.10 (`Pcg64Mcg`), one generator per environment, seeded from the batch seed and the environment's index, tested to give identical results for 1, 2, 8 and default threads including the random resets.
- **Other milestone 2 choices** (reasons in `docs/milestone-2.md`): the drone config is passed as a Python dict (no YAML crate in Rust; the Rust tests and the benchmark carry a labelled literal copy of the numbers); allocation uses a closed form that relies on the mixing-matrix rows being at right angles; each batch owns its own thread pool; the crate's minimum Rust version is now 1.86 (criterion 0.8 needs it) and it builds as both `cdylib` and `rlib`.
- **Finding for later:** at hover collective thrust, a full-scale yaw command (4 rad/s) asks two motors for below zero thrust, so yaw saturates in normal flight. Roll and pitch at full scale do not. No change made.

## Next

Once CI is green on the milestone 3 commits and you say go, milestone 4 is gates: `track.rs` (gates, the crossing test with unit tests), the 43-value racing observation, the reward (progress, gate bonus, rate and action penalties, crash), episode ends and random-gate resets, the figure-8 track, a Rerun replay and a live view of the current policy, and monitoring metrics; exit test: on the figure-8 with 2.0x gate openings, the drone passes 3 gates in a row in 90% of episodes. It will need Rerun (`rerun-sdk`) and matplotlib, which are approved in CLAUDE.md but not installed (I will check their size first and ask if over 100 MB). New physics or interface changes go into `ref_sim.py`/the parity tests first. I have not started it. Read handoff sections 5 and 8 first.

## Parked ideas (not to be worked on)

`docs/future-ideas.md` holds ideas the owner wants to keep for after the project: a collaborative challenge where friends design drones (within limits enforced in Rust) and rewards, train locally, and are compared by lap time, and later racing several drones at once. **They are not part of any milestone. Do not start, prepare for or build anything from that page until milestones 0 to 8 are finished and the owner asks.**

## Open questions

- Should the clipping of actions outside [-1, 1] stay as it is? (Handoff is silent; I chose clipping.)
