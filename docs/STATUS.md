# STATUS

Last updated: 2026-11-15.

## Current milestone

**Milestone 1 (NumPy reference simulator): the exit test is met locally; CI has not run on it.** Milestone 0 is done (CI green on both systems, runs #1 and #2). Milestone 2 has not been started, per the session protocol; it waits for the owner's go.

Milestone 1 exit test: all non-parity tests in handoff section 7 pass.

| Exit-test part | State |
| --- | --- |
| Non-parity tests in section 7 pass | **Met.** 31 test cases in `tests/test_ref_sim.py` pass (16 map to the section 7 bullets); full suite 43 passed. Measured 2026-11-15, see `docs/milestone-1.md`. |
| CI green (handoff rule 9) | **Not verified.** The milestone 1 commits are local; CI runs when they are pushed. |

## Owner actions (what only you can do)

1. **Push the milestone 1 commits when you are ready** (`git push`, a normal push; I only push when asked), then check the Actions page: https://github.com/pyro30213131-png/QUADRACE/actions. If either job fails, send me the failing step's log. When both are green, milestone 1 is done and I wait for your go on milestone 2.
2. **Note on the commit identity.** Commits carry your existing git identity (`git config user.name` and `user.email`). The repo is public (GitHub's API answered without a login), so that name and the commit email are visible to everyone. If you want a different email, tell me before more commits pile up: changing the commits already pushed means another history rewrite and force-push.

## Last test results

Measured on 2026-11-15 on this machine (Rust 1.100.0, Python 3.12.10, NumPy 2.5.4):

| Check | Result |
| --- | --- |
| `pytest tests/test_ref_sim.py -q` | 31 passed |
| `pytest -q` | 43 passed in 1.82 s (12 milestone 0 tests plus 31 milestone 1 tests) |
| `cargo test -p quadsim` | 2 passed, 0 failed (Rust code unchanged since milestone 0) |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0) |
| Mutation check on `ref_sim.py` | 15 deliberate bugs, all caught by the tests (one needed a new test, added) |
| CI runs #1 and #2 on GitHub (milestone 0 code) | both `windows-latest` and `ubuntu-latest` passed every step, on both runs |

The individual physics numbers (free-fall error 1.3e-14 m, roll rate 5.23 rad/s at 0.2 s, angular-momentum drift 7e-11, and so on) are in `docs/milestone-1.md`. No test was weakened, skipped or deleted.

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
- **Milestone 1:** `python/quadrace/ref_sim.py` (one quadrotor in NumPy float64: rigid body with RK4, rate controller, mixing and allocation, motor lag, action mapping, optional linear drag), `tests/test_ref_sim.py` (31 test cases) and `docs/milestone-1.md`. Three commits, none pushed yet.

## Things that were unverified and now are

1. The Rust code compiled on the first build with no changes.
2. The maturin layout works: `maturin develop` writes `quadrace.pth` pointing at `python/`, so both packages import from any folder, and `python-packages = ["quadrace"]` puts `quadrace` in the wheel.
3. `cargo test` runs on Windows (the system Python is on PATH).
4. `cargo fmt --check` and `cargo clippy -- -D warnings` pass.

5. **CI works on both systems.** The `actions/checkout@v7` and `actions/setup-python@v7` tags exist, the Ubuntu steps (virtual environment, linking against the runner's Python) and the Windows steps passed, and the code builds and passes its tests on Linux as well as Windows (CI run #1).

## Still unverified

1. **CI on the milestone 1 commits.** They are local only. The new tests use only NumPy and PyYAML and no platform-specific code, so a failure is not expected, but it has not been run on Linux.
2. **The reference simulator has never been compared with the Rust code,** because the Rust simulator does not exist yet. Until the milestone 2 parity test passes, the section 7 tests are the only evidence the physics is right.
3. **Drone parameters are still the handoff's approximate values**, not from a verified source (handoff section 13). Fine for now; the Swift values are needed for milestone 6.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **PyTorch is not installed.** It is a download well over 100 MB, which CLAUDE.md says to ask about, and it is not needed until milestone 3.
- **Not installed:** TensorBoard, Rerun, matplotlib, ffmpeg (all later milestones).
- **`docs/milestone-0.md`** is final: milestone 0 is done.
- **Milestone 1 choices that the Rust port must copy** (details and reasons in `docs/milestone-1.md`): the rotation matrix is built from a normalized quaternion even inside RK4's intermediate stages; RK4 uses the thrust after this substep's motor lag; actions outside [-1, 1] are clipped (the handoff is silent, so this is my choice and is easy to change); the thrust clamp reads `thrust_min` from the config.
- **Finding for later:** at hover collective thrust, a full-scale yaw command (4 rad/s) asks two motors for below zero thrust, so yaw saturates in normal flight. Roll and pitch at full scale do not. No change made.

## Next

When you say go, milestone 2 is the Rust simulator: port one environment from `ref_sim.py`, parity tests (Rust and NumPy agree to 1e-9 after 1 s on 100 random states and action sequences), the PyO3 API, then rayon batching, a determinism test (1 and 8 threads) and a criterion benchmark. I have not started it. Read handoff sections 5 and 7 and `docs/milestone-1.md` first.

## Open questions

- Should the clipping of actions outside [-1, 1] stay as it is? (Handoff is silent; I chose clipping.)
