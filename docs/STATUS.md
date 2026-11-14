# STATUS

Last updated: 2026-11-15.

## Current milestone

**Milestone 0 (setup): done. Both halves of the exit test are met.** Milestone 1 has not been started, per the session protocol; it waits for the owner's go.

Exit test: `import quadsim` works locally and CI is green.

| Exit-test part | State |
| --- | --- |
| `import quadsim` works locally | **Met.** Measured 2026-11-14, see "Last test results". |
| CI green on Windows and Ubuntu | **Met.** CI run #1 (2026-11-14, https://github.com/pyro30213131-png/QUADRACE/actions) passed every step on both `windows-latest` and `ubuntu-latest`: format check, clippy, Rust tests, extension build, Python tests. That run was on the commit pushed before the commit messages were rewritten; the files in it are identical to the current local files except `CLAUDE.md` and `docs/STATUS.md` (checked with `git diff`). |

## Owner actions (what only you can do)

1. **Nothing is waiting on you for milestone 0.** The rewritten history was force-pushed on 2026-11-14 (by you, after the safety check blocked me). Checked afterwards: GitHub `main` equals local `main`, no commit message on it carries a `Co-Authored-By` line, and GitHub's contributors list shows only your account.
2. **Note on the commit identity.** Commits carry your existing git identity (`git config user.name` and `user.email`). The repo is public (GitHub's API answered without a login), so that name and the commit email are visible to everyone. If you want a different email, tell me before more commits pile up: changing the commits already pushed means another history rewrite and force-push.

## Last test results

Measured on 2026-11-14 on this machine (Rust 1.100.0, Python 3.12.10, release build for the extension):

| Check | Result |
| --- | --- |
| `cargo test -p quadsim` | 2 passed, 0 failed |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0) |
| `maturin develop --release` | builds and installs (exit code 0) |
| `pytest -q` | 12 passed in 1.28 s |
| `import quadsim, quadrace` from another folder | works; `add_arrays([1, 2], [10, 20])` returned `[11, 22]` |
| `maturin build` wheel | contains `quadsim` (with the compiled `.pyd`) and `quadrace` |
| CI run #1 on GitHub (2026-11-14) | `windows-latest`: all steps passed. `ubuntu-latest`: all steps passed |

Before the build, `pytest -q` gave 4 passed and 8 failed (`ModuleNotFoundError`), as expected with no extension. No test was weakened, skipped or changed between the two runs.

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

## Things that were unverified and now are

1. The Rust code compiled on the first build with no changes.
2. The maturin layout works: `maturin develop` writes `quadrace.pth` pointing at `python/`, so both packages import from any folder, and `python-packages = ["quadrace"]` puts `quadrace` in the wheel.
3. `cargo test` runs on Windows (the system Python is on PATH).
4. `cargo fmt --check` and `cargo clippy -- -D warnings` pass.

5. **CI works on both systems.** The `actions/checkout@v7` and `actions/setup-python@v7` tags exist, the Ubuntu steps (virtual environment, linking against the runner's Python) and the Windows steps passed, and the code builds and passes its tests on Linux as well as Windows (CI run #1).

## Still unverified

1. CI run #2, on the rewritten history, was still running when this was written: `ubuntu-latest` had passed and `windows-latest` had not finished. Run #1 passed both on identical code, so a failure is not expected, but check the Actions page.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **PyTorch is not installed.** It is a download well over 100 MB, which CLAUDE.md says to ask about, and it is not needed until milestone 3.
- **Not installed:** TensorBoard, Rerun, matplotlib, ffmpeg (all later milestones).
- **`docs/milestone-0.md`** is final except for the CI result, and says so at the top.

## Next

When you say go, milestone 1 is the NumPy reference simulator (`ref_sim.py`: dynamics, rate controller, mixing, motor lag, and the non-parity tests from handoff section 7). I have not started it.

## Open questions

- None for milestone 0.
