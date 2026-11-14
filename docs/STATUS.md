# STATUS

Last updated: 2026-11-14.

## Current milestone

**Milestone 0 (setup): the local half of the exit test passes; the CI half is not verified.** Milestone 1 has not been started, per the session protocol.

Exit test: `import quadsim` works locally and CI is green.

| Exit-test part | State |
| --- | --- |
| `import quadsim` works locally | **Met.** Measured 2026-11-14, see "Last test results". |
| CI green on Windows and Ubuntu | **Not verified.** The workflow has never run: this folder is not a git repository and has no GitHub remote. |

## Owner actions (what only you can do)

1. **Create the GitHub repository and push**, so CI can run. I have not run `git init` or committed anything, because you have not asked me to. Say the word and I will initialise git and make small commits; I will not push.
2. **Open the Actions tab** after the first push. If either job (Windows or Ubuntu) fails, send me the failing step's log and I will fix it.

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

Before the build, `pytest -q` gave 4 passed and 8 failed (`ModuleNotFoundError`), as expected with no extension. No test was weakened, skipped or changed between the two runs.

## Toolchain log (2026-11-14)

- **Rust: installed** with rustup: `stable-x86_64-pc-windows-msvc`, rustc and cargo 1.100.0. A shell opened before the install needs `%USERPROFILE%\.cargo\bin` on PATH or a restart.
- **Visual Studio Community 2026: installed by the rustup flow, failed, and was removed.** The C++ workload was marked incomplete: one package, `Microsoft.VisualCpp.Redist.14.Latest` (x86), failed with Windows error 1622 ("Error opening installation log file"), and the Windows SDK was never installed. I uninstalled it with Visual Studio's own uninstaller (exit code 0).
- **Cause of the failure, most likely: the space in the Windows username.** The package's log shows its path as `C:\Users\Anish Krishna\AppData\...`, and `VCRedistInstall.ps1` (line 254) builds `-ArgumentList "/q /norestart /log $LogFile"` without quotes around the path. The same install then succeeded with `TEMP` and `TMP` set to `C:\vs-temp`, which fits that explanation. It is evidence, not proof.
- **Visual Studio Build Tools 2022: installed** with `winget` (`Microsoft.VisualStudio.2022.BuildTools`, workload `Microsoft.VisualStudio.Workload.VCTools`, with recommended components), launched elevated with `TEMP` and `TMP` set to `C:\vs-temp`. Checked afterwards: `vswhere` reports the instance complete (version 17.14.37906.3) with the MSVC x64/x86 tools, `link.exe` is present (MSVC 14.44.35207), the Windows SDK is 10.0.26100.0, and none of the installer's error logs contains anything. winget showed no agreements to accept, so none were accepted on your behalf.
- **Leftover:** `C:\vs-temp` holds the installer's logs (about 350 files). It is safe to delete when you no longer want them. Free space on C: is now 12.5 GB.

## What is done

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

## Still unverified

1. **CI has never run.** Parts of it that could still fail: the `actions/checkout@v7` and `actions/setup-python@v7` tags (I checked the latest release numbers, not that the short `v7` tag exists), the Ubuntu steps (`source .venv/bin/activate`, linking against the runner's Python), and the Windows runner's Git Bash steps.
2. Nothing here has been run on Linux.

## Decisions and deviations

- **Config extras.** `configs/drone.yaml` also holds gravity (9.81) and the 500 Hz / 50 Hz timing, so no physical constant lives in code (handoff rule 8). These values are already fixed by sections 6 and 7; only their location is my choice. `drag.enabled: false` is how I encoded "drag off until the robustness milestone".
- **PyTorch is not installed.** It is a download well over 100 MB, which CLAUDE.md says to ask about, and it is not needed until milestone 3.
- **Not installed:** TensorBoard, Rerun, matplotlib, ffmpeg (all later milestones).
- **`docs/milestone-0.md`** is final except for the CI result, and says so at the top.

## Next

Once CI is green on both systems and you say go, milestone 1 is the NumPy reference simulator (`ref_sim.py`: dynamics, rate controller, mixing, motor lag, and the non-parity tests from handoff section 7).

## Open questions

- Should I run `git init` and commit the scaffold? Which branch name (`main`)? CI is set up for pushes to `main` and for pull requests.
