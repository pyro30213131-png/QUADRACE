# Milestone 0: setup

> **Local half of the exit test passed; CI half not yet verified.** `import quadsim` works locally and all tests pass (see "Results"). The CI workflow has never run, because the repo has not been pushed to GitHub. This page is final once CI is green on Windows and Ubuntu. See `docs/STATUS.md`.

## What was built

- A Cargo workspace with one crate, `quadsim`, whose only function is `add_arrays`: it adds two 1-D float64 NumPy arrays in Rust. It exists to prove that Rust, PyO3, rust-numpy, maturin and Python all fit together.
- A maturin build (`pyproject.toml`) that compiles the crate and installs it into the virtual environment.
- Two Python packages: `quadsim` (a thin wrapper around the compiled code) and `quadrace` (empty; the learning code goes here later).
- `configs/drone.yaml`: the starting drone parameters from handoff section 7.
- Python tests (`tests/test_setup.py`), Rust unit tests (in `crates/quadsim/src/lib.rs`), and a CI workflow for Windows and Ubuntu.

## How to run it

```powershell
.venv\Scripts\Activate.ps1
maturin develop --release
cargo test -p quadsim
pytest -q
```

## How the code is organised

- `crates/quadsim/src/lib.rs` has two layers. `add_views` is plain Rust with no Python in it, so `cargo test` can check it. The `_quadsim` module underneath is the PyO3 layer: it turns NumPy arrays into Rust views, calls `add_views`, and turns a Rust error into a Python `ValueError`.
- Wrong dtype or wrong number of dimensions never reaches our code: the argument type `PyReadonlyArray1<f64>` makes PyO3 raise `TypeError`. A length mismatch is checked by us and raises `ValueError`. Nothing in this path can panic into Python.
- The compiled file is `quadsim._quadsim`. `python/quadsim/__init__.py` re-exports it so users write `import quadsim`.
- `configs/drone.yaml` also holds gravity and the 500 Hz / 50 Hz timing, so no physical constant is hard-coded.

### Choices worth knowing

- **Versions.** PyO3 0.30 and rust-numpy 0.30 must share a version number; rust-numpy 0.30.0 declares `pyo3 ^0.30.0`. maturin must be at least 1.9.4. Rust 1.85 or newer is needed.
- **Layout.** maturin fails if `python-source` is set, `module-name` has no dot, and there is no Python directory of that name. So the compiled module is `quadsim._quadsim` inside a real `quadsim` package. This is the documented maturin pattern. `maturin develop` also writes a `quadrace.pth` file pointing at `python/`, so both packages import from any folder, and a built wheel contains both (checked on 2026-11-14).
- **Line endings.** `.gitattributes` forces LF so `cargo fmt --check` behaves the same on Windows and Linux in CI.
- **Cargo.lock is committed** so builds use exactly the same dependency versions.
- **Windows username with a space.** The Visual C++ installer failed with error 1622 because one of its scripts passes a log path to a program without quotes. Running the installer with `TEMP` and `TMP` set to `C:\vs-temp` fixed it on this machine.

## Results

Measured on 2026-11-14 on the owner's Windows machine (Rust 1.100.0, Python 3.12.10):

| Check | Result |
| --- | --- |
| `cargo test -p quadsim` | 2 passed, 0 failed |
| `cargo fmt --check` | passes (exit code 0) |
| `cargo clippy --all-targets -- -D warnings` | passes (exit code 0) |
| `maturin develop --release` | builds and installs (exit code 0) |
| `pytest -q` | 12 passed in 1.28 s |
| `import quadsim, quadrace` from another folder | works; `quadsim.add_arrays([1, 2], [10, 20])` returned `[11, 22]` |
| `maturin build` wheel | contains `quadsim` (with the compiled `.pyd`) and `quadrace` |
| CI on Windows and Ubuntu | **not run yet** |

Before the build, the same `pytest` run gave 4 passed and 8 failed (`ModuleNotFoundError`), which is what a missing extension should look like. No test was changed between the two runs.

## Five questions you should be able to answer

1. **Why is the compiled module called `_quadsim` and not `quadsim`?**
   The package directory `python/quadsim/` has to exist for maturin's mixed layout. The compiled code is an implementation detail inside it, so it gets an underscore, and `__init__.py` re-exports the public functions.

2. **What does `maturin develop --release` do, and why `--release`?**
   It compiles the Rust crate, copies the compiled file into the Python package, and registers the package in the active virtual environment so `import quadsim` works. `--release` turns on compiler optimisations; a debug build can be many times slower, so no speed number may ever come from one.

3. **What happens if you pass a float32 array, or two arrays of different length?**
   A float32 array (or a 2-D array) is rejected by PyO3 with a `TypeError` before our code runs, because the parameter type demands a 1-D float64 array. Different lengths are caught by our own check and raised as a `ValueError`. Neither can crash the interpreter, which is the "validate at the boundary" rule.

4. **Why split `add_views` from the PyO3 function?**
   So the logic can be unit-tested in plain Rust without a Python interpreter, and so the Python-facing layer stays thin. The same pattern will apply to the simulator: physics in plain Rust, a thin PyO3 shell.

5. **Why do PyO3 and rust-numpy need matching version numbers?**
   rust-numpy is built on PyO3's types (`Bound`, `Python`), and those change between PyO3 releases. rust-numpy 0.30 is written against PyO3 0.30, and Cargo would refuse to link two incompatible copies of PyO3 into one program.
