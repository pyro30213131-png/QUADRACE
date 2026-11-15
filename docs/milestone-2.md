# Milestone 2: Rust simulator

> **Milestone 2 is done: parity passes, 1-thread and 8-thread results match, a throughput number is recorded, and CI is green.** CI run #4 passed every step on both Windows (216 s) and Ubuntu (96 s), and a fresh clone from GitHub built and passed every check on the owner's machine. See `docs/STATUS.md`.

## What was built

- **The Rust simulator**, a line-by-line port of `python/quadrace/ref_sim.py`, in `crates/quadsim/src/`: one drone, then a batch of drones stepped side by side on several CPU threads.
- **`quadsim.QuadSim`**, the Python class that wraps it. It is the physics-only version of the API in handoff section 5 (see "Choices worth knowing", item 1).
- **Tests.** `tests/test_parity.py` (Rust against NumPy), `tests/test_determinism.py` (thread count does not change results), and 38 Rust unit tests.
- **A criterion benchmark**, `crates/quadsim/benches/step.rs`.

## How to run it

```powershell
.venv\Scripts\Activate.ps1
maturin develop --release          # rebuild the Rust extension (do this after any Rust edit)
cargo test -p quadsim              # 38 Rust unit tests
pytest -q                          # all Python tests, including parity and determinism (about 50 s)
pytest tests/test_parity.py -q     # just parity
cargo bench -p quadsim             # throughput benchmark (about 3 minutes, release build)
```

Using it from Python:

```python
import yaml, numpy as np, quadsim
cfg = yaml.safe_load(open("configs/drone.yaml"))
sim = quadsim.QuadSim(4096, cfg)                  # 4096 drones, all CPU threads
sim.step_physics(np.zeros((4096, 4), np.float32)) # one 50 Hz step for every drone
state = sim.get_state()                           # float64, shape [4096, 17]
```

## How the code is organised

| File | What is in it |
| --- | --- |
| `math.rs` | Hand-written `Vec3`, `Quat` (Hamilton product, body to world) and `Mat3`. No nalgebra. |
| `params.rs` | `DroneParams`: every physical constant, plus `validate()` so a bad config fails with a message. |
| `motors.rs` | Motor layout, thrust to torque, allocation (wanted thrust and torque to four motor commands, clamped), motor lag. |
| `controller.rs` | The body-rate controller. |
| `dynamics.rs` | `State` (the 17 values), the equations of motion, RK4, `physics_substep` (the six steps of section 7), `policy_step` (10 substeps). |
| `env.rs` | `Env`: one drone. Observations, rewards and gates will be added here in milestones 3 and 4. |
| `batch.rs` | `Batch`: many `Env`s, stepped on a rayon thread pool. `step_sequential` does the same on one thread. |
| `lib.rs` | The thin PyO3 layer: reads the config dict, checks shapes and dtypes, calls the plain Rust, turns errors into Python exceptions. |

One call to `QuadSim.step_physics` goes: check the input, release the Python GIL, hand the drones to the thread pool, each drone runs `policy_step` (10 times `physics_substep`) on its own, then check no state went bad. Nothing in the per-step path allocates on the heap; every value lives on the stack.

The Rust code mirrors the Python function names (`rate_controller`, `allocate`, `motor_lag`, `rk4_step`, `physics_substep`, `policy_step`), so the two files can be read side by side.

## Choices worth knowing

1. **(Updated in milestone 3: this is now closed. `QuadSim` takes `track_cfg` and `seed`, and has the full `reset()` / `step()`; `step_physics` remains for physics-only use. See `docs/milestone-3.md`.) The API is a subset of handoff section 5, and the step method is called `step_physics`, not `step`.** `QuadSim(num_envs, drone_cfg, num_threads=None)` has no `track_cfg` and no `seed`, and `step_physics` returns nothing: no observations, rewards, `terminated`, `truncated`, `final_obs` or `info`. Those need the environment code of milestones 3 and 4, and nothing in the physics is random yet, so an unused `seed` would be misleading. I named it `step_physics` so it does not pretend to be the full `step()` contract. `get_state` and `set_state` follow section 5. **This is the one place I deviate from section 5, so please say if you would rather have it another way.**
2. **The config arrives as a Python dict, not a YAML path.** Python reads the file (`yaml.safe_load`) and hands the dict over, so no YAML crate was needed in Rust. The cost: the Rust unit tests and the benchmark carry a literal copy of the drone numbers (clearly labelled), because Rust cannot read the file. The parity tests use the real file.
3. **Allocation uses a closed form instead of a matrix solve.** The four rows of the mixing matrix are at right angles to each other for the X layout, so inverting it is "project onto each row, divide by its squared length". A test checks the round trip, and parity checks it against NumPy's general solve. If the motor layout ever changes so the rows are no longer at right angles, this function must change.
4. **Each batch owns its own thread pool.** `num_threads=None` uses one thread per logical CPU. That is what makes "1 thread versus 8 threads" testable inside one process.
5. **Actions are float32, states are float64.** Section 5 says actions are float32 (they come from PyTorch) and physics is float64. The parity tests draw float32 actions and give NumPy the same values widened to float64, so both sides see identical inputs.
6. **The crate's minimum Rust version went from 1.85 to 1.86**, because criterion 0.8 needs 1.86. Also, the crate is now built as both `cdylib` (for Python) and `rlib` (so the benchmark can link to it).
7. **No `rand` crate yet.** Nothing is random until resets and domain randomization (milestones 3 to 7). The rule "one seeded generator per environment, results independent of thread count" is satisfied trivially today, and **the determinism tests must be extended when randomness arrives.**
8. **Same choices as the reference** (milestone 1, `docs/milestone-1.md`): the rotation matrix is built from a normalized quaternion even inside RK4, RK4 sees the thrust after this substep's motor lag, actions outside [-1, 1] are clipped.

## Results

Measured on 2026-11-15 on the owner's machine: Windows 11, Intel Core i5-13450HX (10 cores, 16 threads), Rust 1.100.0, release build.

| Check | Result |
| --- | --- |
| `cargo test -p quadsim` | 38 passed |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` | pass (clippy also compiles the benchmark) |
| `pytest -q` (everything) | 71 passed in 50 s |
| **Parity**, 100 random states, random float32 actions, 1 s (limit 1e-9) | worst difference 1.7e-14 m in position, 2.4e-15 in attitude, 3.8e-14 m/s in velocity, 8.7e-15 rad/s in body rates, 2.7e-15 N in motor thrust |
| Parity with drag enabled | same size (2.1e-14 m in position) |
| Parity after a single policy step | passes at 1e-12 |
| Parity with actions far outside [-1, 1], and with every action at a corner of the action box | passes at 1e-9 |
| **1 thread and 8 threads** | bit-for-bit identical, in Rust (100 drones, 1, 2, 8 and default threads against a sequential run) and in Python (512 drones, 100 steps) |

The drones in the parity runs moved up to 16 m and spun at up to 7 rad/s, so the comparison covers real flight and not just hovering.

### Throughput

One "env step" is one 50 Hz policy step of one drone, which is 10 physics substeps. Numbers are criterion's mean estimate, in million env steps per second; the handoff's target for "simulator only" is at least 1 million (an estimate, not a requirement from a source).

| Drones in the batch | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
| --- | --- | --- | --- | --- | --- |
| 1,024 | 0.85 | 1.39 | 2.61 | 3.86 | 4.55 |
| 4,096 | 0.85 | 1.57 | 2.83 | 3.71 | 4.81 |
| 16,384 | 0.55 | 1.01 | 2.12 | 3.77 | 5.05 |

- **One drone on one thread: 1.19 microseconds per env step**, about 0.84 million per second.
- **The target is met with threads** (4.8 million per second at 4,096 drones and 16 threads) **but not on a single thread** (0.85 million).
- Through the Python call `step_physics` (a one-off script, best of 3 runs, not in the repo): 0.94 million per second on 1 thread and 5.9 million with the default 16 threads at 4,096 drones. The Python call adds little on top of the Rust time.
- Scaling is not linear: 16 threads gives about 5.7 times one thread. This CPU has 6 fast cores and 4 slower ones, and the 16 threads include hyperthreads that share a core.
- The 16,384-drone single-thread number is lower than the others. My guess, not checked, is that the state of that many drones no longer fits in the CPU cache.
- Expect run-to-run noise of about 10 percent on a laptop. The recorded numbers are one run of `cargo bench -p quadsim`.
- This is the simulator alone. The handoff's other target, 100,000 env steps per second end to end including training, belongs to milestone 3.

### Does the test suite catch a wrong port? (mutation check)

As in milestone 1, I broke the Rust code in 14 different ways, one at a time (a flipped quaternion product order, a flipped pitch torque sign, a flipped yaw sign in allocation, RK4 using the thrust from before the lag, no clamp, no renormalization, doubled motor time constant, halved gravity, and others), rebuilt, and ran the parity tests. The first pass caught 12. One was missed and one was skipped by my script, and both were then resolved:

- **Missed: removing the quaternion renormalization.** Parity at 1e-9 cannot see it, because the drift in one second is smaller than that. I added `test_rust_quaternion_stays_unit_length_to_1e_12_after_one_second`, which catches it (a Rust unit test already did too).
- **Skipped (my script's text no longer matched after rustfmt): a wrong sign in one entry of the rotation matrix.** Re-run properly, it is caught by the drag-enabled parity test. The default tests (drag off) cannot see it: with drag off the physics only uses the third column of the rotation matrix (the thrust direction), and the entry I changed sits in the second column, which only the drag term reads. That is why a drag-enabled parity test exists.

All 14 are now caught, and the files were restored from git after each. The check script is not part of the repo.

## Five questions you should be able to answer

1. **What does the parity test prove, and what can it not prove?**
   It proves the Rust code does the same arithmetic as the Python code, to about 1e-14 here against a 1e-9 limit. It does not prove the physics is right: that was milestone 1's job, using results we could work out by hand (free fall, conserved angular momentum). It also only sees what the inputs exercise. With drag off, only the third column of the rotation matrix is ever used, so a wrong entry elsewhere is invisible, and a missing quaternion renormalization is smaller than the tolerance. So the suite has a drag-enabled parity run and a separate unit-length test.

2. **Why can the number of threads not change the results, and how do we know?**
   Each drone is stepped by the same code using only its own state and its own action. No drone reads or writes another, so it does not matter which thread runs it or in what order. We check it: a Rust test and a Python test step the same drones with 1, 2, 8 and the default number of threads and compare every bit of every state. The catch: once drones use random numbers (resets, randomization), each drone needs its own seeded generator, or the results would depend on scheduling. That is why the tests have to be extended in later milestones.

3. **What happens at the Python-to-Rust boundary when the input is wrong?**
   Nothing crashes. A wrong dtype (float64 actions, float32 states) gives `TypeError` before Rust code runs. A wrong shape, a NaN action, a state with a NaN or a quaternion that is not unit length, a bad config value, zero drones or zero threads all give `ValueError` or `KeyError` with a message naming the problem. `set_state` checks every row first and changes nothing if any row is bad. After each step the states are checked again for NaN. A Rust panic would otherwise become an exception that is hard to read, so the code avoids them.

4. **Why is allocation a closed-form formula and not a matrix inverse?**
   The four rows of the mixing matrix (total thrust, roll, pitch, yaw) are at right angles to each other for this motor layout. For such a matrix the inverse is: take the dot product with each row and divide by that row's squared length. It is shorter, needs no general linear algebra, and is exact. The price is that it only works for this layout, and a unit test and the parity test both check it against the general solve.

5. **What does "5 million env steps per second" mean, and why is 16 threads not 16 times faster?**
   An env step is one 50 Hz policy step of one drone, which is 10 physics substeps, each with a controller, motor lag and four derivative evaluations for RK4. So 5 million env steps per second is 50 million physics substeps per second. Threads are not 16 times faster because this CPU has 6 fast and 4 slow cores, and the 16 threads include hyperthreads that share a core. Memory and cache also limit it.
