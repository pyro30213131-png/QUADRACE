# quadrace: agent handoff

Version 1.2, 13 November 2026. This file is the source of truth. It supersedes the earlier planning doc and chat plans wherever they differ.

## 1. What you are building

A quadrotor racing simulator written from scratch in Rust, exposed to Python through PyO3, plus a hand-written PPO implementation in PyTorch that trains a policy to fly a gate course as fast as possible. State-based only: no camera, no real drone.

Two tracks:

- **Figure-8** (8 gates, flat): the debug and first training track.
- **Split-S** (7 gates, from the Swift paper, Kaufmann et al., Nature 2023): the headline track.

The finished repo must let a stranger clone it, run two commands, and watch the trained policy fly a lap.

Out of scope unless the owner asks: vision policies, real hardware, multi-drone racing, obstacles other than gates, randomized tracks for generalization, differentiable simulation, PPO in Rust, SAC, GRU policies, asymmetric actor-critic, ONNX inference in Rust, SIMD tuning, minimum-snap baseline, battery voltage sag, Weights and Biases, Hydra, and scikit-learn analysis.

## 2. Owner context

- Machine: Windows laptop, native Windows (not WSL), NVIDIA RTX 4050 with 6 GB VRAM.
- The owner has written no Rust. You write all the code. The owner reviews, runs things, and must be able to explain the project in job interviews.
- Because of that, clarity beats cleverness everywhere. See the working rules.

## 3. Working rules

1. Work one milestone at a time, in order. Each has an exit test. Do not start the next milestone until the exit test passes.
2. At the end of every milestone, stop and report: what was built, test results, measured numbers, anything that deviated from this file.
3. Never weaken, skip or delete a test to make it pass. If an exit test cannot be met, stop and report with what you tried.
4. Never change the conventions in section 6 or the interfaces in section 5 without asking. Reward coefficients, PPO hyperparameters and controller gains are starting values; you may tune them, and you must log every change in `docs/tuning-log.md`.
5. Items in section 13 are unverified. Verify each from its source before relying on it. Do not invent values.
6. Rust style: simple and readable. No `unsafe`. No macros beyond `derive` and the PyO3 attributes. No clever generics. Comment the physics: each equation gets a comment naming the equation and the frame of every vector. Variable names carry their frame (`vel_w`, `omega_b`).
7. No heap allocation inside the per-step hot path.
8. Every physical constant lives in a YAML config. None in code.
9. Small commits, one task each. CI must be green before a milestone is reported done.
10. After each milestone, write `docs/milestone-N.md`: what was built, how to run it, how the code is organised, and five questions the owner should be able to answer about it, with answers.
11. Ask before: any training run expected to exceed one hour, any install needing admin rights, any download over 100 MB, anything outside the repo, choosing a licence, or making the repo public.
12. Report only measured numbers in the README. Mark estimates as estimates.

## 4. Environment setup (native Windows, PowerShell)

Check the current official install instructions for each tool; do not rely on remembered version numbers.

- Rust: install with rustup, MSVC toolchain. This needs the Visual Studio Build Tools with the "Desktop development with C++" workload. The owner must install that.
- Python 3.11 or newer, in a virtual environment:
  ```
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install maturin pytest numpy pyyaml
  ```
- PyTorch: the CUDA build, using the selector on pytorch.org. Verify with `torch.cuda.is_available()`.
- Build and test:
  ```
  maturin develop --release
  cargo test
  pytest
  ```
- Also needed later: TensorBoard, Rerun (`rerun-sdk`), matplotlib, ffmpeg on PATH.
- Pick PyO3, rust-numpy and maturin versions that are documented to work together. Record them in the README.
- Add a `.gitattributes` so line endings do not break CI.
- CI: GitHub Actions on `windows-latest` and `ubuntu-latest`, running `cargo test` and `pytest`.

## 5. Architecture

Python owns learning. Rust owns everything physical. They meet at one batched `step()` call per 50 Hz policy step.

```
quadrace/
  Cargo.toml              workspace
  pyproject.toml          maturin build
  crates/quadsim/
    src/lib.rs            PyO3 module
    src/math.rs           Vec3, Quat (hand-written, no nalgebra)
    src/dynamics.rs       rigid body, RK4
    src/motors.rs         mixing, lag, limits
    src/controller.rs     body-rate loop
    src/track.rs          gates, crossing test
    src/env.rs            observations, reward, reset
    src/batch.rs          rayon over N envs
    benches/step.rs       criterion benchmark
  python/quadrace/
    ref_sim.py            NumPy reference
    vec_env.py
    ppo.py
    train.py
    evaluate.py
    viz.py
    replay_progress.py
    baseline.py
  configs/                drone.yaml, drone_swift.yaml, task_hover.yaml,
                          track_fig8.yaml, track_split_s.yaml, ppo.yaml
  data/external/          owner-supplied files, not committed
  docs/
  tests/                  parity and environment tests
  .github/workflows/ci.yml
```

Choose the maturin layout so that the Rust module imports as `quadsim` and the Python package as `quadrace`. Document the choice.

Python-facing API of `quadsim`:

- `QuadSim(num_envs, drone_cfg, track_cfg, seed)` builds N environments. The track config's `task` field is `race` or `hover`.
- `reset()` returns observations, shape `[N, obs_dim]`, float32. `obs_dim` is 43 for racing and 22 for the hover task, and is exposed as `QuadSim.obs_dim`.
- `step(actions)` takes `[N, 4]` float32 and returns `obs, reward, terminated, truncated, final_obs, info`.
- Finished environments reset inside `step()`. `final_obs` holds the last observation of the episode that just ended.
- `info` holds per-environment reward terms `[N, 5]` (progress, gate bonus, rate penalty, action penalty, crash) and a termination-cause code `[N]`.
- `get_state()` and `set_state()` expose the full 17-value state per environment.

Implementation rules:

- Physics in float64. Observations cast to float32 at the boundary.
- `step()` releases the Python GIL and writes into preallocated NumPy arrays.
- Each environment has its own seeded random generator. Results must not depend on the thread count. Test with 1 and 8 threads.
- Start with a `Vec` of environment structs. Single-threaded first; add rayon only after parity passes.

Other fixed choices: TensorBoard for logging, plain YAML for configs (no Hydra), Rerun for interactive visualization.

## 6. Conventions (fixed)

- Units: SI. Angles in radians.
- World frame: z up, gravity `(0, 0, -9.81)` m/s^2.
- Body frame: x forward, y left, z up along the thrust axis.
- Quaternion: scalar first `(w, x, y, z)`, Hamilton product, rotates body to world. Renormalize after every physics substep. `q` and `-q` are the same rotation; compare up to sign in tests.
- Angular velocity `omega` is always in the body frame.
- State (17): position (3, world), quaternion (4), velocity (3, world), body rates (3, body), motor thrusts (4).
- Motors, X layout, `d = arm_length / sqrt(2)`:

  | motor | x | y | spin sign s |
  | --- | --- | --- | --- |
  | 0 | +d | -d | +1 |
  | 1 | -d | +d | +1 |
  | 2 | +d | +d | -1 |
  | 3 | -d | -d | -1 |

  Body torque from thrusts `f_i`: `tau_x = sum(y_i * f_i)`, `tau_y = -sum(x_i * f_i)`, `tau_z = sum(s_i * kappa * f_i)`. Collective thrust `T = sum(f_i)`.
- Gate frame: x along the pass direction, y left, z up. Opening corners in order, seen while approaching: top-left `(0, +h, +h)`, top-right `(0, -h, +h)`, bottom-right `(0, -h, -h)`, bottom-left `(0, +h, -h)`, with `h` half the opening.

## 7. Dynamics and control

Timing: physics and rate loop at 500 Hz (`dt = 0.002`), policy at 50 Hz (10 substeps per action).

Equations of motion (`R = R(q)`, body to world):

```
p_dot     = v
v_dot     = g + (1/m) * R * [0, 0, T] - (1/m) * R * D * R^T * v
q_dot     = 0.5 * q (x) (0, omega)
J*omega_dot = tau - omega x (J*omega)
f_i_dot   = (f_cmd_i - f_i) / tau_m
```

`D` is a diagonal drag matrix. `tau_m` is the motor time constant.

One physics substep, in this order:

1. Rate controller: `tau_des = J * (Kp .* (omega_cmd - omega)) + omega x (J*omega)`.
2. Allocation: invert the 4x4 mixing matrix to turn `[T_cmd, tau_des]` into four thrust commands.
3. Clamp each thrust command to `[0, f_max]`.
4. Motor lag, exact update: `f = f_cmd + (f - f_cmd) * exp(-dt / tau_m)`.
5. RK4 on position, velocity, quaternion and body rates, with motor thrusts held constant.
6. Renormalize the quaternion.

Action mapping, `a` in `[-1, 1]^4`: `T_cmd = (a[0] + 1) / 2 * 4 * f_max`; `omega_cmd = a[1:4] .* max_rates`.

`configs/drone.yaml`, approximate starting values for a 5-inch racing quad (not from a verified source):

| Parameter | Value |
| --- | --- |
| Mass | 0.75 kg |
| Inertia (x, y, z) | 0.0025, 0.0021, 0.0043 kg m^2 |
| Arm length | 0.15 m |
| Thrust per motor | 0 to 8.5 N |
| Motor time constant | 0.03 s |
| Yaw torque coefficient kappa | 0.02 m |
| Max body rates (roll, pitch, yaw) | 10, 10, 4 rad/s |
| Rate gain Kp (roll, pitch, yaw) | 20, 20, 10 per second |
| Linear drag | 0.3 N s/m per axis, off until the robustness milestone |

`configs/drone_swift.yaml`: same structure, with mass, thrust and rate limits taken from the Swift paper once verified (section 13).

Tests the simulator must pass before any learning:

- Parity: for 100 random states and action sequences, Rust and NumPy agree to 1e-9 in position and attitude after 1 s.
- Free fall with motors off follows `z = z0 - 4.905 t^2`.
- Hover thrust holds altitude within 1 mm over 5 s.
- A commanded roll rate of 5 rad/s is reached to within 10% in 0.2 s; a rate beyond what the motors can deliver saturates.
- With no torque and no drag, world-frame angular momentum `R * J * omega` stays constant.
- Constant `omega` with equal inertias matches the exact formula `q_next = q (x) (cos(|w|dt/2), sin(|w|dt/2) * w/|w|)`.
- Equal thrust on all motors gives zero torque; each single motor gives the torque signs implied by the table in section 6.
- A 90 degree yaw quaternion `(0.7071, 0, 0, 0.7071)` maps body `(1, 0, 0)` to world `(0, 1, 0)`.

## 8. Racing environment

One episode lasts at most 30 s (1,500 policy steps).

### Tracks

Track YAML: a list of gates, each with position, full orientation (quaternion), opening size and frame width. Gates are passed in list order, looping.

- `track_fig8.yaml`: 8 gates at z = 2.0 m on the curve `x = 8 sin(s)`, `y = 4 sin(2s)`, at `s = pi/8 + k*pi/4` for `k = 0..7`. Each gate faces along the curve tangent `(8 cos(s), 8 cos(2s))`. Opening 1.0 m, frame 0.15 m.
- `track_split_s.yaml`: 7 gates from the Swift paper's `track.yaml`. Positions, orientations and sizes must come from the original file (section 13). Gates 4 and 5 are stacked at the same x and y.

### Gates and collisions

The drone is a sphere of radius `r = 0.15` m. Run the crossing test every physics substep, on the segment from the previous position to the current one, in each gate's frame. With `h` half the opening, `w` the frame width, and `m = max(|y|, |z|)` at the crossing point:

- `m <= h - r`: through the opening.
- `h - r < m <= h + w + r`: frame hit.
- Otherwise: outside the gate.

Outcomes:

- Target gate, right direction, through the opening: **pass**. The target advances.
- Any gate, either direction, frame hit: **crash**. The episode terminates.
- Target gate, right direction, outside the gate but with `m <= 2 * (2h + 2w)`: **missed gate**. The episode terminates.
- Everything else (far-away plane crossings, flying through a non-target opening, crossing the target opening backwards): ignored.

### Observation (43 values, all relative to the drone)

| Part | Size |
| --- | --- |
| Velocity in the body frame | 3 |
| Rotation matrix, body to world, row-major | 9 |
| Body rates | 3 |
| Four opening corners of the next gate, in the body frame | 12 |
| Four opening corners of the gate after it, in the body frame | 12 |
| Previous action | 4 |

### Reward per policy step (starting values)

| Term | Value |
| --- | --- |
| Progress toward the next gate centre | +1.0 per metre gained |
| Gate passed | +2.5 |
| Body-rate penalty | -0.0002 x squared rate |
| Action-change penalty | -0.002 x squared change |
| Crash or missed gate | -5.0, once |

When a gate is passed, the progress distance switches to the new gate on that step, so the reward does not jump negative.

### Episode ends

Termination-cause codes: 0 none, 1 frame hit, 2 missed gate, 3 altitude below 0.1 m, 4 out of bounds (5 m outside the track's bounding box, or above 8 m), 5 no gate passed for 6 s, 6 time limit of 30 s.

Codes 1 to 4 set `terminated`. Codes 5 and 6 set `truncated`.

### Resets

Training episodes start at a random gate, not the start line:

- Pick a random target gate. Spawn on the straight segment from the previous gate's centre to the target's centre, at a fraction in `[0.1, 0.6]`.
- If those two gates are less than 2 m apart horizontally (the stacked Split-S pair), spawn 0.5 to 2 m beyond the previous gate along its pass direction instead.
- Position noise up to 0.5 m, yaw up to 0.3 rad off the direction of travel, tilt up to 0.2 rad, forward speed 0 to 8 m/s. Motors start at hover thrust.

Evaluation episodes start at rest, 3 m before gate 0 along its pass direction. A lap is complete when the drone has passed every gate in order and gate 0 again. Lap time is the time between two consecutive passes of gate 0.

### Curriculum

Gate openings start at 2.0 times nominal, then 1.5, then 1.0. Advance when 90% of evaluation episodes complete a lap. On Split-S, cap the scale so the frames of gates 4 and 5 never overlap.

### Domain randomization (per episode, robustness milestone)

| Quantity | Range |
| --- | --- |
| Mass, inertia, max thrust | plus or minus 10% |
| Motor time constant | plus or minus 20% |
| Drag | 0 to 2 times nominal |
| Action delay | 0 to 20 ms |
| Gate positions | plus or minus 0.1 m |
| Observation noise | 0.05 m/s velocity, 0.02 rad/s rates, 0.02 m corners |

### Hover task (milestone 3 only)

A simpler task on the same simulator, used to prove the physics and PPO work before gates exist. Config: `configs/task_hover.yaml` with `task: hover` and a target point, default `(0, 0, 2)`.

Observation (22 values): velocity in the body frame (3), rotation matrix (9), body rates (3), target position relative to the drone in the body frame (3), previous action (4).

Reward per policy step (starting values): +1.0 per metre of progress toward the target, -0.05 x distance to the target in metres, -0.0002 x squared body rate, -0.002 x squared action change, -5.0 once on a crash. The five `info` reward columns are, in order: progress, distance penalty, rate penalty, action penalty, crash.

Ends: `terminated` when altitude drops below 0.1 m or the drone is more than 5 m from the target. `truncated` at 15 s (750 steps).

Resets: position uniformly within 1.5 m of the target and at least 0.5 m above the ground, tilt up to 0.5 rad, speed up to 2 m/s, body rates up to 2 rad/s, motors at hover thrust.

Hover evaluation: 200 episodes, deterministic policy. Pass when, in at least 95% of episodes, the drone stays within 0.1 m of the target for the final 10 s.

## 9. Training and evaluation

PPO written by hand in PyTorch (`ppo.py`). Use CleanRL's single-file PPO as a reference and cross-check against Stable-Baselines3 on CartPole and Pendulum before touching the drone.

Starting hyperparameters:

| Setting | Value |
| --- | --- |
| Parallel environments | 2,048 (lower it if the CPU has few cores) |
| Rollout length | 64 steps |
| Minibatches and epochs | 8 and 5 |
| Discount and GAE lambda | 0.99 and 0.95 |
| Clip range | 0.2 |
| Learning rate | 3e-4, linear decay to zero |
| Entropy bonus | 0.001 |
| Gradient clip | 1.0 |
| Actor and critic | separate MLPs, 2 hidden layers of 256, tanh |
| Action noise | state-independent log standard deviation, starting at -0.5 |
| Observations | running mean and variance normalization |
| Seeds per reported result | 3 |

Correctness details:

- On a truncated episode the value target bootstraps from `final_obs`. On a terminated one it does not.
- Save observation statistics with every checkpoint.
- Save a checkpoint every fixed number of PPO updates, not only at the end.
- Each run stores its config, git commit and seed next to its checkpoints.

Live monitoring during training:

- Every 10 to 20 s of wall-clock time, run one episode of the current policy on a separate `QuadSim` with 1 to 4 environments and stream it to Rerun. Never render the training environments.
- TensorBoard, every update: lap completion rate and gates passed per episode next to total reward; each reward term separately; episode length; share of episodes per termination cause; crashes by gate; speed and altitude histograms.
- A run is suspect when reward rises while lap completion stays flat, when one reward term supplies most of the return, or when most episodes end by timeout. Known exploits: circling near a gate, oscillating at the gate plane, hovering to avoid the crash penalty, hugging the floor or boundary. Stop and report if you see one.

Evaluation: 200 episodes, deterministic policy.

| Metric | Target |
| --- | --- |
| Lap completion, nominal physics, figure-8 | at least 95% |
| Lap completion, nominal physics, Split-S | at least 90% |
| Lap completion, randomized physics, headline track | at least 85% |
| Lap completion, held-out physics (every randomization range widened to 1.5 times its training width) | reported, no target |
| Best and mean lap time | reported, at least 2 times faster than the baseline |
| Peak speed, crashes by gate | reported |
| Simulator throughput, simulator only | at least 1 million env steps per second (estimate) |
| Training throughput, end to end | at least 100,000 env steps per second (estimate) |

For scale: a public NumPy racing simulator (`MEO41/gate_racer`) reports about 190k drone-steps per second on one CPU core.

Baseline (`baseline.py`): a hand-tuned cascaded controller that flies gate to gate at a fixed low speed by commanding a velocity toward the next gate centre.

Training-progress replay (`replay_progress.py`): load 6 to 8 checkpoints from untrained to final, roll each out deterministically from the evaluation start, save the trajectories, and render an animation with the gates drawn (side by side, or a ghost trail from red to green) with the lap time on screen. Render with matplotlib and ffmpeg. Also save a Rerun `.rrd` recording of the final lap for an interactive replay link.

## 10. Milestones

| # | Milestone | Main tasks | Exit test |
| --- | --- | --- | --- |
| 0 | Setup | Repo, Cargo workspace, maturin hello-world (add two NumPy arrays in Rust), pytest, CI on Windows and Ubuntu, `drone.yaml`, conventions in README | `import quadsim` works locally and CI is green |
| 1 | NumPy reference | `ref_sim.py`: dynamics, rate controller, mixing, motor lag, sanity tests | All non-parity tests in section 7 pass |
| 2 | Rust simulator | Port single env, parity tests, PyO3 API, then rayon batching, determinism test, criterion benchmark | Parity passes, 1-thread and 8-thread results match, a throughput number is recorded |
| 3 | PPO and hover | `ppo.py`, checks on CartPole and Pendulum, hover task, checkpointing, per-term reward logging | The hover evaluation in section 8 passes |
| 4 | Gates | `track.rs`, crossing test with unit tests, observations, reward, ends, resets, Rerun replay, live view, monitoring metrics | On figure-8 at 2.0x openings, passes 3 gates in a row in 90% of episodes |
| 5 | Full lap, figure-8 | Curriculum to 1.0x, tuning, 3 seeds | At least 95% lap completion, nominal physics |
| 6 | Split-S | Load the verified `track.yaml`, `drone_swift.yaml`, train (try fine-tuning the figure-8 policy first) | At least 90% lap completion, nominal physics |
| 7 | Robustness | Drag, action delay, domain randomization, randomized evaluation | At least 85% lap completion, randomized physics, on Split-S |
| 8 | Results | Baseline, plots, training-progress video, Rerun recording, README, benchmark chart | README shows results over 3 seeds, both videos, benchmark number, CI badge |

Stop conditions:

- Milestones 5 and 6 have no natural end to tuning. After 10 full training runs without meeting the exit test, stop and report. Acceptable fallback: report results at 1.5x openings and say so plainly.
- Milestone 6 is blocked until the owner supplies the original `track.yaml`. Do not guess orientations or sizes.
- If Split-S fails after the cap, the figure-8 result is the headline and Split-S is reported as attempted.

## 11. Risks

| Risk | Sign | Response |
| --- | --- | --- |
| Sign or frame error in dynamics | Drift, flips, impossible speeds | Section 7 tests before any training |
| PPO bug mistaken for a hard task | No learning on hover | CartPole and Pendulum checks, SB3 comparison |
| Reward exploit | Reward up, laps flat | Live view and per-term metrics; stop and report |
| Results depend on thread count | Same seed, different results | Per-environment random generators, tested |
| Windows build friction | maturin or linker errors | Confirm MSVC Build Tools; record the working versions |
| Tuning never converges | Stop condition reached | Fallback above |

## 12. Definition of done

- `maturin develop --release` and `python -m quadrace.evaluate` work on a clean Windows machine.
- CI runs the Rust tests, parity tests and environment tests on Windows and Ubuntu.
- README has a results table over 3 seeds, a lap video, a training-progress video, an interactive replay link, and the measured benchmark number.
- Lap completion targets in section 9 are met, or the fallback is stated plainly.
- `docs/` has one page per milestone and a write-up covering the dynamics model, the reward, what failed and what fixed it.

## 13. Unverified items (check before use)

| Item | Where to verify |
| --- | --- |
| Split-S gate positions, orientations, sizes | `track.yaml` inside `racing_data.zip` (about 446 MB), Zenodo record 7955278, https://doi.org/10.5281/zenodo.7955278. The owner downloads it and puts `track.yaml` in `data/external/`. Check the record's licence before committing any of it. |
| Swift drone mass, max thrust, rate limits | Methods section of Kaufmann et al., Nature 620, 982 to 987 (2023), https://doi.org/10.1038/s41586-023-06419-4 |
| Compatible PyO3, rust-numpy, maturin versions | Each project's current documentation |
| PyTorch CUDA install command | pytorch.org selector |
| Throughput targets | Your own benchmark; the numbers above are estimates |
| Drone parameters in `drone.yaml` | Approximate; fine as defaults, not a citation |

Secondhand Split-S gate centres, for cross-checking only (from the public repo `mrandri19/quadcopter-racing`, which took them from the Swift `track.yaml` and dropped the orientations), in metres:

```
gate   x      y      z
1     -0.60  -0.86   3.68
2      9.00   6.45   1.05
3      8.85  -3.80   1.05
4     -4.30  -5.60   3.40
5     -4.30  -5.60   1.42
6      4.50  -0.45   1.05
7     -1.95   6.81   1.05
```

If the original file disagrees with these, the original wins.
