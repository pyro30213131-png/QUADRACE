# Milestone 3: PPO and hover

> **Exit test met locally: on three training seeds, the hover evaluation passes (199 to 200 of 200 episodes; the requirement is 95%).** CI has not run on the final commits because they are not pushed. The handoff says CI must be green before a milestone is reported done, so treat this milestone as done once they are pushed and both CI jobs are green. See `docs/STATUS.md`.

Exit test (handoff sections 8 and 10): over 200 episodes with a deterministic policy, in at least 95% the drone stays within 0.1 m of the hover target for the final 10 s.

## What was built

**Part 1: PPO**

- `python/quadrace/ppo.py`: PPO written by hand in PyTorch. The training loop, the actor and critic networks, GAE, the clipped loss, observation (and optional reward) normalization, checkpoints that carry the observation statistics, evaluation, and per-reward-term logging.
- `python/quadrace/vec_env.py`: the `VecEnv` interface PPO trains against, a gymnasium adapter (for CartPole and Pendulum), and `QuadVecEnv`, which makes the Rust simulator a `VecEnv`.
- `python/quadrace/ppo_crosscheck.py`: runs our PPO and Stable-Baselines3 side by side on CartPole and Pendulum.

**Part 2: the hover task**

- **In the Rust simulator** (`crates/quadsim/src/env.rs`, `batch.rs`, `lib.rs`): the hover task's 22-value observation, 5-term reward, episode ends (hit the ground, flew away, time limit), random resets with one seeded random generator per drone, automatic restart of finished episodes with `final_obs`, and the full `reset()` / `step()` interface of handoff section 5.
- `configs/task_hover.yaml`: every number of the task (target, limits, reset ranges, reward weights).
- `configs/ppo.yaml`, `python/quadrace/train.py`: the training script. `python/quadrace/evaluate.py`: the hover evaluation.
- **A viewer** (`python/quadrace/viz.py`), added at the owner's request after the exit test passed: watch a trained policy fly in an interactive 3D Rerun window, or as a GIF (see "Seeing it fly" below).
- **Tests:** 24 for the hover environment through Python, 14 for the training script, the wrapper and the evaluation, 47 for PPO, 21 for the viewer (177 Python tests in total across all milestones), and 71 Rust tests.

## How to run it

```powershell
.venv\Scripts\Activate.ps1
maturin develop --release                      # after any Rust edit

# Train the hover policy (about 4 minutes on the RTX 4050) and evaluate it:
python -m quadrace.train --run-name hover1 --total-timesteps 50003968
python -m quadrace.evaluate --checkpoint runs\hover1\checkpoint_000381.pt

tensorboard --logdir runs                      # watch training: returns, losses, each reward term, why episodes end

# Watch the trained policy fly (a GIF and a PNG in runs\renders\, and the interactive Rerun viewer):
python -m quadrace.viz --checkpoint runs\hover1\checkpoint_000381.pt --rerun
python -m quadrace.viz --checkpoint runs\hover1\checkpoint_000381.pt --hardest-of 64 --rerun   # the stress demo: most tilted starts
pytest -q                                      # all Python tests (about 2 minutes)
cargo test -p quadsim                          # all Rust tests
```

A run writes to `runs/<name>/` (not committed): `meta.json` (settings, git commit, seed), `metrics.jsonl`, TensorBoard logs, and checkpoints.

## How the code is organised

| Part | What it does |
| --- | --- |
| `env.rs`: `HoverConfig`, `Env` | One drone plus its episode bookkeeping. `reset_hover` draws a random start; `step_hover` does physics, reward, episode-end checks, observation, and the automatic restart. Each `Env` owns its own seeded generator. |
| `env.rs`: `hover_observation` | The 22 observation values, all relative to the drone. |
| `batch.rs`: `with_task`, `reset_task`, `step_task` | Many `Env`s on the thread pool, writing into preallocated output arrays. `step_task_sequential` is the one-thread reference. |
| `lib.rs`: `QuadSim` | The Python class: `reset(seed)`, `step(actions)` returning `(obs, reward, terminated, truncated, final_obs, info)`, `obs_dim`, `task`, `reward_term_names`, plus `step_physics`, `get_state`, `set_state` from milestone 2. |
| `ppo.py` | `PPOConfig`, `Agent` (actor and critic), `compute_gae`, `ppo_loss`, `apply_update`, `train`, checkpoints, `evaluate`. |
| `vec_env.py`: `QuadVecEnv` | `QuadSim` as a `VecEnv`; keeps the latest `info` so PPO can log each reward term. |
| `train.py`, `evaluate.py` | The command-line scripts. |

## Choices worth knowing

**About PPO**

1. **Tanh is inside the network, not on the output.** Continuous actions are sampled from a Gaussian and are not squashed; the environment clips them to [-1, 1], as CleanRL and Stable-Baselines3 do. A tanh-squashed output (as in SAC) is an alternative if actions get stuck at the limits; the handoff does not decide this.
2. **Time-limit cut-offs are bootstrapped from `final_obs`; real endings are not.** Tests check this against values worked out by hand.
3. **The value loss is a plain mean squared error with weight 0.5 and no value clipping**, matching Stable-Baselines3 so the cross-check is fair. Reward normalization exists but is off by default.
4. **Checkpoints are loaded with `weights_only=True`**, so opening one cannot run code.

**About the hover task** (all numbers are in `configs/task_hover.yaml`)

5. **The handoff leaves some reset details open; I chose:** the start yaw is 0 (yaw does not matter for reaching a point); "speed up to 2 m/s" and "body rates up to 2 rad/s" mean a random direction with a size drawn uniformly from 0 to the maximum; position is uniform in the ball of radius 1.5 m around the target, redrawn until it is at least 0.5 m up (with the default target the floor never binds).
6. **Reward terms are reported already weighted**, so the five columns in `info["reward_terms"]` add up to the reward. The action-change penalty uses the clipped actions.
7. **A step that terminates is never also reported as truncated**, even if the time limit coincides.
8. **The first observation of a new episode has a zero previous action**, and a drone whose episode ends restarts inside `step`, so `obs` is already the new episode and `final_obs` holds the old one. This is the contract PPO relies on.
9. **`QuadSim` keeps a physics-only mode.** Without a `track_cfg` it works as in milestone 2 (`step_physics`), so every milestone 2 test still runs unchanged. With one, it is the environment. **This closes the deviation from handoff section 5 recorded after milestone 2**: `QuadSim(num_envs, drone_cfg, track_cfg, seed, num_threads)`, `reset()`, `step()` with the six outputs, `obs_dim`, `get_state`/`set_state`.
10. **Random numbers:** the `rand` 0.10 and `rand_pcg` 0.10 crates (approved in CLAUDE.md), with `Pcg64Mcg` generators. Environment `i` of a batch seeded `s` uses seed `s + i * 0x9E3779B97F4A7C15` (scrambled by `seed_from_u64`), so an environment's random numbers do not depend on its thread or on the other environments.
11. **The crate's minimum Rust version stays at 1.86**, and `rand` 0.10 needs 1.85, so nothing changed there.

## Results

Measured on 2026-11-15 on the owner's machine (Windows 11, Intel i5-13450HX, RTX 4050, PyTorch 2.14.1, release build of the simulator).

### The exit test

Same settings on every run: the handoff's starting values for PPO and for the reward, unchanged. Only the training budget differs between the first attempt and the rest. "Standard evaluation" is `python -m quadrace.evaluate`: 200 episodes, seed 10,000, deterministic policy. "Independent check" re-scores 500 episodes on a seed (777) that neither training nor the standard evaluation used, from the simulator's true positions rather than from the observation.

| Run | Steps | Training time | Standard evaluation (200 episodes) | Independent check (500 episodes) |
| --- | --- | --- | --- | --- |
| First attempt, seed 1 | 10,027,008 | 0.7 min | **0 of 200**: all survive, but settle 0.12 to 0.14 m from the target | not run |
| Seed 1 | 49,938,432 | 3.9 min | **200 of 200** | 497 of 500 (2 ended early, 1 slow to settle) |
| Seed 2 | 49,938,432 | 4.7 min | **199 of 200** (1 hit the ground) | 500 of 500 |
| Seed 3 | 49,938,432 | 5.0 min | **200 of 200** | 500 of 500 |

- **Requirement: 95%. All three seeds are above it by a wide margin.** The drones settle within about 0.4 to 1 mm of the target (median worst distance in the final 10 s).
- The seed 2 and 3 training times were measured while other jobs shared the machine, so they are a little slow.
- **What the first attempt tells us:** after 10 million steps the policy had learned not to crash (98% of episodes reached the time limit) but did not hold the target closely: the distance penalty is only 0.05 per metre, small next to the smoothness penalties, so "close enough" was a good trade. Five times more training with the same reward closed the gap, so no reward or PPO setting was changed (see `docs/tuning-log.md`).
- **At the end of training** the reward is almost all small penalties: per step about -0.00017 for distance (a mean distance of about 3 mm), -0.00047 for action changes, -0.00009 for body rate, and nothing for crashes. So the policy holds the target and only pays for the last bits of smoothness.
- **Trained policies are not committed** (`runs/` is ignored, per CLAUDE.md); the commands above reproduce them.

### Is the exit test itself sound?

- **A classical controller passes it.** A hand-written position/attitude controller that reads the same 22 observation values (no learning) scores 200 of 200. The same controller aimed 0.3 m off the target scores 0 of 200 with every miss at 0.300 m. So the evaluation scores correctly in both directions, and the test is achievable. These controllers are in `tests/test_hover_training.py`.
- **The two measurements agree.** The standard evaluation uses distances from the observation; the independent check uses the simulator's own positions on a different seed. They give the same picture.

### Speed

| Measurement | Result |
| --- | --- |
| Whole training loop (simulator, PPO updates on the GPU, policy forward passes), 2,048 environments | about 215,000 to 260,000 environment steps per second (target: at least 100,000) |
| One PPO update at 131,072 samples | GPU 284 ms; CPU with 10 threads 2,586 ms; 1 thread 12,474 ms |
| The hover environment's `step` alone (physics, observation, reward, resets) | 0.85 million steps per second on 1 thread; 3.5 to 4.1 million on 16 threads |

The observation and reward add almost nothing to the physics cost (0.85 million per second on one thread, as in milestone 2). The 50-million-step runs took about 4 to 5 minutes, so no run came near the one-hour limit that needs your OK.

### Tests

| Check | Result |
| --- | --- |
| `pytest -q` | 177 passed in 82 s |
| `cargo test -p quadsim` | 71 passed |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` | pass |
| Earlier CI runs (#1 to #5) | all passed on Windows and Ubuntu; #5 was the PPO commits with PyTorch installed |

### Do the tests catch a wrong implementation? (mutation checks)

- **PPO: 18 deliberate bugs in `ppo.py`, all caught.** 17 on the first pass; the 18th (no gradient clipping) led me to split that step out and test it.
- **The hover environment: 24 deliberate bugs in `env.rs`, all caught.** A flipped progress or distance sign, a rate penalty that is not squared, a crash penalty paid on a time-limit cut-off, a wrong altitude test, a time limit off by one, no restart after an episode, `final_obs` not set, the target or velocity in the wrong frame, a transposed rotation matrix, reset ranges doubled, the altitude floor ignored, motors not at hover thrust, every environment with the same seed, stale previous action or distance, and others. The first pass found five real gaps (the crash penalty on truncation, a termination coinciding with the time limit, stale previous distance, a symmetric rotation matrix hiding a transposition, and a reset floor that the default numbers never reach); I added a test for each, and one more mutation turned out to be a flaw in my script. Files were restored from git after every mutation.

### Cross-check of PPO against Stable-Baselines3 (CartPole and Pendulum)

Same settings, same number of environments, same seeds (1, 2, 3), same budget; the settings are in `docs/tuning-log.md`. Mean and standard deviation across the 3 seeds. "Return at x%" is the mean return of the last 100 training episodes at that fraction of the budget; "Deterministic eval" is 50 episodes with the noise off.

**CartPole-v1** (500,000 steps; best possible return 500; "learned" is a mean of at least 475):

| | Ours | Stable-Baselines3 |
| --- | --- | --- |
| Return at 25% | 363.9 +- 66.8 | 365.4 +- 20.3 |
| Return at 50% | 363.8 +- 98.2 | 460.6 +- 28.3 |
| Return at 75% | 424.8 +- 61.5 | 378.3 +- 156.0 |
| Return at 100% | 426.9 +- 92.1 | 340.5 +- 118.7 |
| Deterministic eval | 414.3 +- 100.4 | 355.2 +- 103.7 |
| Seeds reaching 475 | 3 of 3 | 3 of 3 |
| Mean steps to reach 475 | 208,725 | 194,560 |

**Pendulum-v1** (1,000,000 steps; returns are negative, closer to zero is better; "learned" is a mean of at least -250):

| | Ours | Stable-Baselines3 |
| --- | --- | --- |
| Return at 25% | -191.1 +- 29.2 | -169.8 +- 6.0 |
| Return at 50% | -245.7 +- 17.8 | -183.7 +- 3.1 |
| Return at 75% | -187.2 +- 5.5 | -169.7 +- 10.9 |
| Return at 100% | -214.5 +- 48.5 | -180.8 +- 15.0 |
| Deterministic eval | -193.8 +- 37.6 | -185.2 +- 30.3 |
| Seeds reaching -250 | 3 of 3 | 3 of 3 |
| Mean steps to reach -250 | 103,083 | 92,160 |

Both implementations learn both problems on every seed at about the same speed. The differences are inside the seed-to-seed spread, which is large with 3 seeds; this is evidence that `ppo.py` is not broken, not proof that it is identical. (The cross-checks started before two behaviour-preserving refactors of `ppo.py`; I checked that a short training run gives bit-for-bit the same weights before and after each.)

## Seeing it fly: the viewer

Added on 2026-11-16 at the owner's request, after the exit test passed. It is a small piece pulled forward from later milestones: the handoff plans Rerun for milestone 4 (a live view of the policy while it trains, and a replay) and videos for milestone 8. Those stay where they are; this lets the hover policy be watched now.

**What you are watching.** Each episode throws the drone into the air at a random point within 1.5 m of the target, tilted by up to about 29° in a random direction, moving up to 2 m/s in a random direction and spinning up to 2 rad/s. The policy sees only its own velocity, orientation, rates, the direction to the target, and its last action, and it has to level out and settle on the target. Two recorded demos, both measured from the recordings:

| Demo | The starts | Within 0.1 m of the target from | Within 1 mm from | Final distance |
| --- | --- | --- | --- | --- |
| Default (seed 2026) | Three random starts, not hand-picked. They happened to be gentle: tilts of 3.9°, 4.8° and 5.0°, 0.34 to 0.77 m from the target | 0.6 to 1.1 s | 2.7 to 3.0 s | 0.37 mm |
| **Stress** (`--hardest-of 64`) | The 3 most tilted of 64 random starts: tilts of 28.2°, 27.9° and 27.8° (the task's maximum is 28.6°), 0.64 to 1.03 m from the target | 0.8 to 1.4 s | 2.4 to 3.3 s | 0.4 mm |

The stress demo is chosen for tilt only: the starts are still drawn from the task's own start distribution, but the three shown are the worst of 64, and the picture's title says so. The limits: it never starts upside down or past 29°, and it always starts facing the same way (yaw 0).

**3D or 2D?** Both views are **3D**, drawn as a clean wireframe rather than a game-style world with textures: a ground grid, each drone as an X-frame with four rotor discs and a black nose showing which way it faces, a red cross for the target with a 0.1 m ring, and the path flown so far.

| View | What it is |
| --- | --- |
| **Rerun** (`--rerun`) | An interactive 3D window: drag to rotate, scroll to zoom, press play or drag the timeline to scrub through the flight. Beside it, two plots of the distance to the target (plain, and log10). The recording is a 2 MB `.rrd` file that you can reopen any time with `rerun <file>`. |
| **GIF** (`runs/renders/hover_demo.gif`) | The same flight as a fixed animation: a 3D scene whose camera drifts slowly, next to a log-scale plot of the distance to the target. It takes about 65 s to draw because matplotlib draws every frame on the CPU. |
| **PNG** (`runs/renders/hover_demo.png`) | A still, in 2D: top and side views of the paths. |

**Honest details.**

- The drones are drawn **5 times life size**, because a 0.15 m arm is tiny in a 6 m scene (`--scale` changes it).
- Every drone flies the same task to the same target point, so drawn in one place they would overlap. They are **drawn side by side** (shifted along x). That moves only the picture, not the flights.
- The flights are recorded on their **own small simulator** (3 drones by default, 8 at most), never on the training environments (CLAUDE.md).
- The recordings are **not live**; a live view while training is milestone 4. The recording uses the seed-1 hover policy; all three seeds pass the exit test.
- I confirmed that the recording is valid (Rerun's own `rrd verify` passes), that the viewer window opens, and that the GIF is correct frame by frame. I did not look at the Rerun window myself, so tell me if anything looks off.
- Tests (21): the drawing geometry (motor positions from the handoff table, yaw and roll, scale, side-by-side offsets), the recorder (frame counts, random starts, crashed drones frozen instead of showing their restart, repeatability, the start tilt, and that `--hardest-of` keeps the most tilted starts of a real random pool), and the files written (GIF frame count, PNG, a valid Rerun recording with one row per frame for every entity).

**The decision, and the downloads.**

- The owner approved Rerun on 2026-11-16 ("go for rerun"). Its wheel is **145.6 MB, over the 100 MB that CLAUDE.md says to ask about**, which is why I asked first. `rerun-sdk` 0.43.2 is installed, along with its dependencies pyarrow (29 MB), psutil and attrs. `matplotlib` 3.11.2 (8.9 MB) is installed too; Pillow was already there. Both are approved in CLAUDE.md. CI installs both so the viewer tests run there.
- **Why the download was so slow.** The first attempt crawled at 0.03 to 0.06 MB/s and then gave up. The cause was not the Wi-Fi (96% signal, 1.2 Gbps link, 6 ms latency): **a Steam download was using about 4.7 MB/s of the roughly 5 MB/s the connection delivers**, so everything else got scraps. After the owner paused Steam, the same download ran at 3.3 MB/s and installed in 66 seconds. The earlier 2.42 GB PyTorch download was fast for the same reason: Steam was idle then.

## Things not done or not verified

- **CI on the final commits** (see the banner). Not pushed.
- **Only the hover task exists.** Gates, the racing observation and reward, and the Rerun views are milestone 4.
- **Three seeds, one set of settings.** The exit test is passed on all three, but I did not explore other settings; the handoff's defaults worked.
- **Two failure cases remain unexplained.** A few episodes end early (2 of 500 on one check, 1 of 200 on another). The reset can start a drone 0.5 m up while falling at 2 m/s and tilted by 0.5 rad, which may be too hard to recover from with the thrust available; I did not check this.

## Five questions you should be able to answer

1. **What does the "clip" in PPO do, and why is it there?**
   After each batch of experience PPO changes the policy over several passes through the same data. The clip says: once an action's probability has changed by more than about 20% from what it was when the data was collected, stop giving extra credit for pushing a good action's probability further up (or a bad action's further down). It only ever removes credit, never a penalty: a bad action whose probability went up is still punished in full. That keeps each update small, so one noisy batch cannot wreck a policy that was working.

2. **What are the critic and the advantage for?**
   The critic estimates how much total future reward to expect from a state. The advantage is roughly "what actually happened minus what the critic expected". Using it makes the learning signal much less noisy than raw returns: an action is rewarded for doing better than expected, not for happening in a lucky state. GAE blends one-step and long-horizon estimates; lambda 0.95 is the handoff's trade-off between bias and noise.

3. **Why is the value bootstrapped on a time-limit cut-off but not on a crash?**
   A crash really ends the future, so there is nothing to add. A time limit is an accident of the setup: the drone was fine, the clock ran out. Treating it like a crash would teach the critic that states near the end of an episode are worth less, which is false. Two tests show the difference: with a time limit the critic learns the true value (10); with a real ending it learns the short return (about 2.7).

4. **How does the hover reward make the drone hover, and why did 10 million steps fall short while 50 million worked?**
   Each step pays for getting closer to the target (progress), and charges for distance, for spinning, for changing the action, and a one-off penalty for crashing. Flying off or crashing costs far more than anything else, so the policy first learns to survive; that took the first few million steps. The distance penalty is small (0.05 per metre), so a drone 0.12 m away pays only 0.006 per step, which is comparable to the smoothness penalties. At 10 million steps the policy had not yet squeezed out the last centimetres; with more training the same reward pushed it to about a millimetre. A stronger distance penalty would probably get there sooner, but it was not needed, so I did not change the handoff's values.

5. **How do we know the hover result is real and not a bug in the evaluation?**
   Five independent pieces of evidence. The evaluation scores a hand-written controller 200 of 200 and the same controller aimed 0.3 m off 0 of 200, so it scores both ways. A second evaluation using the simulator's true positions on a new seed agrees. Three different training seeds all pass. The environment's observation and reward are recomputed independently in NumPy in the tests. And 42 deliberately broken versions of the PPO and the environment were all caught by the tests, so a wrong environment would not slip through unnoticed.
