# Milestone 3: PPO and hover (in progress)

> **IN PROGRESS. Part 1 is done locally: the hand-written PPO is built and checked against Stable-Baselines3 on CartPole and Pendulum. Part 2 is not started: the hover task in the Rust simulator, and the hover evaluation that is this milestone's exit test.** CI has not run on part 1 because it is not pushed. This page is final when the exit test passes. See `docs/STATUS.md`.

Milestone 3 exit test (handoff sections 8 and 10): over 200 episodes with a deterministic policy, in at least 95% of them the drone stays within 0.1 m of the hover target for the final 10 s.

## What is built so far (part 1)

- **`python/quadrace/ppo.py`**: PPO written by hand in PyTorch. The training loop, the actor and critic networks, GAE, the clipped loss, observation (and optional reward) normalization, checkpoints, and evaluation.
- **`python/quadrace/vec_env.py`**: the small `VecEnv` interface PPO trains against (the same one the Rust simulator will offer), and an adapter that makes copies of a gymnasium environment fit it.
- **`python/quadrace/ppo_crosscheck.py`**: runs our PPO and Stable-Baselines3 side by side on CartPole and Pendulum with identical settings and seeds.
- **`tests/test_ppo.py`**: 44 tests.
- **`docs/tuning-log.md`**: the settings used for the checks.
- PyTorch 2.14.1 with the CUDA 12.6 runtime (a 2.42 GB download from `download.pytorch.org`), TensorBoard, gymnasium and Stable-Baselines3 are installed in `.venv`. CI installs the small CPU-only PyTorch build.

## How to run it

```powershell
.venv\Scripts\Activate.ps1
pytest tests/test_ppo.py -q                                   # the PPO tests (about 45 s)
python -m quadrace.ppo_crosscheck --env CartPole-v1 --seeds 1 2 3    # ours versus Stable-Baselines3
python -m quadrace.ppo_crosscheck --env Pendulum-v1 --seeds 1 2 3    # (a few minutes each)
```

To install PyTorch for your GPU (what was done here; your NVIDIA driver must support CUDA 12.6 or newer, and `nvidia-smi` shows it):

```powershell
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install tensorboard gymnasium stable-baselines3
```

## How the code is organised

| Part of `ppo.py` | What it does |
| --- | --- |
| `PPOConfig` | Every PPO setting, defaulting to the handoff's starting values for the drone. |
| `RunningMeanStd`, `RewardScaler` | Running statistics for normalizing observations and (optionally) rewards. |
| `Agent` | The actor and the critic: two separate networks, tanh hidden units, orthogonal initial weights. The actor outputs the mean of a Gaussian (or the logits of a categorical choice for CartPole). |
| `bootstrap_truncated_rewards`, `compute_gae` | The maths that turns rewards into advantages: time-limit bootstrapping, then GAE. |
| `ppo_loss`, `apply_update` | The clipped loss for one minibatch, and one gradient step. |
| `train` | The loop: collect experience, score it, improve, record, checkpoint. |
| `save_checkpoint`, `load_checkpoint`, `Policy` | Checkpoints that carry the observation statistics, config, git commit and seed; a loaded policy flies from raw observations. |
| `evaluate` | Runs a policy deterministically for a number of episodes. |

## Choices worth knowing

1. **Tanh is inside the network, not on the output.** The hidden units use tanh. Continuous actions are sampled from a Gaussian and are not squashed, so the log-probability is for the sampled value. The environment clips the action to its limits, as CleanRL and Stable-Baselines3 do. A tanh-squashed output (as in SAC) is an alternative if actions turn out to get stuck at the limits; the handoff does not decide this.
2. **Time-limit cut-offs are bootstrapped from `final_obs`; real endings are not.** Two tests check this against values worked out by hand (see Results).
3. **The value loss is a plain mean squared error with weight 0.5, with no value clipping.** The handoff does not say; this matches Stable-Baselines3 so the cross-check is fair.
4. **Reward normalization exists but is off by default.** Pendulum's large negative rewards needed it. The drone's rewards are small, so the handoff's default is kept.
5. **Checkpoints are loaded with `weights_only=True`**, so opening a checkpoint file cannot run code.
6. **The checks ran on the CPU.** The networks are tiny. For the drone-sized update the GPU is much faster (see Results), so training the drone will use it.
7. **`ppo.py` knows nothing about drones.** It only talks to the `VecEnv` interface. The hover task and gates plug in through the Rust simulator in later work.

## Results

Measured on 2026-11-15 on the owner's machine (Windows 11, Intel i5-13450HX, RTX 4050, PyTorch 2.14.1).

| Check | Result |
| --- | --- |
| `pytest tests/test_ppo.py -q` | 44 passed |
| `pytest -q` (everything) | 115 passed in 101 s |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test -p quadsim` | pass, 38 Rust tests |
| Critic's value after training on "reward 1 forever, cut off by a time limit every 5 steps" (true value 10, gamma 0.9) | learns 10 within 0.5, as the test requires |
| The same, but episodes really end after 5 steps (hand-worked answer about 2.7) | learns 2.7 within 0.5 |
| Mutation check: 18 deliberate bugs in `ppo.py` | all 18 caught (17 on the first pass; see below) |

### Cross-check against Stable-Baselines3

Same settings, same number of environments, same seeds (1, 2, 3), same budget; the full settings are in `docs/tuning-log.md`. Numbers are the mean and standard deviation across the 3 seeds. "Return at x%" is the mean return of the last 100 training episodes at that fraction of the budget. "Deterministic eval" is 50 episodes with the noise switched off.

**CartPole-v1** (500,000 steps; the best possible return is 500; "learned" means a mean of at least 475):

| | Ours | Stable-Baselines3 |
| --- | --- | --- |
| Return at 25% | 363.9 +- 66.8 | 365.4 +- 20.3 |
| Return at 50% | 363.8 +- 98.2 | 460.6 +- 28.3 |
| Return at 75% | 424.8 +- 61.5 | 378.3 +- 156.0 |
| Return at 100% | 426.9 +- 92.1 | 340.5 +- 118.7 |
| Deterministic eval | 414.3 +- 100.4 | 355.2 +- 103.7 |
| Seeds reaching 475 | 3 of 3 | 3 of 3 |
| Mean steps to reach 475 | 208,725 | 194,560 |

**Pendulum-v1** (1,000,000 steps; returns are negative, closer to zero is better; "learned" means a mean of at least -250):

| | Ours | Stable-Baselines3 |
| --- | --- | --- |
| Return at 25% | -191.1 +- 29.2 | -169.8 +- 6.0 |
| Return at 50% | -245.7 +- 17.8 | -183.7 +- 3.1 |
| Return at 75% | -187.2 +- 5.5 | -169.7 +- 10.9 |
| Return at 100% | -214.5 +- 48.5 | -180.8 +- 15.0 |
| Deterministic eval | -193.8 +- 37.6 | -185.2 +- 30.3 |
| Seeds reaching -250 | 3 of 3 | 3 of 3 |
| Mean steps to reach -250 | 103,083 | 92,160 |

How to read these:

- **Both implementations learn both problems on every seed, at about the same speed.** The differences are inside the seed-to-seed spread, which is large with only 3 seeds. For example, the final CartPole evaluations were 273, 470 and 500 for ours and 500, 263 and 303 for Stable-Baselines3.
- **On Pendulum, Stable-Baselines3's training curve looks somewhat steadier** (-181 against -215 at the end, with a smaller spread). With 3 seeds I cannot tell whether that is real or noise. The deterministic evaluations are closer (-194 against -185).
- **This is evidence that `ppo.py` is not broken, not proof that it is identical.** The tests carry the rest of the weight.
- Run times (about 90 to 200 s per run on the CPU, about the same for both) were measured with three jobs sharing the machine, so treat them as rough.
- The cross-checks were started before two small behaviour-preserving refactors of `ppo.py` (moving the loss and the gradient step into their own functions so they can be tested). I checked that a 20,480-step training run gives the bit-for-bit same final weights before and after each refactor.

### Do the tests catch a wrong PPO? (mutation check)

As in the earlier milestones, I broke `ppo.py` in 18 different ways, one at a time (GAE ignoring episode ends, no truncation bootstrap, bootstrapping from the wrong observation, no clipping of the ratio, `min` instead of `max`, a flipped policy-loss or entropy sign, a critic that does not train, observation statistics not updated, unshuffled minibatches, no gradient clipping, and others) and ran the tests each time. 17 were caught on the first pass. The 18th, removing gradient clipping, was not, so I moved that step into its own function (`apply_update`) and added tests for it. All 18 are now caught, and the file was restored from git after each.

### GPU or CPU for the drone

How long one PPO update takes at the drone's planned size (131,072 samples = 2,048 environments x 64 steps; 8 minibatches x 5 epochs; two 256-unit networks), on random data with no simulator:

| Where | One update |
| --- | --- |
| RTX 4050 | 284 ms |
| CPU, 10 threads | 2,586 ms |
| CPU, 1 thread | 12,474 ms |
| For comparison: simulating the same 131,072 steps at the measured 4.8 million steps per second | about 27 ms |

So at the drone's batch size the GPU is about 9 times faster than the CPU, and the network update, not the simulator, is the slow part. That is a measurement of the update alone; the real training loop also needs the policy's forward pass at every step and CPU-to-GPU copies, which I have not measured. For CartPole and Pendulum the CPU is fine.

## What is not done yet (part 2)

- The **hover task** in the Rust simulator: the 22-value observation, the 5-term reward, the termination and truncation rules, random resets with a seeded generator per environment, and the full `reset()` / `step()` interface of handoff section 5 (this also closes the API gap noted in `docs/STATUS.md`). It needs the `rand` crate, which is approved in CLAUDE.md.
- A Python wrapper that makes `QuadSim` a `VecEnv` for `ppo.py`.
- Per-term reward logging in TensorBoard (the five `info` columns).
- The first training runs on the hover task, and the **hover evaluation**: 200 deterministic episodes, pass at 95%.
- Extending the determinism tests to cover the random resets.

## Five questions you should be able to answer

1. **What does the "clip" in PPO do, and why is it there?**
   After each batch of experience PPO changes the policy over several passes through the same data. The clip says: once an action's probability has changed by more than about 20% from what it was when the data was collected, stop giving extra credit for pushing a good action's probability further up (or a bad action's further down). It only ever removes credit, never a penalty: a bad action whose probability went up is still punished in full. That keeps each update small, so one noisy batch cannot wreck a policy that was working. Tests check four cases: a good action past the limit, a bad action past the limit, a bad action whose probability went up, and an action inside the limit.

2. **What are the critic and the advantage for?**
   The critic estimates how much total future reward to expect from a state. The advantage is roughly "what actually happened minus what the critic expected". Using it makes the learning signal much less noisy than raw returns: an action is rewarded for doing better than expected, not for happening in a lucky state. GAE blends one-step and long-horizon estimates; lambda 0.95 is the handoff's trade-off between bias and noise.

3. **Why is the value bootstrapped on a time-limit cut-off but not on a crash?**
   A crash really ends the future, so there is nothing to add. A time limit is an accident of the setup: the drone was fine, the clock ran out. If we treated it like a crash, the critic would learn that states near the end of an episode are worth less, which is false. The two tests show the difference: with a time limit the critic learns the true value (10); with a real ending it learns the short return (about 2.7).

4. **Why normalize observations, and why save the statistics with every checkpoint?**
   Observations have very different sizes (velocities, rotation-matrix entries, positions), and networks learn better when inputs are around zero with spread near 1. The normalization uses the mean and variance seen so far. A policy trained on normalized inputs is meaningless on raw ones, so the statistics are part of the policy and must be saved with it.

5. **Why did we test on CartPole and Pendulum before touching the drone?**
   If PPO fails to learn to hover, there are two possible reasons: the task or reward is wrong, or PPO has a bug. CartPole (discrete actions) and Pendulum (continuous actions, like the drone) are small problems with known good scores, and Stable-Baselines3 is a trusted implementation. Matching it on both rules out most PPO bugs, so a later failure points at the task.
