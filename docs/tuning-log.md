# Tuning log

Every change to a reward term, a PPO setting or a controller gain goes here, with the reason (CLAUDE.md). The drone's starting values are the handoff's (sections 7, 8 and 9) until an entry below says otherwise.

## 2026-11-15: settings for the CartPole and Pendulum cross-checks (milestone 3)

These are **not** tuning of the drone. They are the settings used to check `ppo.py` against Stable-Baselines3 on two classic problems, listed here so the results in `docs/milestone-3.md` can be reproduced. The drone keeps the handoff's defaults in `PPOConfig`.

| Setting | CartPole-v1 | Pendulum-v1 | Drone (handoff default) | Why they differ |
| --- | --- | --- | --- | --- |
| Environments | 4 | 8 | 2,048 | The small problems need far less data per update. |
| Rollout length | 128 | 256 | 64 | CleanRL's CartPole value; a longer rollout for Pendulum's slow swing-up. |
| Minibatches, epochs | 4, 4 | 8, 10 | 8, 5 | CleanRL's CartPole values; the SB3 zoo's epochs for Pendulum. |
| Discount (gamma) | 0.99 | 0.9 | 0.99 | The SB3 zoo uses a short horizon for Pendulum, which helps it learn the swing-up. |
| GAE lambda | 0.95 | 0.95 | 0.95 | Same. |
| Learning rate | 2.5e-4 | 1e-3 | 3e-4 | CleanRL's CartPole value; the SB3 zoo's for Pendulum. Both decay linearly to zero. |
| Entropy bonus | 0.01 | 0.0 | 0.001 | CleanRL's CartPole value; the SB3 zoo uses none for Pendulum. |
| Gradient clip | 0.5 | 0.5 | 1.0 | Stable-Baselines3's default, so both implementations match. |
| Network | 2 x 64, tanh | 2 x 64, tanh | 2 x 256, tanh | Stable-Baselines3's default size, so both implementations match. |
| Initial log std | -0.5 | -0.5 | -0.5 | Same as the handoff, applied to both implementations. |
| Observation normalization | off | on | on | CleanRL does not normalize CartPole; Pendulum's observations vary in scale. |
| Reward normalization | off | on | off | Pendulum's rewards reach -16 per step, which makes the critic's targets large. The handoff does not call for reward normalization; it is an option in `ppo.py`, default off. |
| Budget | 500,000 steps | 1,000,000 steps | n/a | Enough for both to learn with margin. |

Two implementation choices that follow Stable-Baselines3 so the comparison is fair, and apply to the drone too:

- **Value loss weight.** The loss is `policy_loss - ent_coef * entropy + vf_coef * MSE(value, return)` with `vf_coef = 0.5` and a plain mean squared error (no extra 0.5 factor, no value clipping). The handoff does not specify this.
- **Adam epsilon 1e-5** instead of PyTorch's 1e-8, as in CleanRL and Stable-Baselines3.

## 2026-11-15: hover training (milestone 3)

Settings used for the hover runs that pass the exit test (`docs/milestone-3.md`): **all the handoff's starting values, unchanged.**

- PPO (`configs/ppo.yaml`): 2,048 environments, rollout 64, 8 minibatches, 5 epochs, gamma 0.99, GAE lambda 0.95, clip 0.2, learning rate 3e-4 decaying linearly to zero, entropy bonus 0.001, gradient clip 1.0, separate actor and critic of 2 x 256 tanh units, initial log std -0.5, observation normalization on, reward normalization off.
- Hover task (`configs/task_hover.yaml`): reward weights progress 1.0 per metre, distance 0.05, body rate 0.0002, action change 0.002, crash 5.0; the reset ranges and episode ends of handoff section 8.

| Change | Before | After | Reason and result |
| --- | --- | --- | --- |
| Training budget (`total_timesteps`) | 10,027,008 (76 updates) | 49,938,432 (381 updates, run with `--total-timesteps 50003968`) | After 10 million steps the policy survived (98% of episodes reached the time limit) but settled 0.12 to 0.14 m from the target, so 0 of 200 episodes passed the 0.1 m test. The same settings trained 5 times longer reached about a millimetre on all three seeds (199, 200 and 200 of 200 pass). |

Not changed, but considered: a larger distance weight (0.05 per metre is small next to the smoothness penalties, which is why precision came slowly). It was not needed, so the handoff's value stays.

No reward term, PPO setting or controller gain has been changed from the handoff's values.
