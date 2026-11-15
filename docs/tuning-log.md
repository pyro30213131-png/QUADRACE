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

No drone reward term, PPO setting or controller gain has been changed so far.
