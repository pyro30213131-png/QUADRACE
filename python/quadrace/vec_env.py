"""The interface between PPO and an environment made of many copies, plus a gymnasium adapter.

PPO (`ppo.py`) only talks to the small `VecEnv` interface below. It is the same interface the
Rust simulator will offer (handoff section 5), so the same PPO code trains CartPole and the
drone. The rules, which both sides must follow:

- `reset()` returns the first observations, shape `[N, obs_dim]`, float32.
- `step(actions)` returns `(obs, reward, terminated, truncated, final_obs)`:
  - `obs` is what the agent sees next. For an environment whose episode just ended it is
    already the first observation of its NEXT episode (finished environments restart inside
    `step`, the same as `QuadSim.step`).
  - `final_obs` holds the last observation of the episode that just ended, for those
    environments only (other rows are copies of `obs` and must not be used). PPO needs it to
    bootstrap the value after a time-limit cut-off.
  - `terminated` means the episode ended because of the task (a crash, a fall). `truncated`
    means it was cut off (time limit). The two are never both used for the same meaning:
    PPO bootstraps from `final_obs` on truncation and does not on termination.
- Actions are float32 `[N, act_dim]` (continuous) or int64 `[N]` (discrete).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VecEnv(Protocol):
    """What PPO needs from an environment made of `num_envs` copies."""

    num_envs: int
    obs_dim: int
    act_dim: int  # number of continuous action values (1 for a discrete action)
    n_actions: int | None  # number of choices if the action is discrete, else None

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class GymVecEnv:
    """`num_envs` copies of one gymnasium environment, stepped one after another.

    Used for the CartPole and Pendulum checks. It is deliberately plain: a Python loop over
    the copies, with finished episodes restarted by hand so that `final_obs` is available.
    Continuous actions are clipped to the environment's action bounds, as the simulator does.
    """

    def __init__(self, env_id: str, num_envs: int) -> None:
        import gymnasium as gym  # imported here so vec_env.py loads without gymnasium installed

        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.envs = [gym.make(env_id) for _ in range(num_envs)]
        self.num_envs = num_envs
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        if len(obs_space.shape) != 1:
            raise ValueError(f"only flat observations are supported, got shape {obs_space.shape}")
        self.obs_dim = int(obs_space.shape[0])
        if isinstance(act_space, gym.spaces.Discrete):
            self.n_actions: int | None = int(act_space.n)
            self.act_dim = 1
            self._low = self._high = None
        elif isinstance(act_space, gym.spaces.Box):
            self.n_actions = None
            self.act_dim = int(act_space.shape[0])
            self._low = act_space.low
            self._high = act_space.high
        else:
            raise ValueError(f"unsupported action space {act_space}")

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            first_obs, _ = env.reset(seed=None if seed is None else seed + i)
            obs[i] = first_obs
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
        reward = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            if self.n_actions is not None:
                action = int(actions[i])
            else:
                action = np.clip(actions[i], self._low, self._high)
            next_obs, reward[i], terminated[i], truncated[i], _ = env.step(action)
            if terminated[i] or truncated[i]:
                final_obs[i] = next_obs
                next_obs, _ = env.reset()
            else:
                final_obs[i] = next_obs
            obs[i] = next_obs
        return obs, reward, terminated, truncated, final_obs
