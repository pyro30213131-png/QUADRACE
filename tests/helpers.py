"""Policies written by hand for tests: no learning, so their behaviour is known in advance.

Each has the same `act(obs, deterministic=True)` method as a trained `ppo.Policy`, so they can be
used wherever a policy is expected (the hover evaluation, the viewer).
"""

import numpy as np


class PDHover:
    """A classical hover controller that reads the 22-value observation and returns an action.

    A PD law on position and velocity gives the acceleration the drone should have. Its direction
    is the thrust axis to aim for, and a P-term on the attitude error turns that into body-rate
    commands; the thrust follows from the acceleration. No learning involved. `target_offset`
    deliberately aims it at the wrong point.
    """

    def __init__(self, drone_cfg: dict, target_offset=(0.0, 0.0, 0.0), kp=6.0, kd=4.9, k_att=8.0) -> None:
        self.kp, self.kd, self.k_att = kp, kd, k_att
        self.offset = np.array(target_offset)
        self.mass, self.gravity = drone_cfg["body"]["mass"], drone_cfg["world"]["gravity"]
        self.thrust_max = drone_cfg["motors"]["thrust_max"]
        self.max_rates = np.array(drone_cfg["limits"]["max_rates"])

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        n = len(obs)
        vel_b, rot, target_b = obs[:, 0:3], obs[:, 3:12].reshape(n, 3, 3), obs[:, 15:18]
        target_rel_w = np.einsum("nij,nj->ni", rot, target_b) + self.offset  # target minus position, world frame
        vel_w = np.einsum("nij,nj->ni", rot, vel_b)
        wanted_accel = self.kp * target_rel_w - self.kd * vel_w + np.array([0.0, 0.0, self.gravity])
        thrust = np.maximum(self.mass * np.sum(wanted_accel * rot[:, :, 2], axis=1), 0.0)
        wanted_axis_w = wanted_accel / np.linalg.norm(wanted_accel, axis=1, keepdims=True)
        wanted_axis_b = np.einsum("nji,nj->ni", rot, wanted_axis_w)  # the thrust axis to aim for, in the body frame
        rate_cmd = np.stack([-wanted_axis_b[:, 1], wanted_axis_b[:, 0], np.zeros(n)], axis=1) * self.k_att
        action = np.zeros((n, 4), np.float32)
        action[:, 0] = 2.0 * thrust / (4.0 * self.thrust_max) - 1.0
        action[:, 1:] = rate_cmd / self.max_rates
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class CrashPolicy:
    """Motors off, always: the drone falls."""

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.tile(np.array([-1.0, 0.0, 0.0, 0.0], np.float32), (len(obs), 1))
