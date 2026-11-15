"""Evaluate a trained policy (milestone 3: the hover evaluation).

    python -m quadrace.evaluate --checkpoint runs/hover1/checkpoint_000076.pt

Hover evaluation (handoff section 8): 200 episodes, the policy acting deterministically (no
action noise). An episode passes if the drone survives to the 15 s time limit AND stays within
0.1 m of the target at every policy step of the final 10 s. The milestone-3 exit test is that at
least 95% of the episodes pass.

The evaluation episodes start from the task's random reset distribution with a seed that is
different from training (default 10,000), so the policy has not seen these exact episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from quadrace import ppo
from quadrace.vec_env import TERMINATION_CAUSE_NAMES, QuadVecEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
HOVER_TOLERANCE_M = 0.1  # the drone must stay this close to the target ...
HOVER_WINDOW_S = 10.0  # ... for this long at the end of the episode
HOVER_PASS_RATE = 0.95  # and at least this share of episodes must pass


def evaluate_hover(
    policy: ppo.Policy,
    drone_cfg: dict,
    task_cfg: dict,
    episodes: int = 200,
    seed: int = 10_000,
    tolerance: float = HOVER_TOLERANCE_M,
    window_seconds: float = HOVER_WINDOW_S,
) -> dict:
    """Run `episodes` hover episodes with the deterministic policy and score them.

    All episodes are the same length (the time limit), so one drone per episode runs in lockstep
    for the whole episode. A drone that terminates early (crash, flew away) has failed; what it
    does after its automatic restart is ignored.
    """
    policy_hz = drone_cfg["timing"]["policy_hz"]
    max_steps = round(task_cfg["episode"]["max_seconds"] * policy_hz)
    window_steps = round(window_seconds * policy_hz)
    env = QuadVecEnv(episodes, drone_cfg, task_cfg, seed=seed)
    obs = env.reset(seed=seed)

    terminated_early = np.zeros(episodes, dtype=bool)
    cause = np.zeros(episodes, dtype=np.int64)
    worst_in_window = np.zeros(episodes)  # largest distance from the target during the final window
    for step in range(1, max_steps + 1):
        obs, _, terminated, truncated, final_obs = env.step(policy.act(obs, deterministic=True))
        newly_failed = terminated & ~terminated_early
        cause[newly_failed] = env.last_info["termination_cause"][newly_failed]
        terminated_early |= terminated
        # Distance to the target is the length of the target vector in the observation (indices 15..17).
        # For a drone that just finished, final_obs holds the observation of the step just taken.
        seen = np.where((terminated | truncated)[:, None], final_obs, obs)
        distance = np.linalg.norm(seen[:, 15:18], axis=1)
        if step > max_steps - window_steps:
            worst_in_window = np.maximum(worst_in_window, distance)

    survived = ~terminated_early
    passed = survived & (worst_in_window <= tolerance)
    result = {
        "episodes": episodes,
        "seed": seed,
        "pass_rate": float(passed.mean()),
        "passed": int(passed.sum()),
        "required_pass_rate": HOVER_PASS_RATE,
        "meets_exit_test": bool(passed.mean() >= HOVER_PASS_RATE),
        "terminated_early": int(terminated_early.sum()),
        "terminated_by": {
            name: int((cause == code).sum()) for code, name in TERMINATION_CAUSE_NAMES.items() if (cause == code).any()
        },
        "survivors_outside_tolerance": int((survived & ~passed).sum()),
        "worst_distance_in_final_window_m": {
            "median_of_survivors": float(np.median(worst_in_window[survived])) if survived.any() else float("nan"),
            "max_of_survivors": float(worst_in_window[survived].max()) if survived.any() else float("nan"),
        },
        "tolerance_m": tolerance,
        "window_seconds": window_seconds,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on the hover task")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drone", type=Path, default=REPO_ROOT / "configs" / "drone.yaml")
    parser.add_argument("--task", type=Path, default=REPO_ROOT / "configs" / "task_hover.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="also write the result as JSON here")
    args = parser.parse_args()

    with open(args.drone, encoding="utf-8") as f:
        drone_cfg = yaml.safe_load(f)
    with open(args.task, encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)
    policy, info = ppo.load_checkpoint(args.checkpoint, device=args.device)
    result = evaluate_hover(policy, drone_cfg, task_cfg, args.episodes, args.seed)
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_update"] = info["update"]
    result["checkpoint_git"] = info["git"]

    print(f"hover evaluation of {args.checkpoint.name} (update {info['update']}), {result['episodes']} episodes, seed {result['seed']}")
    print(f"  pass rate: {result['passed']}/{result['episodes']} = {result['pass_rate']:.1%} "
          f"(need {HOVER_PASS_RATE:.0%}: {'MET' if result['meets_exit_test'] else 'not met'})")
    print(f"  ended early: {result['terminated_early']}  by cause: {result['terminated_by']}")
    print(f"  survived but outside {result['tolerance_m']} m in the final {result['window_seconds']:.0f} s: {result['survivors_outside_tolerance']}")
    spread = result["worst_distance_in_final_window_m"]
    print(f"  worst distance in the final window (survivors): median {spread['median_of_survivors']:.3f} m, max {spread['max_of_survivors']:.3f} m")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
