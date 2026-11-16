"""Milestone 3: the Python side of the hover task: the `QuadVecEnv` wrapper, the training script's
settings, and the hover evaluation (python/quadrace/evaluate.py).

The evaluation is tested with hand-written policies whose result is known in advance: a classical
controller that really does hover, the same controller aimed at the wrong point, and a policy that
crashes at once. That checks the scoring rule itself, independently of any learning.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from helpers import CrashPolicy, PDHover
from quadrace import ppo
from quadrace.evaluate import HOVER_PASS_RATE, evaluate_hover
from quadrace.train import build_ppo_config
from quadrace.vec_env import QuadVecEnv

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def drone_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "drone.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def task_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "task_hover.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# The hover evaluation
# --------------------------------------------------------------------------


def test_a_classical_hover_controller_passes_the_evaluation_so_the_exit_test_is_achievable(
    drone_cfg: dict, task_cfg: dict
) -> None:
    result = evaluate_hover(PDHover(drone_cfg), drone_cfg, task_cfg, episodes=100)
    assert result["pass_rate"] == 1.0 and result["meets_exit_test"]
    assert result["terminated_early"] == 0 and result["survivors_outside_tolerance"] == 0
    assert result["worst_distance_in_final_window_m"]["max_of_survivors"] < 0.01


def test_a_controller_aimed_0_3_m_off_survives_but_fails_the_0_1_m_tolerance(drone_cfg: dict, task_cfg: dict) -> None:
    result = evaluate_hover(PDHover(drone_cfg, target_offset=(0.3, 0.0, 0.0)), drone_cfg, task_cfg, episodes=50)
    assert result["pass_rate"] == 0.0 and not result["meets_exit_test"]
    assert result["terminated_early"] == 0 and result["survivors_outside_tolerance"] == 50
    assert result["worst_distance_in_final_window_m"]["median_of_survivors"] == pytest.approx(0.3, abs=0.01)


def test_a_controller_aimed_0_05_m_off_still_passes_because_the_tolerance_is_0_1_m(drone_cfg: dict, task_cfg: dict) -> None:
    result = evaluate_hover(PDHover(drone_cfg, target_offset=(0.05, 0.0, 0.0)), drone_cfg, task_cfg, episodes=50)
    assert result["pass_rate"] == 1.0


def test_a_policy_that_crashes_fails_every_episode_by_hitting_the_ground(drone_cfg: dict, task_cfg: dict) -> None:
    result = evaluate_hover(CrashPolicy(), drone_cfg, task_cfg, episodes=40)
    assert result["pass_rate"] == 0.0 and result["terminated_early"] == 40
    assert set(result["terminated_by"]) <= {"altitude", "out_of_bounds"}
    assert result["terminated_by"].get("altitude", 0) > 0


def test_the_evaluation_counts_exactly_the_requested_number_of_episodes(drone_cfg: dict, task_cfg: dict) -> None:
    result = evaluate_hover(PDHover(drone_cfg), drone_cfg, task_cfg, episodes=37)
    assert result["episodes"] == 37 and result["passed"] == 37


def test_the_evaluation_is_repeatable_for_the_same_seed(drone_cfg: dict, task_cfg: dict) -> None:
    a = evaluate_hover(PDHover(drone_cfg, target_offset=(0.1, 0.0, 0.0)), drone_cfg, task_cfg, episodes=30, seed=5)
    b = evaluate_hover(PDHover(drone_cfg, target_offset=(0.1, 0.0, 0.0)), drone_cfg, task_cfg, episodes=30, seed=5)
    assert a == b


def test_the_pass_rate_needed_is_95_percent() -> None:
    assert HOVER_PASS_RATE == 0.95


def test_a_drone_that_is_inside_the_tolerance_for_less_than_the_whole_window_fails(drone_cfg: dict, task_cfg: dict) -> None:
    # Aimed 0.3 m off for the whole episode is outside the tolerance for the whole window. Shrink
    # the window to the very last step and the same controller must still fail (0.3 m > 0.1 m),
    # while a tolerance of 0.5 m lets it pass: the tolerance and the window are both really used.
    policy = PDHover(drone_cfg, target_offset=(0.3, 0.0, 0.0))
    narrow = evaluate_hover(policy, drone_cfg, task_cfg, episodes=20, window_seconds=0.02)
    loose = evaluate_hover(policy, drone_cfg, task_cfg, episodes=20, tolerance=0.5)
    assert narrow["pass_rate"] == 0.0 and loose["pass_rate"] == 1.0


# --------------------------------------------------------------------------
# QuadVecEnv: the simulator as a VecEnv for PPO
# --------------------------------------------------------------------------


def test_quad_vec_env_follows_the_vec_env_interface(drone_cfg: dict, task_cfg: dict) -> None:
    env = QuadVecEnv(6, drone_cfg, task_cfg, seed=3)
    assert (env.num_envs, env.obs_dim, env.act_dim, env.n_actions) == (6, 22, 4, None)
    assert env.reward_term_names == ("progress", "distance_penalty", "rate_penalty", "action_penalty", "crash")
    obs = env.reset(seed=3)
    assert obs.shape == (6, 22) and obs.dtype == np.float32
    obs, reward, terminated, truncated, final_obs = env.step(np.zeros((6, 4), np.float64))  # float64 is accepted
    assert obs.shape == final_obs.shape == (6, 22)
    assert reward.dtype == np.float64 and reward.shape == (6,)
    assert terminated.dtype == truncated.dtype == np.bool_
    assert set(env.last_info) == {"reward_terms", "termination_cause"}


def test_quad_vec_env_reset_with_a_seed_is_repeatable(drone_cfg: dict, task_cfg: dict) -> None:
    env = QuadVecEnv(4, drone_cfg, task_cfg, seed=1)
    first = env.reset(seed=8)
    env.reset(seed=9)
    np.testing.assert_array_equal(env.reset(seed=8), first)


# --------------------------------------------------------------------------
# The training script's settings
# --------------------------------------------------------------------------


def test_ppo_yaml_builds_the_handoff_defaults() -> None:
    with open(REPO_ROOT / "configs" / "ppo.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config = build_ppo_config(cfg["ppo"], {})
    default = ppo.PPOConfig(total_timesteps=config.total_timesteps)
    assert config == default  # configs/ppo.yaml holds exactly the handoff's starting values
    assert cfg["env"]["num_envs"] == 2048


def test_command_line_overrides_replace_yaml_values_and_none_leaves_them() -> None:
    config = build_ppo_config({"total_timesteps": 1000, "seed": 4, "hidden_sizes": [32, 32]}, {"seed": 9, "device": None})
    assert config.seed == 9 and config.total_timesteps == 1000
    assert config.hidden_sizes == (32, 32)  # a list in the YAML becomes the tuple PPOConfig expects
    assert config.device == "auto"


def test_an_unknown_ppo_setting_in_the_yaml_is_an_error_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="unknown PPO setting"):
        build_ppo_config({"total_timesteps": 1000, "learning_rte": 1e-3}, {})


# --------------------------------------------------------------------------
# A tiny end-to-end run
# --------------------------------------------------------------------------


def test_a_tiny_training_run_on_the_hover_task_logs_reward_terms_and_writes_loadable_checkpoints(
    drone_cfg: dict, task_cfg: dict, tmp_path: Path
) -> None:
    env = QuadVecEnv(16, drone_cfg, task_cfg, seed=2)
    config = ppo.PPOConfig(
        total_timesteps=16 * 32 * 3, rollout_len=32, num_minibatches=4, update_epochs=2,
        hidden_sizes=(16, 16), seed=2, device="cpu", save_every_updates=3,
    )
    result = ppo.train(env, config, run_dir=tmp_path)

    last = result.history[-1]
    for name in env.reward_term_names:
        assert f"reward_terms/{name}" in last
    assert last["reward_terms/distance_penalty"] < 0  # always paying a distance penalty away from the target
    assert abs(sum(last[f"reward_terms/{n}"] for n in env.reward_term_names)) < 50  # sensible scale
    assert "episode_end_share/altitude" in last or "episode_end_share/out_of_bounds" in last  # untrained: it crashes
    assert (tmp_path / "checkpoint_000003.pt").is_file()

    policy, info = ppo.load_checkpoint(tmp_path / "checkpoint_000003.pt")
    assert info["config"]["rollout_len"] == 32
    scores = evaluate_hover(policy, drone_cfg, task_cfg, episodes=10, seed=1)
    assert scores["episodes"] == 10 and scores["pass_rate"] == 0.0  # three updates cannot learn to hover
