"""Milestone 3: tests for the hover task of `quadsim.QuadSim` (observation, reward, episode ends,
resets, the reset()/step() interface of handoff section 5).

The observation and reward are recomputed here in NumPy from the simulator's own state (using the
reference simulator's rotation helpers), so these tests do not just repeat the Rust code.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

import quadsim
from quadrace import ref_sim
from quadrace.ref_sim import POS, QUAT, RATES, STATE_SIZE, THRUST, VEL

REPO_ROOT = Path(__file__).resolve().parent.parent
OBS_DIM = 22
TARGET = np.array([0.0, 0.0, 2.0])


@pytest.fixture(scope="module")
def drone_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "drone.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def task_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "task_hover.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def params() -> ref_sim.DroneParams:
    return ref_sim.load_drone_params(REPO_ROOT / "configs" / "drone.yaml")


def make_sim(drone_cfg: dict, task_cfg: dict, n: int = 8, seed: int = 1, **kwargs) -> quadsim.QuadSim:
    return quadsim.QuadSim(n, drone_cfg, task_cfg, seed=seed, **kwargs)


def hover_state_row(params: ref_sim.DroneParams, position: np.ndarray) -> np.ndarray:
    return ref_sim.hover_state(params, position)


def place(sim: quadsim.QuadSim, rows: np.ndarray) -> None:
    sim.set_state(rows)


def expected_obs(state: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """The 22-value hover observation from handoff section 8, computed in NumPy."""
    rot = ref_sim.quat_to_rotation_matrix(state[QUAT])  # body to world
    vel_b = rot.T @ state[VEL]
    target_b = rot.T @ (TARGET - state[POS])
    return np.concatenate([vel_b, rot.reshape(9), state[RATES], target_b, prev_action]).astype(np.float32)


# --------------------------------------------------------------------------
# The reset() / step() interface
# --------------------------------------------------------------------------


def test_obs_dim_is_22_for_the_hover_task(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg)
    assert sim.obs_dim == OBS_DIM and sim.task == "hover"
    assert sim.reward_term_names == ["progress", "distance_penalty", "rate_penalty", "action_penalty", "crash"]


def test_a_sim_built_without_a_task_has_no_obs_dim_and_refuses_reset_and_step(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(2, drone_cfg)
    assert sim.obs_dim is None and sim.task is None
    with pytest.raises(ValueError, match="no task"):
        sim.reset()
    with pytest.raises(ValueError, match="no task"):
        sim.step(np.zeros((2, 4), np.float32))


def test_reset_returns_float32_observations_of_the_right_shape(drone_cfg: dict, task_cfg: dict) -> None:
    obs = make_sim(drone_cfg, task_cfg, n=5).reset()
    assert obs.shape == (5, OBS_DIM) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_step_returns_obs_reward_terminated_truncated_final_obs_and_info(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=5)
    sim.reset()
    obs, reward, terminated, truncated, final_obs, info = sim.step(np.zeros((5, 4), np.float32))
    assert obs.shape == final_obs.shape == (5, OBS_DIM) and obs.dtype == final_obs.dtype == np.float32
    assert reward.shape == (5,) and reward.dtype == np.float32
    assert terminated.shape == truncated.shape == (5,) and terminated.dtype == truncated.dtype == np.bool_
    assert info["reward_terms"].shape == (5, 5) and info["reward_terms"].dtype == np.float32
    assert info["termination_cause"].shape == (5,) and info["termination_cause"].dtype == np.uint8


def test_step_returns_new_arrays_each_call(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=3)
    sim.reset()
    first = sim.step(np.zeros((3, 4), np.float32))
    kept = first[0].copy()
    sim.step(np.zeros((3, 4), np.float32))
    np.testing.assert_array_equal(first[0], kept)  # the earlier result was not overwritten


def test_the_reward_terms_add_up_to_the_reward(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=16)
    sim.reset()
    rng = np.random.default_rng(0)
    for _ in range(60):
        _, reward, _, _, _, info = sim.step(rng.uniform(-1, 1, (16, 4)).astype(np.float32))
        np.testing.assert_allclose(info["reward_terms"].sum(axis=1), reward, atol=1e-5)


def test_step_rejects_float64_wrong_shape_and_nan_actions(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=2)
    sim.reset()
    with pytest.raises(TypeError):
        sim.step(np.zeros((2, 4), np.float64))
    with pytest.raises(ValueError, match="shape"):
        sim.step(np.zeros((3, 4), np.float32))
    bad = np.zeros((2, 4), np.float32)
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sim.step(bad)


def test_step_accepts_a_strided_action_array(drone_cfg: dict, task_cfg: dict) -> None:
    wide = np.zeros((4, 8), np.float32)
    wide[:, ::2] = 0.25
    strided = wide[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    a, b = make_sim(drone_cfg, task_cfg, n=4, seed=3), make_sim(drone_cfg, task_cfg, n=4, seed=3)
    a.reset(), b.reset()
    ra, rb = a.step(strided), b.step(np.ascontiguousarray(strided))
    np.testing.assert_array_equal(ra[0], rb[0])
    np.testing.assert_array_equal(ra[1], rb[1])


def test_the_track_cfg_must_be_a_dict_with_task_hover_and_its_keys(drone_cfg: dict, task_cfg: dict) -> None:
    with pytest.raises(TypeError):
        quadsim.QuadSim(1, drone_cfg, "configs/task_hover.yaml")  # type: ignore[arg-type]
    racing = {**task_cfg, "task": "race"}
    with pytest.raises(ValueError, match="not available yet"):
        quadsim.QuadSim(1, drone_cfg, racing)
    missing = copy.deepcopy(task_cfg)
    del missing["reward"]["progress"]
    with pytest.raises(KeyError, match="reward.progress"):
        quadsim.QuadSim(1, drone_cfg, missing)
    no_target = {k: v for k, v in task_cfg.items() if k != "target"}
    with pytest.raises(KeyError, match="target"):
        quadsim.QuadSim(1, drone_cfg, no_target)
    wrong_type = copy.deepcopy(task_cfg)
    wrong_type["reward"]["crash"] = "lots"
    with pytest.raises(TypeError, match="reward.crash"):
        quadsim.QuadSim(1, drone_cfg, wrong_type)
    bad_value = copy.deepcopy(task_cfg)
    bad_value["reset"]["max_tilt"] = 3.0
    with pytest.raises(ValueError, match="max_tilt"):
        quadsim.QuadSim(1, drone_cfg, bad_value)


# --------------------------------------------------------------------------
# Observation and reward, recomputed independently
# --------------------------------------------------------------------------


def test_observation_matches_the_handoff_formula_for_random_states(drone_cfg: dict, task_cfg: dict, params) -> None:
    rng = np.random.default_rng(1)
    n = 12
    sim = make_sim(drone_cfg, task_cfg, n=n)
    sim.reset()
    states = np.zeros((n, STATE_SIZE))
    states[:, POS] = TARGET + rng.uniform(-1.0, 1.0, (n, 3))
    states[:, POS][:, 2] = np.maximum(states[:, POS][:, 2], 1.0)
    q = rng.normal(size=(n, 4))
    q[:, 0] = np.abs(q[:, 0]) + 2.0  # keep attitudes within a moderate tilt
    states[:, QUAT] = q / np.linalg.norm(q, axis=1, keepdims=True)
    states[:, VEL] = rng.normal(0, 0.5, (n, 3))
    states[:, RATES] = rng.normal(0, 0.5, (n, 3))
    states[:, THRUST] = params.mass * params.gravity / 4.0
    place(sim, states)
    action = rng.uniform(-1, 1, (n, 4)).astype(np.float32)

    obs, _, terminated, truncated, _, _ = sim.step(action)

    assert not terminated.any() and not truncated.any()
    after = sim.get_state()
    for i in range(n):
        np.testing.assert_allclose(obs[i], expected_obs(after[i], action[i].astype(np.float64)), atol=2e-6)


def test_the_first_observation_after_reset_has_a_zero_previous_action(drone_cfg: dict, task_cfg: dict) -> None:
    obs = make_sim(drone_cfg, task_cfg, n=6).reset()
    np.testing.assert_array_equal(obs[:, 18:22], 0.0)


def test_reset_observations_match_the_formula_for_the_reset_states(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=6)
    obs = sim.reset()
    states = sim.get_state()
    for i in range(6):
        np.testing.assert_allclose(obs[i], expected_obs(states[i], np.zeros(4)), atol=2e-6)


def test_reward_terms_match_the_handoff_formulas(drone_cfg: dict, task_cfg: dict, params) -> None:
    rng = np.random.default_rng(2)
    n = 10
    sim = make_sim(drone_cfg, task_cfg, n=n)
    sim.reset()
    before = np.zeros((n, STATE_SIZE))
    before[:, POS] = TARGET + rng.uniform(-1.0, 1.0, (n, 3))
    before[:, QUAT] = [1.0, 0.0, 0.0, 0.0]
    before[:, VEL] = rng.normal(0, 0.5, (n, 3))
    before[:, THRUST] = params.mass * params.gravity / 4.0
    place(sim, before)
    action = rng.uniform(-0.5, 0.5, (n, 4)).astype(np.float32)

    _, _, terminated, truncated, _, info = sim.step(action)

    assert not terminated.any() and not truncated.any()
    after = sim.get_state()
    reward = task_cfg["reward"]
    dist_before = np.linalg.norm(TARGET - before[:, POS], axis=1)
    dist_after = np.linalg.norm(TARGET - after[:, POS], axis=1)
    expected = np.stack(
        [
            reward["progress"] * (dist_before - dist_after),
            -reward["distance"] * dist_after,
            -reward["body_rate"] * np.sum(after[:, RATES] ** 2, axis=1),
            -reward["action_change"] * np.sum(action.astype(np.float64) ** 2, axis=1),  # previous action was 0
            np.zeros(n),
        ],
        axis=1,
    )
    np.testing.assert_allclose(info["reward_terms"], expected, atol=1e-5)


def test_the_action_change_penalty_uses_the_change_from_the_previous_action(drone_cfg: dict, task_cfg: dict, params) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=1)
    sim.reset()
    place(sim, hover_state_row(params, TARGET)[None])
    first = np.array([[0.3, 0.1, -0.2, 0.4]], np.float32)
    second = np.array([[0.1, 0.1, 0.0, -0.1]], np.float32)
    sim.step(first)
    _, _, _, _, _, info = sim.step(second)
    change_sq = float(np.sum((second.astype(np.float64) - first.astype(np.float64)) ** 2))
    assert info["reward_terms"][0, 3] == pytest.approx(-task_cfg["reward"]["action_change"] * change_sq, abs=1e-6)


# --------------------------------------------------------------------------
# Episode ends and automatic reset
# --------------------------------------------------------------------------


def test_hitting_the_ground_terminates_with_cause_3_and_a_crash_penalty(drone_cfg: dict, task_cfg: dict, params) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=1)
    sim.reset()
    low = np.zeros((1, STATE_SIZE))
    low[0, POS] = [0.0, 0.0, 0.3]
    low[0, QUAT] = [1.0, 0.0, 0.0, 0.0]  # motors off, falling
    place(sim, low)
    for _ in range(30):
        _, reward, terminated, truncated, _, info = sim.step(np.array([[-1.0, 0.0, 0.0, 0.0]], np.float32))
        if terminated[0]:
            break
    assert terminated[0] and not truncated[0]
    assert info["termination_cause"][0] == 3
    assert info["reward_terms"][0, 4] == pytest.approx(-task_cfg["reward"]["crash"])


def test_flying_far_from_the_target_terminates_with_cause_4(drone_cfg: dict, task_cfg: dict, params) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=1)
    sim.reset()
    far = hover_state_row(params, TARGET + np.array([5.5, 0.0, 0.0]))[None]
    place(sim, far)
    _, _, terminated, _, _, info = sim.step(ref_sim.hover_action(params)[None].astype(np.float32))
    assert terminated[0] and info["termination_cause"][0] == 4


def test_the_time_limit_truncates_at_750_steps_with_cause_6(drone_cfg: dict, task_cfg: dict, params) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=2)
    sim.reset()
    place(sim, np.tile(hover_state_row(params, TARGET), (2, 1)))
    action = np.tile(ref_sim.hover_action(params).astype(np.float32), (2, 1))
    for step in range(1, 751):
        _, _, terminated, truncated, _, info = sim.step(action)
        assert not terminated.any(), f"step {step}"
        assert truncated.all() == (step == 750) and truncated.any() == (step == 750)
    assert (info["termination_cause"] == 6).all()


def test_a_drone_whose_episode_ends_restarts_and_final_obs_keeps_its_last_observation(
    drone_cfg: dict, task_cfg: dict, params
) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=2)
    sim.reset()
    rows = np.tile(hover_state_row(params, TARGET), (2, 1))
    rows[1, POS] = TARGET + np.array([5.5, 0.0, 0.0])  # drone 1 starts too far away and ends at once
    place(sim, rows)
    action = np.tile(ref_sim.hover_action(params).astype(np.float32), (2, 1))

    obs, _, terminated, _, final_obs, _ = sim.step(action)

    assert terminated.tolist() == [False, True]
    np.testing.assert_array_equal(final_obs[0], obs[0])  # not finished: final_obs is a copy of obs
    state = sim.get_state()
    assert np.linalg.norm(state[1, POS] - TARGET) <= task_cfg["reset"]["max_offset"] + 1e-9  # restarted near the target
    np.testing.assert_array_equal(obs[1, 18:22], 0.0)  # a fresh episode: zero previous action
    assert np.linalg.norm(final_obs[1, 15:18]) > task_cfg["termination"]["max_distance"]  # last obs: far from target
    assert final_obs[1, 18] != 0.0  # ... and it carries the action just taken


# --------------------------------------------------------------------------
# Resets and seeds
# --------------------------------------------------------------------------


def test_reset_states_follow_the_ranges_in_the_config(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=2000, seed=11)
    sim.reset()
    s = sim.get_state()
    reset = task_cfg["reset"]
    assert np.linalg.norm(s[:, POS] - TARGET, axis=1).max() <= reset["max_offset"] + 1e-9
    assert s[:, POS][:, 2].min() >= reset["min_altitude"]
    tilt = np.arccos(np.clip(1.0 - 2.0 * (s[:, 4] ** 2 + s[:, 5] ** 2), -1.0, 1.0))  # angle of the body z axis from vertical
    assert tilt.max() <= reset["max_tilt"] + 1e-9 and tilt.max() > 0.9 * reset["max_tilt"]
    assert np.abs(s[:, 6]).max() == 0.0  # yaw starts at zero
    assert np.linalg.norm(s[:, VEL], axis=1).max() <= reset["max_speed"] + 1e-9
    assert np.linalg.norm(s[:, RATES], axis=1).max() <= reset["max_body_rate"] + 1e-9
    np.testing.assert_allclose(s[:, THRUST], drone_cfg["body"]["mass"] * drone_cfg["world"]["gravity"] / 4.0)


def test_reset_positions_cover_the_whole_ball_not_just_one_corner(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=4000, seed=12)
    sim.reset()
    offsets = sim.get_state()[:, POS] - TARGET
    assert abs(offsets[:, 0].mean()) < 0.06 and abs(offsets[:, 1].mean()) < 0.06
    # The floor at 0.5 m cuts the ball of radius 1.5 around z = 2 only at its very bottom.
    assert offsets[:, 0].min() < -1.3 and offsets[:, 0].max() > 1.3


def test_the_same_seed_gives_the_same_episodes(drone_cfg: dict, task_cfg: dict) -> None:
    a = make_sim(drone_cfg, task_cfg, n=16, seed=5).reset()
    b = make_sim(drone_cfg, task_cfg, n=16, seed=5).reset()
    c = make_sim(drone_cfg, task_cfg, n=16, seed=6).reset()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reset_with_a_seed_restarts_the_random_stream(drone_cfg: dict, task_cfg: dict) -> None:
    sim = make_sim(drone_cfg, task_cfg, n=8, seed=1)
    first = sim.reset(seed=99)
    sim.reset()
    again = sim.reset(seed=99)
    np.testing.assert_array_equal(first, again)


def test_different_environments_in_one_batch_get_different_episodes(drone_cfg: dict, task_cfg: dict) -> None:
    obs = make_sim(drone_cfg, task_cfg, n=8, seed=2).reset()
    assert len({tuple(row) for row in obs}) == 8


# --------------------------------------------------------------------------
# Results must not depend on the thread count, even with random resets
# --------------------------------------------------------------------------


def run_with_threads(drone_cfg: dict, task_cfg: dict, threads: int | None) -> tuple[bytes, int]:
    n = 256
    sim = make_sim(drone_cfg, task_cfg, n=n, seed=21, num_threads=threads)
    chunks = [sim.reset().tobytes()]
    rng = np.random.default_rng(4)
    episodes_ended = 0
    for _ in range(200):
        obs, reward, terminated, truncated, final_obs, info = sim.step(rng.uniform(-1.5, 1.5, (n, 4)).astype(np.float32))
        episodes_ended += int((terminated | truncated).sum())
        chunks += [obs.tobytes(), reward.tobytes(), terminated.tobytes(), truncated.tobytes(), final_obs.tobytes(),
                   info["reward_terms"].tobytes(), info["termination_cause"].tobytes()]
    chunks.append(sim.get_state().tobytes())
    return b"".join(chunks), episodes_ended


def test_1_thread_and_8_threads_give_identical_episodes_including_the_random_resets(drone_cfg: dict, task_cfg: dict) -> None:
    one, ended = run_with_threads(drone_cfg, task_cfg, 1)
    eight, _ = run_with_threads(drone_cfg, task_cfg, 8)
    default, _ = run_with_threads(drone_cfg, task_cfg, None)
    assert ended > 200, "episodes must end and restart for this test to cover the random resets"
    assert one == eight == default
