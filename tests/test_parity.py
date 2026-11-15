"""Milestone 2: parity between the Rust simulator (`quadsim.QuadSim`) and the NumPy reference
(`quadrace.ref_sim`), plus checks on the Python boundary of the Rust code.

Handoff section 7: for 100 random states and action sequences, Rust and NumPy agree to 1e-9 in
position and attitude after 1 s. Here 1 s is 50 policy steps, each of 10 physics substeps.

The Rust side takes float32 actions (handoff section 5). The tests draw actions as float32 and
give the NumPy reference the very same values widened to float64, so both sides see identical inputs.
"""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import quadsim
from quadrace import ref_sim
from quadrace.ref_sim import POS, QUAT, RATES, STATE_SIZE, THRUST, VEL

REPO_ROOT = Path(__file__).resolve().parent.parent
DRONE_YAML = REPO_ROOT / "configs" / "drone.yaml"
NUM_STATES = 100
STEPS_PER_SECOND = 50


@pytest.fixture(scope="module")
def drone_cfg() -> dict:
    with open(DRONE_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def random_states(rng: np.random.Generator, n: int, thrust_max: float) -> np.ndarray:
    """n random but physically ordinary states: anywhere in a 10 m box, any attitude,
    moving at a few m/s, spinning at a few rad/s, with motors anywhere in their range."""
    states = np.zeros((n, STATE_SIZE))
    states[:, POS] = rng.uniform(-5.0, 5.0, size=(n, 3))
    q = rng.normal(size=(n, 4))
    states[:, QUAT] = q / np.linalg.norm(q, axis=1, keepdims=True)
    states[:, VEL] = rng.normal(0.0, 2.0, size=(n, 3))
    states[:, RATES] = rng.normal(0.0, 2.0, size=(n, 3))
    states[:, THRUST] = rng.uniform(0.0, thrust_max, size=(n, 4))
    return states


def random_actions(rng: np.random.Generator, n: int, steps: int) -> np.ndarray:
    """float32 actions in [-1, 1], shape (n, steps, 4)."""
    return rng.uniform(-1.0, 1.0, size=(n, steps, 4)).astype(np.float32)


def run_rust(cfg: dict, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    sim = quadsim.QuadSim(len(states), cfg)
    sim.set_state(states)
    for t in range(actions.shape[1]):
        sim.step_physics(np.ascontiguousarray(actions[:, t]))
    return sim.get_state()


def run_reference(cfg: dict, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    drone = load_reference_params(cfg)
    out = np.zeros_like(states)
    for i in range(len(states)):
        state = states[i].copy()
        for t in range(actions.shape[1]):
            state = ref_sim.policy_step(drone, state, actions[i, t].astype(np.float64))
        out[i] = state
    return out


def load_reference_params(cfg: dict) -> ref_sim.DroneParams:
    """ref_sim reads a YAML file. The only setting these tests change is drag.enabled, so load
    the file and override that one value to match `cfg`."""
    params = ref_sim.load_drone_params(DRONE_YAML)
    return replace(params, drag_enabled=bool(cfg["drag"]["enabled"]))


def quaternion_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Largest component difference per row, up to the sign of the quaternion (q and -q are the
    same rotation)."""
    plus = np.abs(a - b).max(axis=1)
    minus = np.abs(a + b).max(axis=1)
    return np.minimum(plus, minus)


def assert_parity(rust: np.ndarray, ref: np.ndarray, tolerance: float) -> None:
    assert np.abs(rust[:, POS] - ref[:, POS]).max() < tolerance, "position"
    assert quaternion_error(rust[:, QUAT], ref[:, QUAT]).max() < tolerance, "attitude"
    assert np.abs(rust[:, VEL] - ref[:, VEL]).max() < tolerance, "velocity"
    assert np.abs(rust[:, RATES] - ref[:, RATES]).max() < tolerance, "body rates"
    assert np.abs(rust[:, THRUST] - ref[:, THRUST]).max() < tolerance, "motor thrusts"


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------


def test_rust_and_numpy_agree_to_1e_9_after_one_second_on_100_random_states(drone_cfg: dict) -> None:
    rng = np.random.default_rng(10)
    states = random_states(rng, NUM_STATES, drone_cfg["motors"]["thrust_max"])
    actions = random_actions(rng, NUM_STATES, STEPS_PER_SECOND)

    rust = run_rust(drone_cfg, states, actions)
    ref = run_reference(drone_cfg, states, actions)

    assert_parity(rust, ref, 1e-9)


def test_rust_and_numpy_agree_with_drag_enabled(drone_cfg: dict) -> None:
    cfg = copy.deepcopy(drone_cfg)
    cfg["drag"]["enabled"] = True
    rng = np.random.default_rng(11)
    states = random_states(rng, NUM_STATES, cfg["motors"]["thrust_max"])
    actions = random_actions(rng, NUM_STATES, STEPS_PER_SECOND)

    assert_parity(run_rust(cfg, states, actions), run_reference(cfg, states, actions), 1e-9)


def test_rust_and_numpy_agree_after_a_single_policy_step_to_1e_12(drone_cfg: dict) -> None:
    rng = np.random.default_rng(12)
    states = random_states(rng, NUM_STATES, drone_cfg["motors"]["thrust_max"])
    actions = random_actions(rng, NUM_STATES, 1)

    assert_parity(run_rust(drone_cfg, states, actions), run_reference(drone_cfg, states, actions), 1e-12)


def test_rust_and_numpy_agree_on_actions_outside_minus_one_to_one(drone_cfg: dict) -> None:
    # Both sides clip, so wild actions must give the same answer on both.
    rng = np.random.default_rng(13)
    states = random_states(rng, NUM_STATES, drone_cfg["motors"]["thrust_max"])
    actions = (rng.normal(0.0, 3.0, size=(NUM_STATES, 10, 4))).astype(np.float32)

    assert_parity(run_rust(drone_cfg, states, actions), run_reference(drone_cfg, states, actions), 1e-9)


def test_rust_and_numpy_agree_on_saturating_actions(drone_cfg: dict) -> None:
    # Every action is exactly -1 or +1 on each channel: the corners of the action box, where the
    # motors saturate most often.
    rng = np.random.default_rng(14)
    states = random_states(rng, NUM_STATES, drone_cfg["motors"]["thrust_max"])
    actions = rng.choice([-1.0, 1.0], size=(NUM_STATES, STEPS_PER_SECOND, 4)).astype(np.float32)

    assert_parity(run_rust(drone_cfg, states, actions), run_reference(drone_cfg, states, actions), 1e-9)


# --------------------------------------------------------------------------
# The Python boundary of QuadSim
# --------------------------------------------------------------------------


def test_new_quadsim_starts_level_at_the_origin_at_hover_thrust(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(3, drone_cfg)
    state = sim.get_state()
    assert sim.num_envs == 3
    assert state.shape == (3, STATE_SIZE) and state.dtype == np.float64
    expected = ref_sim.hover_state(ref_sim.load_drone_params(DRONE_YAML), np.zeros(3))
    np.testing.assert_allclose(state, np.tile(expected, (3, 1)), rtol=1e-15)


def test_set_state_then_get_state_returns_what_was_set(drone_cfg: dict) -> None:
    rng = np.random.default_rng(15)
    states = random_states(rng, 5, drone_cfg["motors"]["thrust_max"])
    sim = quadsim.QuadSim(5, drone_cfg)
    sim.set_state(states)
    np.testing.assert_array_equal(sim.get_state(), states)


def test_step_physics_accepts_a_strided_action_array(drone_cfg: dict) -> None:
    rng = np.random.default_rng(16)
    states = random_states(rng, 4, drone_cfg["motors"]["thrust_max"])
    actions = random_actions(rng, 4, 1)[:, 0]
    wide = np.zeros((4, 8), dtype=np.float32)
    wide[:, ::2] = actions  # every other column: a strided, non-contiguous (4, 4) view
    strided = wide[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"]

    a = quadsim.QuadSim(4, drone_cfg)
    a.set_state(states)
    a.step_physics(strided)
    b = quadsim.QuadSim(4, drone_cfg)
    b.set_state(states)
    b.step_physics(np.ascontiguousarray(actions))
    np.testing.assert_array_equal(a.get_state(), b.get_state())


def test_step_physics_does_not_modify_the_actions_it_is_given(drone_cfg: dict) -> None:
    actions = np.full((2, 4), 0.3, dtype=np.float32)
    before = actions.copy()
    quadsim.QuadSim(2, drone_cfg).step_physics(actions)
    np.testing.assert_array_equal(actions, before)


def test_step_physics_rejects_float64_actions(drone_cfg: dict) -> None:
    with pytest.raises(TypeError):
        quadsim.QuadSim(2, drone_cfg).step_physics(np.zeros((2, 4), dtype=np.float64))


def test_step_physics_rejects_the_wrong_shape(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(2, drone_cfg)
    with pytest.raises(ValueError, match="shape"):
        sim.step_physics(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        sim.step_physics(np.zeros((2, 3), dtype=np.float32))


def test_step_physics_rejects_nan_actions(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(2, drone_cfg)
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sim.step_physics(actions)


def test_set_state_rejects_the_wrong_shape(drone_cfg: dict) -> None:
    with pytest.raises(ValueError, match="shape"):
        quadsim.QuadSim(2, drone_cfg).set_state(np.zeros((2, 16)))


def test_set_state_rejects_float32(drone_cfg: dict) -> None:
    with pytest.raises(TypeError):
        quadsim.QuadSim(1, drone_cfg).set_state(np.zeros((1, STATE_SIZE), dtype=np.float32))


def test_set_state_rejects_a_quaternion_that_is_not_unit_length_and_changes_nothing(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(2, drone_cfg)
    before = sim.get_state()
    bad = before.copy()
    bad[1, QUAT] = [2.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="environment 1"):
        sim.set_state(bad)
    np.testing.assert_array_equal(sim.get_state(), before)


def test_set_state_rejects_nan(drone_cfg: dict) -> None:
    sim = quadsim.QuadSim(1, drone_cfg)
    bad = sim.get_state()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sim.set_state(bad)


def test_quadsim_rejects_zero_envs(drone_cfg: dict) -> None:
    with pytest.raises(ValueError, match="num_envs"):
        quadsim.QuadSim(0, drone_cfg)


def test_quadsim_rejects_a_config_with_a_missing_key(drone_cfg: dict) -> None:
    cfg = copy.deepcopy(drone_cfg)
    del cfg["body"]["mass"]
    with pytest.raises(KeyError, match="body.mass"):
        quadsim.QuadSim(1, cfg)


def test_quadsim_rejects_a_config_with_a_missing_section(drone_cfg: dict) -> None:
    cfg = copy.deepcopy(drone_cfg)
    del cfg["motors"]
    with pytest.raises(KeyError, match="motors"):
        quadsim.QuadSim(1, cfg)


def test_quadsim_rejects_a_config_value_of_the_wrong_type(drone_cfg: dict) -> None:
    cfg = copy.deepcopy(drone_cfg)
    cfg["body"]["mass"] = "heavy"
    with pytest.raises(TypeError, match="body.mass"):
        quadsim.QuadSim(1, cfg)


def test_quadsim_rejects_a_config_with_a_non_physical_value(drone_cfg: dict) -> None:
    cfg = copy.deepcopy(drone_cfg)
    cfg["body"]["mass"] = -1.0
    with pytest.raises(ValueError, match="body.mass"):
        quadsim.QuadSim(1, cfg)


def test_quadsim_rejects_a_config_that_is_not_a_dict() -> None:
    with pytest.raises(TypeError):
        quadsim.QuadSim(1, "configs/drone.yaml")  # type: ignore[arg-type]
