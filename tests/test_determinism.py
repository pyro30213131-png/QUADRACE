"""Milestone 2: results must not depend on the number of threads (handoff section 5).

Each drone is stepped by the same code using only its own state and action, so the thread
that runs it cannot matter. These tests check that by running the same batch with 1, 8 and
the default number of threads and comparing the results exactly (bit for bit, not within a
tolerance).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

import quadsim
from quadrace.ref_sim import POS, QUAT, RATES, STATE_SIZE, THRUST, VEL

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_ENVS = 512
NUM_STEPS = 100  # 2 s of flight at 50 Hz


@pytest.fixture(scope="module")
def drone_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "drone.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def random_states(rng: np.random.Generator, n: int, thrust_max: float) -> np.ndarray:
    states = np.zeros((n, STATE_SIZE))
    states[:, POS] = rng.uniform(-5.0, 5.0, size=(n, 3))
    q = rng.normal(size=(n, 4))
    states[:, QUAT] = q / np.linalg.norm(q, axis=1, keepdims=True)
    states[:, VEL] = rng.normal(0.0, 2.0, size=(n, 3))
    states[:, RATES] = rng.normal(0.0, 2.0, size=(n, 3))
    states[:, THRUST] = rng.uniform(0.0, thrust_max, size=(n, 4))
    return states


def run(cfg: dict, num_threads: int | None, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    sim = quadsim.QuadSim(len(states), cfg, num_threads=num_threads)
    sim.set_state(states)
    for t in range(actions.shape[1]):
        sim.step_physics(np.ascontiguousarray(actions[:, t]))
    return sim.get_state()


@pytest.fixture(scope="module")
def inputs(drone_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20)
    states = random_states(rng, NUM_ENVS, drone_cfg["motors"]["thrust_max"])
    actions = rng.uniform(-1.0, 1.0, size=(NUM_ENVS, NUM_STEPS, 4)).astype(np.float32)
    return states, actions


def test_1_thread_and_8_threads_give_identical_results(
    drone_cfg: dict, inputs: tuple[np.ndarray, np.ndarray]
) -> None:
    states, actions = inputs
    one = run(drone_cfg, 1, states, actions)
    eight = run(drone_cfg, 8, states, actions)
    assert one.tobytes() == eight.tobytes()


def test_default_thread_count_gives_the_same_results_as_1_thread(
    drone_cfg: dict, inputs: tuple[np.ndarray, np.ndarray]
) -> None:
    states, actions = inputs
    assert run(drone_cfg, None, states, actions).tobytes() == run(drone_cfg, 1, states, actions).tobytes()


def test_the_flight_actually_moved_the_drones(drone_cfg: dict, inputs: tuple[np.ndarray, np.ndarray]) -> None:
    # Guards the two tests above against passing because nothing happened.
    states, actions = inputs
    after = run(drone_cfg, 1, states, actions)
    assert np.abs(after[:, POS] - states[:, POS]).max() > 1.0
    assert np.abs(after[:, QUAT] - states[:, QUAT]).max() > 0.1


def test_num_threads_property_reports_the_requested_pool_size(drone_cfg: dict) -> None:
    assert quadsim.QuadSim(4, drone_cfg, num_threads=1).num_threads == 1
    assert quadsim.QuadSim(4, drone_cfg, num_threads=8).num_threads == 8
    assert quadsim.QuadSim(4, drone_cfg).num_threads >= 1


def test_zero_threads_is_rejected(drone_cfg: dict) -> None:
    with pytest.raises(ValueError, match="num_threads"):
        quadsim.QuadSim(4, drone_cfg, num_threads=0)
