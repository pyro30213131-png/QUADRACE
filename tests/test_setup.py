"""Milestone 0 checks: the toolchain works and the drone config is sane.

These are not simulator tests (those start in milestone 1). They prove that
`import quadsim` and `import quadrace` both work after `maturin develop`.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_quadsim() -> None:
    import quadsim  # noqa: F401


def test_import_quadrace() -> None:
    import quadrace  # noqa: F401


def test_add_arrays_adds_element_by_element() -> None:
    import quadsim

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    result = quadsim.add_arrays(a, b)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])


def test_add_arrays_leaves_inputs_unchanged() -> None:
    import quadsim

    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    quadsim.add_arrays(a, b)
    np.testing.assert_array_equal(a, [1.0, 2.0])
    np.testing.assert_array_equal(b, [3.0, 4.0])


def test_add_arrays_accepts_non_contiguous_input() -> None:
    import quadsim

    a = np.arange(6.0)[::2]  # a strided view: 0, 2, 4
    b = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(quadsim.add_arrays(a, b), [1.0, 3.0, 5.0])


def test_add_arrays_rejects_length_mismatch() -> None:
    import quadsim

    with pytest.raises(ValueError, match="length mismatch"):
        quadsim.add_arrays(np.zeros(3), np.zeros(2))


def test_add_arrays_rejects_float32() -> None:
    import quadsim

    with pytest.raises(TypeError):
        quadsim.add_arrays(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))


def test_add_arrays_rejects_two_dimensional_input() -> None:
    import quadsim

    with pytest.raises(TypeError):
        quadsim.add_arrays(np.zeros((2, 2)), np.zeros((2, 2)))


@pytest.fixture(scope="module")
def drone() -> dict:
    with open(REPO_ROOT / "configs" / "drone.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_drone_config_matches_handoff_section_7(drone: dict) -> None:
    assert drone["body"]["mass"] == 0.75
    assert drone["body"]["inertia_diag"] == [0.0025, 0.0021, 0.0043]
    assert drone["body"]["arm_length"] == 0.15
    assert drone["motors"]["thrust_max"] == 8.5
    assert drone["motors"]["time_constant"] == 0.03
    assert drone["motors"]["yaw_torque_coeff"] == 0.02
    assert drone["limits"]["max_rates"] == [10.0, 10.0, 4.0]
    assert drone["controller"]["rate_gain_kp"] == [20.0, 20.0, 10.0]
    assert drone["drag"]["linear"] == [0.3, 0.3, 0.3]


def test_drone_config_drag_is_off_until_robustness_milestone(drone: dict) -> None:
    assert drone["drag"]["enabled"] is False


def test_drone_config_timing_gives_ten_substeps_per_action(drone: dict) -> None:
    assert drone["timing"]["physics_hz"] / drone["timing"]["policy_hz"] == 10


def test_drone_config_can_hover(drone: dict) -> None:
    # Hover needs each motor to supply a quarter of the weight, so that must be
    # below the per-motor maximum. This guards against a unit slip in the YAML.
    hover_thrust_per_motor = drone["body"]["mass"] * drone["world"]["gravity"] / 4
    assert 0.0 < hover_thrust_per_motor < drone["motors"]["thrust_max"]
