"""Milestone 1: tests for the NumPy reference simulator (python/quadrace/ref_sim.py).

The first group is the list of non-parity tests in handoff section 7, one test (or one
parametrized test) per bullet, named after it. The tolerances come from the handoff where it
gives one. Where it does not, each tolerance is explained next to the test and was chosen
from the maths before the test was run.

The second group checks the pieces those tests lean on (config loading, action mapping,
drag, input checks). The parity test against Rust belongs to milestone 2.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quadrace import ref_sim
from quadrace.ref_sim import POS, QUAT, RATES, STATE_SIZE, THRUST, VEL

REPO_ROOT = Path(__file__).resolve().parent.parent
DT = 1.0 / 500.0  # physics substep, s


@pytest.fixture(scope="module")
def params() -> ref_sim.DroneParams:
    return ref_sim.load_drone_params(REPO_ROOT / "configs" / "drone.yaml")


def random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------
# Handoff section 7: tests the simulator must pass before any learning
# --------------------------------------------------------------------------


def test_free_fall_with_motors_off_follows_z0_minus_4_905_t_squared(params: ref_sim.DroneParams) -> None:
    # Action (-1, 0, 0, 0) commands zero thrust, and the motors start at zero thrust.
    # RK4 is exact for a quadratic in t, so only rounding error is left: 1e-9 is generous.
    state = np.zeros(STATE_SIZE)
    state[POS] = [1.0, 2.0, 10.0]
    state[QUAT] = [1.0, 0.0, 0.0, 0.0]
    action = np.array([-1.0, 0.0, 0.0, 0.0])

    for step in range(1, 51):  # 1 s at 50 Hz
        state = ref_sim.policy_step(params, state, action)
        t = step / 50.0
        assert state[POS][2] == pytest.approx(10.0 - 4.905 * t**2, abs=1e-9)
        assert state[POS][0] == pytest.approx(1.0, abs=1e-12)
        assert state[POS][1] == pytest.approx(2.0, abs=1e-12)


def test_hover_thrust_holds_altitude_within_1_mm_over_5_s(params: ref_sim.DroneParams) -> None:
    start_z = 1.5
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, start_z]))
    action = ref_sim.hover_action(params)

    worst_error = 0.0
    for _ in range(250):  # 5 s at 50 Hz
        state = ref_sim.policy_step(params, state, action)
        worst_error = max(worst_error, abs(state[POS][2] - start_z))
    assert worst_error < 1e-3  # the handoff's 1 mm


def test_commanded_roll_rate_of_5_rad_s_is_reached_within_10_percent_in_0_2_s(
    params: ref_sim.DroneParams,
) -> None:
    # "Reached within 10 percent in 0.2 s" is read as: at t = 0.2 s the roll rate is within
    # 10 percent of the 5 rad/s command (between 4.5 and 5.5). The rate loop plus the
    # motor lag form a second-order system with damping ratio about 0.65 and natural
    # frequency about 26 rad/s, so by 0.2 s it has settled.
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, 1.0]))
    action = ref_sim.hover_action(params)
    action[1] = 5.0 / params.max_rates_b[0]  # roll command of 5 rad/s

    for _ in range(10):  # 0.2 s at 50 Hz
        state = ref_sim.policy_step(params, state, action)
    assert state[RATES][0] == pytest.approx(5.0, rel=0.10)


def test_roll_rate_command_leaves_pitch_and_yaw_rates_at_zero(params: ref_sim.DroneParams) -> None:
    # A pure roll command makes no pitch or yaw torque, and omega x (J omega) is zero for a
    # pure x rate, so the other two rates stay at zero up to rounding.
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, 1.0]))
    action = ref_sim.hover_action(params)
    action[1] = 0.5

    for _ in range(10):
        state = ref_sim.policy_step(params, state, action)
    np.testing.assert_allclose(state[RATES][1:], 0.0, atol=1e-9)


def test_rate_command_beyond_motor_capability_clamps_thrust_commands_to_limits(
    params: ref_sim.DroneParams,
) -> None:
    # A roll rate of 100 rad/s (the action mapping would never ask for it) needs far more
    # torque than the motors can make. Positive roll torque comes from the motors with
    # y = +d (motors 1 and 2), so those saturate high and motors 0 and 3 saturate at zero.
    hover_thrust = params.mass * params.gravity
    tau_des_b = ref_sim.rate_controller(params, np.zeros(3), np.array([100.0, 0.0, 0.0]))
    f_cmd = ref_sim.allocate(params, hover_thrust, tau_des_b)
    np.testing.assert_array_equal(f_cmd, [params.thrust_min, params.thrust_max, params.thrust_max, params.thrust_min])


def _saturated_roll_rate_after(params: ref_sim.DroneParams, seconds: float) -> float:
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, 1.0]))
    hover_thrust = params.mass * params.gravity
    for _ in range(round(seconds / DT)):
        state = ref_sim.physics_substep(params, state, hover_thrust, np.array([100.0, 0.0, 0.0]), DT)
    return float(state[RATES][0])


def test_saturated_roll_response_cannot_exceed_the_torque_limit(params: ref_sim.DroneParams) -> None:
    # The most roll torque the motors can make is 2 d f_max (motors 1 and 2 at f_max,
    # motors 0 and 3 at zero), so the roll rate can grow no faster than that over Jxx.
    d = params.arm_length / np.sqrt(2.0)
    max_roll_accel = 2.0 * d * params.thrust_max / params.inertia_diag_b[0]
    t = 0.05
    assert _saturated_roll_rate_after(params, t) <= max_roll_accel * t


def test_saturated_roll_response_is_slower_than_an_unsaturated_controller(params: ref_sim.DroneParams) -> None:
    # Without any motor limit, the rate loop alone would give omega(t) = cmd (1 - exp(-Kp t)),
    # which is 63.2 rad/s at t = 0.05 s for cmd = 100 and Kp = 20. Saturation and motor lag
    # must keep the real response below that.
    t = 0.05
    unsaturated = 100.0 * (1.0 - np.exp(-params.rate_gain_kp[0] * t))
    assert _saturated_roll_rate_after(params, t) < unsaturated


def test_angular_momentum_in_world_frame_is_constant_without_torque_or_drag(
    params: ref_sim.DroneParams,
) -> None:
    # No torque: all motors off. Unequal inertias, so omega really does change (Euler's
    # equation) while R J omega must not. Per-step RK4 error is about (omega dt)^5 / 120,
    # roughly 1e-12 here, so 1e-8 relative over 500 steps leaves a wide margin.
    rng = np.random.default_rng(1)
    state = np.zeros(STATE_SIZE)
    state[QUAT] = random_unit_quaternion(rng)
    state[RATES] = [3.0, -2.0, 5.0]
    motors_off = np.zeros(4)

    def angular_momentum_w(s: np.ndarray) -> np.ndarray:
        return ref_sim.quat_to_rotation_matrix(s[QUAT]) @ (params.inertia_diag_b * s[RATES])

    l0_w = angular_momentum_w(state)
    worst = 0.0
    for _ in range(500):  # 1 s
        state = ref_sim.rk4_step(params, state, motors_off, DT)
        state[QUAT] = ref_sim.quat_normalize(state[QUAT])
        worst = max(worst, np.linalg.norm(angular_momentum_w(state) - l0_w) / np.linalg.norm(l0_w))

    assert np.linalg.norm(state[RATES] - [3.0, -2.0, 5.0]) > 0.1  # guard: the body really tumbled
    assert worst < 1e-8


@pytest.mark.parametrize("n_steps, tolerance", [(1, 1e-12), (500, 1e-9)])
def test_constant_omega_with_equal_inertias_matches_the_exact_quaternion_formula(
    params: ref_sim.DroneParams, n_steps: int, tolerance: float
) -> None:
    # Equal inertias make omega x (J omega) zero, so omega stays constant and the exact
    # answer is q_next = q (x) (cos(|w| dt / 2), sin(|w| dt / 2) w / |w|). The controller
    # is given omega as its command and T_cmd = 0, so it asks for exactly zero torque.
    # Tolerances: single-step RK4 error is about 1e-14; 500 steps add up to well under 1e-9.
    equal = replace(params, inertia_diag_b=np.array([0.003, 0.003, 0.003]))
    rng = np.random.default_rng(2)
    q0 = random_unit_quaternion(rng)
    omega_b = np.array([1.5, -2.0, 2.5])

    state = np.zeros(STATE_SIZE)
    state[QUAT] = q0
    state[RATES] = omega_b
    for _ in range(n_steps):
        state = ref_sim.physics_substep(equal, state, 0.0, omega_b, DT)

    speed = np.linalg.norm(omega_b)
    half_angle = speed * DT * n_steps / 2.0
    rotation = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * omega_b / speed])
    expected = ref_sim.quat_multiply(q0, rotation)

    np.testing.assert_allclose(state[RATES], omega_b, atol=1e-12)
    # q and -q are the same rotation, so compare up to sign.
    assert min(np.abs(state[QUAT] - expected).max(), np.abs(state[QUAT] + expected).max()) < tolerance


def test_equal_thrust_on_all_motors_gives_zero_torque(params: ref_sim.DroneParams) -> None:
    f = np.full(4, 3.7)
    np.testing.assert_allclose(ref_sim.body_torque(params, f), 0.0, atol=1e-15)
    assert ref_sim.total_thrust(f) == pytest.approx(4 * 3.7)


# Expected torque signs (tau_x, tau_y, tau_z) for thrust on one motor, worked out by hand from
# the section 6 table and tau_x = sum(y f), tau_y = -sum(x f), tau_z = sum(s kappa f):
#   motor 0: x = +d, y = -d, s = +1  ->  (-, -, +)
#   motor 1: x = -d, y = +d, s = +1  ->  (+, +, +)
#   motor 2: x = +d, y = +d, s = -1  ->  (+, -, -)
#   motor 3: x = -d, y = -d, s = -1  ->  (-, +, -)
@pytest.mark.parametrize(
    "motor, signs",
    [(0, (-1, -1, +1)), (1, (+1, +1, +1)), (2, (+1, -1, -1)), (3, (-1, +1, -1))],
)
def test_single_motor_torque_signs_match_the_section_6_table(
    params: ref_sim.DroneParams, motor: int, signs: tuple[int, int, int]
) -> None:
    f = np.zeros(4)
    f[motor] = 2.0
    tau_b = ref_sim.body_torque(params, f)

    d = params.arm_length / np.sqrt(2.0)
    expected_magnitude = np.array([d * 2.0, d * 2.0, params.yaw_torque_coeff * 2.0])
    np.testing.assert_allclose(tau_b, np.array(signs) * expected_magnitude, rtol=1e-12)


@pytest.mark.parametrize("q, tolerance", [([0.7071, 0.0, 0.0, 0.7071], 1e-4), ([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], 1e-12)])
def test_90_degree_yaw_quaternion_maps_body_x_to_world_y(q: list[float], tolerance: float) -> None:
    # The first quaternion is the one written in the handoff (rounded to 4 digits, so 1e-4);
    # the second is the exact one.
    vec_w = ref_sim.rotate_body_to_world(np.array(q), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(vec_w, [0.0, 1.0, 0.0], atol=tolerance)


# --------------------------------------------------------------------------
# Supporting checks
# --------------------------------------------------------------------------


def test_load_drone_params_reads_the_values_in_drone_yaml(params: ref_sim.DroneParams) -> None:
    assert params.mass == 0.75
    assert params.thrust_max == 8.5
    assert params.drag_enabled is False
    assert params.inertia_diag_b.dtype == np.float64
    np.testing.assert_array_equal(params.inertia_diag_b, [0.0025, 0.0021, 0.0043])
    assert ref_sim.substeps_per_action(params) == 10


def test_substep_runs_motor_lag_before_rk4_so_rk4_sees_the_lagged_thrust(params: ref_sim.DroneParams) -> None:
    # Handoff section 7: step 4 (motor lag) comes before step 5 (RK4 with the thrusts held
    # constant). From rest with motors at zero and hover thrust commanded, one substep first
    # lifts every motor to f = f_cmd (1 - exp(-dt / tau_m)); RK4 then sees that thrust for the
    # whole substep. Level and torque-free, so dv_z = dt * (T / m - g) exactly.
    state = np.zeros(STATE_SIZE)
    state[QUAT] = [1.0, 0.0, 0.0, 0.0]
    hover_thrust = params.mass * params.gravity

    after = ref_sim.physics_substep(params, state, hover_thrust, np.zeros(3), DT)

    lagged_thrust = hover_thrust * (1.0 - np.exp(-DT / params.motor_time_constant))
    np.testing.assert_allclose(after[THRUST], lagged_thrust / 4.0, rtol=1e-12)
    assert after[VEL][2] == pytest.approx(DT * (lagged_thrust / params.mass - params.gravity), rel=1e-12)


def test_allocation_inverts_the_mixing_matrix_when_no_motor_saturates(params: ref_sim.DroneParams) -> None:
    rng = np.random.default_rng(3)
    f = rng.uniform(1.0, 7.0, size=4)
    wrench = ref_sim.mixing_matrix(params) @ f
    np.testing.assert_allclose(ref_sim.allocate(params, wrench[0], wrench[1:]), f, rtol=1e-12)


def test_action_mapping_matches_handoff_section_7(params: ref_sim.DroneParams) -> None:
    thrust_cmd, omega_cmd_b = ref_sim.action_to_commands(params, np.array([0.5, 1.0, -1.0, 0.25]))
    assert thrust_cmd == pytest.approx(0.75 * 4 * 8.5)  # (0.5 + 1) / 2 * 4 * f_max
    np.testing.assert_allclose(omega_cmd_b, [10.0, -10.0, 1.0])


def test_hover_action_commands_exactly_the_weight(params: ref_sim.DroneParams) -> None:
    thrust_cmd, omega_cmd_b = ref_sim.action_to_commands(params, ref_sim.hover_action(params))
    assert thrust_cmd == pytest.approx(params.mass * params.gravity, rel=1e-12)
    np.testing.assert_array_equal(omega_cmd_b, 0.0)


def test_actions_outside_minus_one_to_one_are_clipped(params: ref_sim.DroneParams) -> None:
    wild = ref_sim.action_to_commands(params, np.array([7.0, -3.0, 2.0, 9.0]))
    limit = ref_sim.action_to_commands(params, np.array([1.0, -1.0, 1.0, 1.0]))
    assert wild[0] == limit[0]
    np.testing.assert_array_equal(wild[1], limit[1])


def test_quaternion_stays_unit_length_over_many_policy_steps(params: ref_sim.DroneParams) -> None:
    rng = np.random.default_rng(4)
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, 5.0]))
    for _ in range(200):
        state = ref_sim.policy_step(params, state, rng.uniform(-1.0, 1.0, size=4))
        assert abs(np.linalg.norm(state[QUAT]) - 1.0) < 1e-12


def test_policy_step_does_not_modify_its_input_state(params: ref_sim.DroneParams) -> None:
    state = ref_sim.hover_state(params, np.array([0.0, 0.0, 1.0]))
    before = state.copy()
    ref_sim.policy_step(params, state, np.array([0.0, 0.3, -0.2, 0.1]))
    np.testing.assert_array_equal(state, before)


def test_policy_step_rejects_a_state_with_the_wrong_shape(params: ref_sim.DroneParams) -> None:
    with pytest.raises(ValueError, match="shape"):
        ref_sim.policy_step(params, np.zeros(16), ref_sim.hover_action(params))


def test_policy_step_rejects_a_state_that_is_not_finite(params: ref_sim.DroneParams) -> None:
    state = ref_sim.hover_state(params, np.zeros(3))
    state[POS][0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ref_sim.policy_step(params, state, ref_sim.hover_action(params))


def test_policy_step_rejects_a_quaternion_that_is_not_unit_length(params: ref_sim.DroneParams) -> None:
    state = ref_sim.hover_state(params, np.zeros(3))
    state[QUAT] = [2.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="unit length"):
        ref_sim.policy_step(params, state, ref_sim.hover_action(params))


def test_action_with_the_wrong_shape_is_rejected(params: ref_sim.DroneParams) -> None:
    with pytest.raises(ValueError, match="shape"):
        ref_sim.action_to_commands(params, np.zeros(3))


def _horizontal_coast(params: ref_sim.DroneParams, seconds: float) -> np.ndarray:
    # Motors off, level, moving along world x at 5 m/s. With a level attitude R = I, so
    # drag acts straight along x, and gravity only moves z.
    state = np.zeros(STATE_SIZE)
    state[QUAT] = [1.0, 0.0, 0.0, 0.0]
    state[VEL] = [5.0, 0.0, 0.0]
    for _ in range(round(seconds * 50)):
        state = ref_sim.policy_step(params, state, np.array([-1.0, 0.0, 0.0, 0.0]))
    return state


def test_linear_drag_decays_horizontal_velocity_exponentially(params: ref_sim.DroneParams) -> None:
    # v_dot = -(D / m) v gives v(t) = v0 exp(-D t / m). RK4 error is far below the tolerance.
    drag_on = replace(params, drag_enabled=True)
    state = _horizontal_coast(drag_on, 1.0)
    expected_vx = 5.0 * np.exp(-0.3 * 1.0 / 0.75)
    assert state[VEL][0] == pytest.approx(expected_vx, rel=1e-9)


def test_drag_disabled_leaves_horizontal_velocity_unchanged(params: ref_sim.DroneParams) -> None:
    state = _horizontal_coast(params, 1.0)
    assert state[VEL][0] == pytest.approx(5.0, abs=1e-12)
    assert state[THRUST].sum() == 0.0
