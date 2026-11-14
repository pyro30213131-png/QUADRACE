"""NumPy reference simulator for one quadrotor (milestone 1).

This is the oracle. It is written for clarity, not speed, in plain NumPy float64.
In milestone 2 the Rust simulator is ported from it and must agree with it to 1e-9
(the parity test), so every choice made here is also a choice for the Rust code.

What it simulates, following handoff section 7. One physics substep does, in order:

1. Rate controller: from the commanded body rates, the desired body torque.
2. Allocation: invert the 4x4 mixing matrix to get four motor thrust commands.
3. Clamp each thrust command to [thrust_min, thrust_max].
4. Motor lag, exact first-order update.
5. RK4 on position, velocity, quaternion and body rates, motor thrusts held constant.
6. Renormalize the quaternion.

One policy step is `substeps_per_action(params)` (10) substeps with the same action.

Conventions (handoff section 6, do not change):

- SI units, radians. World frame z up, gravity (0, 0, -g). Body frame x forward,
  y left, z up along the thrust axis.
- Quaternion scalar first (w, x, y, z), Hamilton product, rotates body to world.
- Angular velocity omega is always in the body frame.
- State is 17 values: position_w (3), quaternion (4), velocity_w (3),
  omega_b (3), motor thrusts (4). Index it with the slices POS, QUAT, VEL, RATES, THRUST.

Names carry their frame: `_w` is the world frame, `_b` is the body frame.
There is no module-level state; every function takes what it needs and returns new arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Where each part of the 17-value state lives (handoff section 6).
POS = slice(0, 3)  # position, world frame, m
QUAT = slice(3, 7)  # quaternion (w, x, y, z), body to world
VEL = slice(7, 10)  # velocity, world frame, m/s
RATES = slice(10, 13)  # angular velocity, body frame, rad/s
THRUST = slice(13, 17)  # the four motor thrusts, N
STATE_SIZE = 17

# The part of the state that RK4 integrates: position, quaternion, velocity, rates.
_RIGID_BODY = slice(0, 13)


@dataclass(frozen=True, eq=False)
class DroneParams:
    """Every physical constant the simulator uses. Built from configs/drone.yaml."""

    mass: float  # kg
    inertia_diag_b: np.ndarray  # (3,) kg m^2, diagonal of J in the body frame
    arm_length: float  # m, centre to motor
    thrust_min: float  # N per motor
    thrust_max: float  # N per motor
    motor_time_constant: float  # s, tau_m
    yaw_torque_coeff: float  # m, kappa
    max_rates_b: np.ndarray  # (3,) rad/s, used by the action mapping
    rate_gain_kp: np.ndarray  # (3,) 1/s
    drag_enabled: bool
    drag_linear: np.ndarray  # (3,) N s/m, diagonal of D
    gravity: float  # m/s^2, acts along world -z
    physics_hz: float
    policy_hz: float


def load_drone_params(path: str | Path) -> DroneParams:
    """Read a drone YAML file (see configs/drone.yaml) into a DroneParams."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DroneParams(
        mass=float(cfg["body"]["mass"]),
        inertia_diag_b=np.array(cfg["body"]["inertia_diag"], dtype=np.float64),
        arm_length=float(cfg["body"]["arm_length"]),
        thrust_min=float(cfg["motors"]["thrust_min"]),
        thrust_max=float(cfg["motors"]["thrust_max"]),
        motor_time_constant=float(cfg["motors"]["time_constant"]),
        yaw_torque_coeff=float(cfg["motors"]["yaw_torque_coeff"]),
        max_rates_b=np.array(cfg["limits"]["max_rates"], dtype=np.float64),
        rate_gain_kp=np.array(cfg["controller"]["rate_gain_kp"], dtype=np.float64),
        drag_enabled=bool(cfg["drag"]["enabled"]),
        drag_linear=np.array(cfg["drag"]["linear"], dtype=np.float64),
        gravity=float(cfg["world"]["gravity"]),
        physics_hz=float(cfg["timing"]["physics_hz"]),
        policy_hz=float(cfg["timing"]["policy_hz"]),
    )


def substeps_per_action(params: DroneParams) -> int:
    """Physics substeps per policy step: physics_hz / policy_hz (10 for 500 / 50)."""
    ratio = params.physics_hz / params.policy_hz
    substeps = round(ratio)
    if substeps < 1 or abs(ratio - substeps) > 1e-9:
        raise ValueError(
            f"physics_hz / policy_hz must be a whole number, got {params.physics_hz} / {params.policy_hz}"
        )
    return substeps


# --------------------------------------------------------------------------
# Quaternions and rotations
# --------------------------------------------------------------------------


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, scalar first. Not commutative."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Scale q to unit length. The same rotation, with the numerical drift removed."""
    return q / np.linalg.norm(q)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R(q) that maps body-frame vectors to world-frame vectors.

    The quaternion is normalized first. RK4 evaluates the dynamics at intermediate
    quaternions that are very slightly off unit length, and the formula below is only
    a rotation for unit quaternions. The Rust port must normalize here too.
    """
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_body_to_world(q: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """vec_w = R(q) vec_b."""
    return quat_to_rotation_matrix(q) @ vec_b


# --------------------------------------------------------------------------
# Motors: layout, mixing, torque
# --------------------------------------------------------------------------


def motor_layout(arm_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Motor positions and spin signs, X layout (handoff section 6).

    Returns (x_b, y_b, spin): the body-frame x and y of each motor, and its spin
    sign s_i. With d = arm_length / sqrt(2):

        motor 0: (+d, -d, +1)    motor 1: (-d, +d, +1)
        motor 2: (+d, +d, -1)    motor 3: (-d, -d, -1)
    """
    d = arm_length / np.sqrt(2.0)
    x_b = np.array([+d, -d, +d, -d], dtype=np.float64)
    y_b = np.array([-d, +d, +d, -d], dtype=np.float64)
    spin = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.float64)
    return x_b, y_b, spin


def mixing_matrix(params: DroneParams) -> np.ndarray:
    """4x4 matrix M with [T, tau_x, tau_y, tau_z] = M @ f, f being the four motor thrusts.

    Rows (handoff section 6), all in the body frame:
        T     = sum(f_i)                     collective thrust, along body z
        tau_x = sum(y_i * f_i)               roll torque
        tau_y = -sum(x_i * f_i)              pitch torque
        tau_z = sum(s_i * kappa * f_i)       yaw torque
    """
    x_b, y_b, spin = motor_layout(params.arm_length)
    return np.array(
        [
            np.ones(4),
            y_b,
            -x_b,
            spin * params.yaw_torque_coeff,
        ],
        dtype=np.float64,
    )


def total_thrust(f: np.ndarray) -> float:
    """Collective thrust T = sum(f_i), N, along the body z axis."""
    return float(np.sum(f))


def body_torque(params: DroneParams, f: np.ndarray) -> np.ndarray:
    """Torque tau_b (3,) in the body frame from the four motor thrusts f."""
    return mixing_matrix(params)[1:] @ f


# --------------------------------------------------------------------------
# Rigid-body dynamics and RK4
# --------------------------------------------------------------------------


def rigid_body_derivative(params: DroneParams, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Time derivative of x = [position_w, quaternion, velocity_w, omega_b] (13 values).

    The four motor thrusts f are inputs here, not states: they are constant over one RK4 step.

        p_dot       = v_w
        v_dot_w     = g_w + (1/m) R [0, 0, T] - (1/m) R D R^T v_w        (drag only if enabled)
        q_dot       = 0.5 * q (x) (0, omega_b)
        J omega_dot = tau_b - omega_b x (J omega_b)                       (Euler's equation, body frame)
    """
    q = x[QUAT]
    vel_w = x[VEL]
    omega_b = x[RATES]
    rot = quat_to_rotation_matrix(q)  # body to world

    # Newton's second law in the world frame: gravity plus thrust along body z, minus drag.
    gravity_w = np.array([0.0, 0.0, -params.gravity])
    thrust_w = rot @ np.array([0.0, 0.0, total_thrust(f)])
    accel_w = gravity_w + thrust_w / params.mass
    if params.drag_enabled:
        vel_b = rot.T @ vel_w  # R^T v_w
        drag_w = rot @ (params.drag_linear * vel_b)  # R D R^T v_w
        accel_w = accel_w - drag_w / params.mass

    # Quaternion kinematics: omega is in the body frame, so it multiplies q on the right.
    omega_quat = np.array([0.0, omega_b[0], omega_b[1], omega_b[2]])
    q_dot = 0.5 * quat_multiply(q, omega_quat)

    # Euler's rotation equation for a rigid body, all in the body frame.
    inertia_omega_b = params.inertia_diag_b * omega_b  # J omega
    omega_dot_b = (body_torque(params, f) - np.cross(omega_b, inertia_omega_b)) / params.inertia_diag_b

    return np.concatenate([vel_w, q_dot, accel_w, omega_dot_b])


def rk4_step(params: DroneParams, state: np.ndarray, f: np.ndarray, dt: float) -> np.ndarray:
    """One classical RK4 step of the rigid body with motor thrusts f held constant.

    Returns a new 17-value state. The motor-thrust part is copied unchanged, and the
    quaternion is NOT renormalized here (physics_substep does that, as step 6).
    """
    x = state[_RIGID_BODY]
    k1 = rigid_body_derivative(params, x, f)
    k2 = rigid_body_derivative(params, x + 0.5 * dt * k1, f)
    k3 = rigid_body_derivative(params, x + 0.5 * dt * k2, f)
    k4 = rigid_body_derivative(params, x + dt * k3, f)
    new_state = state.copy()
    new_state[_RIGID_BODY] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new_state


# --------------------------------------------------------------------------
# Rate controller, allocation, motor lag
# --------------------------------------------------------------------------


def rate_controller(params: DroneParams, omega_b: np.ndarray, omega_cmd_b: np.ndarray) -> np.ndarray:
    """Desired body torque tau_des_b (3,), step 1.

        tau_des = J (Kp .* (omega_cmd - omega)) + omega x (J omega)

    The first term asks for an angular acceleration proportional to the rate error.
    The second cancels the gyroscopic term in Euler's equation.
    """
    angular_accel_des_b = params.rate_gain_kp * (omega_cmd_b - omega_b)
    inertia_omega_b = params.inertia_diag_b * omega_b
    return params.inertia_diag_b * angular_accel_des_b + np.cross(omega_b, inertia_omega_b)


def allocate(params: DroneParams, thrust_cmd: float, tau_des_b: np.ndarray) -> np.ndarray:
    """Four motor thrust commands from [T_cmd, tau_des], steps 2 and 3.

    Solves M f_cmd = [T_cmd, tau_des] for f_cmd, then clamps each entry to
    [thrust_min, thrust_max]. Clamping is where a torque the motors cannot deliver saturates.
    """
    wrench = np.concatenate([[thrust_cmd], tau_des_b])
    f_cmd = np.linalg.solve(mixing_matrix(params), wrench)
    return np.clip(f_cmd, params.thrust_min, params.thrust_max)


def motor_lag(f: np.ndarray, f_cmd: np.ndarray, dt: float, time_constant: float) -> np.ndarray:
    """Step 4, the exact solution of f_dot = (f_cmd - f) / tau_m when f_cmd is constant over dt."""
    return f_cmd + (f - f_cmd) * np.exp(-dt / time_constant)


# --------------------------------------------------------------------------
# Stepping
# --------------------------------------------------------------------------


def physics_substep(
    params: DroneParams,
    state: np.ndarray,
    thrust_cmd: float,
    omega_cmd_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One physics substep (handoff section 7, steps 1 to 6). Returns a new 17-value state.

    thrust_cmd is the collective thrust T_cmd in newtons; omega_cmd_b the commanded body rates
    in rad/s. Unlike `policy_step`, nothing here limits those commands, so tests can ask for
    more than the motors can deliver.
    """
    omega_b = state[RATES]
    tau_des_b = rate_controller(params, omega_b, omega_cmd_b)  # step 1
    f_cmd = allocate(params, thrust_cmd, tau_des_b)  # steps 2 and 3
    f = motor_lag(state[THRUST], f_cmd, dt, params.motor_time_constant)  # step 4
    new_state = rk4_step(params, state, f, dt)  # step 5, uses the lagged thrusts
    new_state[THRUST] = f
    new_state[QUAT] = quat_normalize(new_state[QUAT])  # step 6
    return new_state


def action_to_commands(params: DroneParams, action: np.ndarray) -> tuple[float, np.ndarray]:
    """Map an action a in [-1, 1]^4 to (T_cmd, omega_cmd_b) (handoff section 7).

        T_cmd     = (a[0] + 1) / 2 * 4 * f_max
        omega_cmd = a[1:4] .* max_rates

    Actions outside [-1, 1] are clipped first. The handoff does not say what to do with them;
    clipping keeps the commands inside the range the mapping is defined for.
    """
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (4,):
        raise ValueError(f"action must have shape (4,), got {a.shape}")
    a = np.clip(a, -1.0, 1.0)
    thrust_cmd = (a[0] + 1.0) / 2.0 * 4.0 * params.thrust_max
    omega_cmd_b = a[1:4] * params.max_rates_b
    return float(thrust_cmd), omega_cmd_b


def validate_state(state: np.ndarray) -> None:
    """Raise ValueError unless state has 17 finite values and a unit-length quaternion."""
    if state.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains NaN or infinity")
    if abs(np.linalg.norm(state[QUAT]) - 1.0) > 1e-6:
        raise ValueError(f"quaternion is not unit length: norm {np.linalg.norm(state[QUAT])}")


def policy_step(params: DroneParams, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """One 50 Hz policy step: 10 physics substeps with the same action. Returns a new state."""
    state = np.asarray(state, dtype=np.float64)
    validate_state(state)
    thrust_cmd, omega_cmd_b = action_to_commands(params, action)
    dt = 1.0 / params.physics_hz
    for _ in range(substeps_per_action(params)):
        state = physics_substep(params, state, thrust_cmd, omega_cmd_b, dt)
    validate_state(state)
    return state


# --------------------------------------------------------------------------
# Convenience: hover
# --------------------------------------------------------------------------


def hover_action(params: DroneParams) -> np.ndarray:
    """The action that commands exactly hover thrust (T_cmd = m g) and zero body rates."""
    hover_thrust = params.mass * params.gravity
    a0 = 2.0 * hover_thrust / (4.0 * params.thrust_max) - 1.0
    return np.array([a0, 0.0, 0.0, 0.0], dtype=np.float64)


def hover_state(params: DroneParams, position_w: np.ndarray) -> np.ndarray:
    """A level, motionless state at position_w with every motor already at hover thrust m g / 4."""
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[POS] = position_w
    state[QUAT] = [1.0, 0.0, 0.0, 0.0]
    state[THRUST] = params.mass * params.gravity / 4.0
    return state
