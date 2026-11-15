//! The rigid body, RK4, and one physics substep / one policy step.
//!
//! This file is a line-by-line port of `python/quadrace/ref_sim.py`. If the two disagree, the
//! parity test (`tests/test_parity.py`) fails and the bug is in this file.
//!
//! One physics substep does, in order (handoff section 7):
//!
//! 1. Rate controller: from the commanded body rates, the desired body torque.
//! 2. Allocation: invert the mixing matrix to get four thrust commands.
//! 3. Clamp each thrust command to `[thrust_min, thrust_max]`.
//! 4. Motor lag, exact update.
//! 5. RK4 on position, velocity, quaternion and body rates, motor thrusts held constant.
//! 6. Renormalize the quaternion.
//!
//! Nothing here allocates on the heap: every value lives on the stack.

use crate::controller::rate_controller;
use crate::math::{Quat, Vec3};
use crate::motors::{allocate, body_torque, motor_lag, total_thrust};
use crate::params::DroneParams;

/// Number of values in the flat state used at the Python boundary.
pub const STATE_SIZE: usize = 17;

/// The full state of one drone (17 values, handoff section 6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub pos_w: Vec3,      // position, world frame, m
    pub quat: Quat,       // attitude, body to world
    pub vel_w: Vec3,      // velocity, world frame, m/s
    pub omega_b: Vec3,    // angular velocity, body frame, rad/s
    pub thrust: [f64; 4], // the four motor thrusts, N
}

impl State {
    /// A level, motionless state at `pos_w` with every motor already at `thrust` newtons.
    pub fn level_at(pos_w: Vec3, thrust: f64) -> State {
        State {
            pos_w,
            quat: Quat::IDENTITY,
            vel_w: Vec3::ZERO,
            omega_b: Vec3::ZERO,
            thrust: [thrust; 4],
        }
    }

    /// Build a state from the flat layout `[pos(3), quat(4), vel(3), omega(3), thrust(4)]`.
    /// Returns `None` unless the slice has exactly 17 values.
    pub fn from_slice(s: &[f64]) -> Option<State> {
        if s.len() != STATE_SIZE {
            return None;
        }
        Some(State {
            pos_w: Vec3::new(s[0], s[1], s[2]),
            quat: Quat::new(s[3], s[4], s[5], s[6]),
            vel_w: Vec3::new(s[7], s[8], s[9]),
            omega_b: Vec3::new(s[10], s[11], s[12]),
            thrust: [s[13], s[14], s[15], s[16]],
        })
    }

    /// Write the state into a 17-value slice, in the same flat layout.
    /// Does nothing unless the slice has exactly 17 values.
    pub fn write_to_slice(&self, out: &mut [f64]) {
        if out.len() != STATE_SIZE {
            return;
        }
        out[0..3].copy_from_slice(&[self.pos_w.x, self.pos_w.y, self.pos_w.z]);
        out[3..7].copy_from_slice(&[self.quat.w, self.quat.x, self.quat.y, self.quat.z]);
        out[7..10].copy_from_slice(&[self.vel_w.x, self.vel_w.y, self.vel_w.z]);
        out[10..13].copy_from_slice(&[self.omega_b.x, self.omega_b.y, self.omega_b.z]);
        out[13..17].copy_from_slice(&self.thrust);
    }

    /// Why this state is not usable, or `Ok` if it has only finite values and a unit quaternion.
    pub fn validate(&self) -> Result<(), String> {
        let finite = self.pos_w.is_finite()
            && self.quat.is_finite()
            && self.vel_w.is_finite()
            && self.omega_b.is_finite()
            && self.thrust.iter().all(|t| t.is_finite());
        if !finite {
            return Err("state contains NaN or infinity".to_string());
        }
        if (self.quat.norm() - 1.0).abs() > 1e-6 {
            return Err(format!(
                "quaternion is not unit length: norm {}",
                self.quat.norm()
            ));
        }
        Ok(())
    }

    /// `self + h * d` for position, quaternion, velocity and rates. The motor thrusts are
    /// copied unchanged, and the quaternion is NOT renormalized here.
    fn advanced(&self, d: &Derivative, h: f64) -> State {
        State {
            pos_w: self.pos_w + d.pos_dot * h,
            quat: self.quat + d.quat_dot * h,
            vel_w: self.vel_w + d.vel_dot * h,
            omega_b: self.omega_b + d.omega_dot * h,
            thrust: self.thrust,
        }
    }
}

/// Time derivative of the rigid-body part of the state.
#[derive(Clone, Copy, Debug)]
struct Derivative {
    pos_dot: Vec3,
    quat_dot: Quat,
    vel_dot: Vec3,
    omega_dot: Vec3,
}

/// Time derivative of position, quaternion, velocity and body rates. The motor thrusts
/// `state.thrust` are inputs here (constant over one RK4 step).
///
/// ```text
/// p_dot       = v_w
/// v_dot_w     = g_w + (1/m) R [0, 0, T] - (1/m) R D R^T v_w     (drag only if enabled)
/// q_dot       = 0.5 * q (x) (0, omega_b)
/// J omega_dot = tau_b - omega_b x (J omega_b)                    (Euler's equation, body frame)
/// ```
fn rigid_body_derivative(params: &DroneParams, state: &State) -> Derivative {
    let rot = state.quat.to_rotation_matrix(); // body to world

    // Newton's second law in the world frame: gravity plus thrust along body z, minus drag.
    let gravity_w = Vec3::new(0.0, 0.0, -params.gravity);
    let thrust_w = rot.mul_vec(Vec3::new(0.0, 0.0, total_thrust(&state.thrust)));
    let mut accel_w = gravity_w + thrust_w / params.mass;
    if params.drag_enabled {
        let vel_b = rot.transpose_mul_vec(state.vel_w); // R^T v_w
        let drag_w = rot.mul_vec(params.drag_linear.mul_elementwise(vel_b)); // R D R^T v_w
        accel_w = accel_w - drag_w / params.mass;
    }

    // Quaternion kinematics: omega is in the body frame, so it multiplies q on the right.
    let omega_quat = Quat::new(0.0, state.omega_b.x, state.omega_b.y, state.omega_b.z);
    let quat_dot = state.quat.multiply(omega_quat) * 0.5;

    // Euler's rotation equation for a rigid body, all in the body frame.
    let inertia_omega_b = params.inertia_diag_b.mul_elementwise(state.omega_b); // J omega
    let omega_dot_b = (body_torque(params, &state.thrust) - state.omega_b.cross(inertia_omega_b))
        .div_elementwise(params.inertia_diag_b);

    Derivative {
        pos_dot: state.vel_w,
        quat_dot,
        vel_dot: accel_w,
        omega_dot: omega_dot_b,
    }
}

/// One classical RK4 step with the motor thrusts in `state` held constant (step 5).
/// The quaternion is not renormalized here; `physics_substep` does that (step 6).
fn rk4_step(params: &DroneParams, state: &State, dt: f64) -> State {
    let k1 = rigid_body_derivative(params, state);
    let k2 = rigid_body_derivative(params, &state.advanced(&k1, 0.5 * dt));
    let k3 = rigid_body_derivative(params, &state.advanced(&k2, 0.5 * dt));
    let k4 = rigid_body_derivative(params, &state.advanced(&k3, dt));
    // (k1 + 2 k2 + 2 k3 + k4), then scaled by dt / 6.
    let blend = Derivative {
        pos_dot: k1.pos_dot + 2.0 * k2.pos_dot + 2.0 * k3.pos_dot + k4.pos_dot,
        quat_dot: k1.quat_dot + k2.quat_dot * 2.0 + k3.quat_dot * 2.0 + k4.quat_dot,
        vel_dot: k1.vel_dot + 2.0 * k2.vel_dot + 2.0 * k3.vel_dot + k4.vel_dot,
        omega_dot: k1.omega_dot + 2.0 * k2.omega_dot + 2.0 * k3.omega_dot + k4.omega_dot,
    };
    state.advanced(&blend, dt / 6.0)
}

/// One physics substep (handoff section 7, steps 1 to 6).
///
/// `thrust_cmd` is the collective thrust `T_cmd` in newtons and `omega_cmd_b` the commanded
/// body rates in rad/s. Nothing here limits those commands, so callers can ask for more than
/// the motors can deliver.
pub fn physics_substep(
    params: &DroneParams,
    state: &State,
    thrust_cmd: f64,
    omega_cmd_b: Vec3,
    dt: f64,
) -> State {
    let tau_des_b = rate_controller(params, state.omega_b, omega_cmd_b); // step 1
    let f_cmd = allocate(params, thrust_cmd, tau_des_b); // steps 2 and 3
    let lagged = State {
        thrust: motor_lag(&state.thrust, &f_cmd, dt, params.motor_time_constant), // step 4
        ..*state
    };
    let mut next = rk4_step(params, &lagged, dt); // step 5, uses the lagged thrusts
    next.quat = next.quat.normalized(); // step 6
    next
}

/// Map an action `a` in `[-1, 1]^4` to `(T_cmd, omega_cmd_b)` (handoff section 7):
///
/// ```text
/// T_cmd     = (a[0] + 1) / 2 * 4 * f_max
/// omega_cmd = a[1..4] .* max_rates
/// ```
///
/// Actions outside `[-1, 1]` are clipped first. The handoff does not say what to do with them,
/// and the NumPy reference clips too.
pub fn action_to_commands(params: &DroneParams, action: [f64; 4]) -> (f64, Vec3) {
    let a = action.map(|v| v.clamp(-1.0, 1.0));
    let thrust_cmd = (a[0] + 1.0) / 2.0 * 4.0 * params.thrust_max;
    let omega_cmd_b = Vec3::new(a[1], a[2], a[3]).mul_elementwise(params.max_rates_b);
    (thrust_cmd, omega_cmd_b)
}

/// One 50 Hz policy step: `substeps_per_action` physics substeps with the same action.
pub fn policy_step(params: &DroneParams, state: &State, action: [f64; 4]) -> State {
    let (thrust_cmd, omega_cmd_b) = action_to_commands(params, action);
    let dt = params.dt();
    let mut current = *state;
    for _ in 0..params.substeps_per_action() {
        current = physics_substep(params, &current, thrust_cmd, omega_cmd_b, dt);
    }
    debug_assert!(
        current.validate().is_ok(),
        "state went bad: {:?}",
        current.validate()
    );
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::test_params;

    fn hover_action(params: &DroneParams) -> [f64; 4] {
        let hover_thrust = params.mass * params.gravity;
        [
            2.0 * hover_thrust / (4.0 * params.thrust_max) - 1.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    #[test]
    fn free_fall_with_motors_off_follows_z0_minus_4_905_t_squared() {
        let p = test_params();
        let mut state = State::level_at(Vec3::new(1.0, 2.0, 10.0), 0.0);
        for step in 1..=50 {
            state = policy_step(&p, &state, [-1.0, 0.0, 0.0, 0.0]);
            let t = step as f64 / 50.0;
            assert!(
                (state.pos_w.z - (10.0 - 4.905 * t * t)).abs() < 1e-9,
                "step {step}"
            );
        }
        assert_eq!((state.pos_w.x, state.pos_w.y), (1.0, 2.0));
    }

    #[test]
    fn hover_thrust_holds_altitude_within_1_mm_over_5_s() {
        let p = test_params();
        let mut state = State::level_at(Vec3::new(0.0, 0.0, 1.5), p.mass * p.gravity / 4.0);
        let action = hover_action(&p);
        for _ in 0..250 {
            state = policy_step(&p, &state, action);
            assert!((state.pos_w.z - 1.5).abs() < 1e-3);
        }
    }

    #[test]
    fn substep_runs_motor_lag_before_rk4_so_rk4_sees_the_lagged_thrust() {
        // From rest with motors at zero and hover thrust commanded, one substep lifts every
        // motor to f_cmd (1 - exp(-dt / tau_m)) and RK4 then sees that thrust for the whole
        // substep. Level and torque-free, so dv_z = dt * (T / m - g) exactly.
        let p = test_params();
        let dt = p.dt();
        let hover_thrust = p.mass * p.gravity;
        let after = physics_substep(
            &p,
            &State::level_at(Vec3::ZERO, 0.0),
            hover_thrust,
            Vec3::ZERO,
            dt,
        );
        let lagged_thrust = hover_thrust * (1.0 - (-dt / p.motor_time_constant).exp());
        assert!((total_thrust(&after.thrust) - lagged_thrust).abs() < 1e-12);
        assert!((after.vel_w.z - dt * (lagged_thrust / p.mass - p.gravity)).abs() < 1e-12);
    }

    #[test]
    fn quaternion_stays_unit_length_over_many_policy_steps() {
        let p = test_params();
        let mut state = State::level_at(Vec3::new(0.0, 0.0, 5.0), p.mass * p.gravity / 4.0);
        for i in 0..200 {
            let phase = i as f64 * 0.37;
            let action = [
                0.2 * phase.sin(),
                phase.cos(),
                (1.7 * phase).sin(),
                (0.5 * phase).cos(),
            ];
            state = policy_step(&p, &state, action);
            assert!((state.quat.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn constant_omega_with_equal_inertias_matches_the_exact_quaternion_formula() {
        // Equal inertias make omega x (J omega) zero. The controller is given omega as its
        // command and T_cmd = 0, so it asks for zero torque and omega stays constant.
        let mut p = test_params();
        p.inertia_diag_b = Vec3::new(0.003, 0.003, 0.003);
        let omega_b = Vec3::new(1.5, -2.0, 2.5);
        let q0 = Quat::new(0.4, -0.3, 0.8, 0.1).normalized();
        let mut state = State::level_at(Vec3::ZERO, 0.0);
        state.quat = q0;
        state.omega_b = omega_b;
        let dt = p.dt();
        for _ in 0..500 {
            state = physics_substep(&p, &state, 0.0, omega_b, dt);
        }
        let half_angle = omega_b.norm() * dt * 500.0 / 2.0;
        let axis = omega_b / omega_b.norm();
        let expected = q0.multiply(Quat::new(
            half_angle.cos(),
            half_angle.sin() * axis.x,
            half_angle.sin() * axis.y,
            half_angle.sin() * axis.z,
        ));
        // q and -q are the same rotation, so compare up to sign.
        let plus = (state.quat.w - expected.w)
            .abs()
            .max((state.quat.x - expected.x).abs());
        let rest = (state.quat.y - expected.y)
            .abs()
            .max((state.quat.z - expected.z).abs());
        assert!(plus.max(rest) < 1e-9);
    }

    #[test]
    fn flat_state_round_trips_through_a_slice() {
        let state = State {
            pos_w: Vec3::new(1.0, 2.0, 3.0),
            quat: Quat::new(0.5, 0.5, 0.5, 0.5),
            vel_w: Vec3::new(4.0, 5.0, 6.0),
            omega_b: Vec3::new(7.0, 8.0, 9.0),
            thrust: [1.0, 2.0, 3.0, 4.0],
        };
        let mut flat = [0.0; STATE_SIZE];
        state.write_to_slice(&mut flat);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[3], 0.5);
        assert_eq!(flat[16], 4.0);
        assert_eq!(State::from_slice(&flat), Some(state));
    }

    #[test]
    fn from_slice_rejects_the_wrong_length() {
        assert_eq!(State::from_slice(&[0.0; 16]), None);
    }

    #[test]
    fn validate_rejects_nan_and_a_non_unit_quaternion() {
        let mut state = State::level_at(Vec3::ZERO, 1.0);
        assert!(state.validate().is_ok());
        state.quat = Quat::new(2.0, 0.0, 0.0, 0.0);
        assert!(state.validate().unwrap_err().contains("unit length"));
        state.quat = Quat::IDENTITY;
        state.pos_w.x = f64::NAN;
        assert!(state.validate().unwrap_err().contains("NaN"));
    }

    #[test]
    fn actions_outside_minus_one_to_one_are_clipped() {
        let p = test_params();
        let wild = action_to_commands(&p, [7.0, -3.0, 2.0, 9.0]);
        let limit = action_to_commands(&p, [1.0, -1.0, 1.0, 1.0]);
        assert_eq!(wild, limit);
    }

    #[test]
    fn action_mapping_matches_handoff_section_7() {
        let p = test_params();
        let (thrust_cmd, omega_cmd_b) = action_to_commands(&p, [0.5, 1.0, -1.0, 0.25]);
        assert!((thrust_cmd - 0.75 * 4.0 * 8.5).abs() < 1e-12);
        assert_eq!(omega_cmd_b, Vec3::new(10.0, -10.0, 1.0));
    }
}
