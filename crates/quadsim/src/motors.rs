//! Motors: where they sit, how thrust turns into torque, how a wanted force and torque turn
//! into four thrust commands, and the first-order lag.
//!
//! Motor numbering and signs (handoff section 6, X layout, `d = arm_length / sqrt(2)`):
//!
//! | motor | x  | y  | spin s |
//! | ----- | -- | -- | ------ |
//! | 0     | +d | -d | +1     |
//! | 1     | -d | +d | +1     |
//! | 2     | +d | +d | -1     |
//! | 3     | -d | -d | -1     |
//!
//! All torques are in the body frame (x forward, y left, z up).

use crate::math::Vec3;
use crate::params::DroneParams;

/// Body-frame x and y of each motor, and its spin sign.
fn motor_layout(params: &DroneParams) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let d = params.motor_offset();
    let x_b = [d, -d, d, -d];
    let y_b = [-d, d, d, -d];
    let spin = [1.0, 1.0, -1.0, -1.0];
    (x_b, y_b, spin)
}

/// Collective thrust `T = sum(f_i)`, in newtons, along the body z axis.
pub fn total_thrust(f: &[f64; 4]) -> f64 {
    f[0] + f[1] + f[2] + f[3]
}

/// Body torque `tau_b` from the four motor thrusts `f`:
///
/// ```text
/// tau_x =  sum(y_i * f_i)            roll
/// tau_y = -sum(x_i * f_i)            pitch
/// tau_z =  sum(s_i * kappa * f_i)    yaw
/// ```
pub fn body_torque(params: &DroneParams, f: &[f64; 4]) -> Vec3 {
    let (x_b, y_b, spin) = motor_layout(params);
    let kappa = params.yaw_torque_coeff;
    let mut tau_x = 0.0;
    let mut tau_y = 0.0;
    let mut tau_z = 0.0;
    for i in 0..4 {
        tau_x += y_b[i] * f[i];
        tau_y -= x_b[i] * f[i];
        tau_z += spin[i] * kappa * f[i];
    }
    Vec3::new(tau_x, tau_y, tau_z)
}

/// Four motor thrust commands from a wanted `[T_cmd, tau_des]` (handoff section 7, steps 2 and 3).
///
/// The mixing matrix `M` has `[T, tau_x, tau_y, tau_z] = M f` with rows
/// `(1, 1, 1, 1)`, `y`, `-x` and `kappa * s`. For this layout those four rows are at right
/// angles to each other (a test checks it), so `M` is inverted by projecting onto each row and
/// dividing by that row's squared length (`4`, `4 d^2`, `4 d^2`, `4 kappa^2`):
///
/// ```text
/// f_i = T/4 + y_i tau_x / (4 d^2) - x_i tau_y / (4 d^2) + s_i tau_z / (4 kappa)
/// ```
///
/// Each result is then clamped to `[thrust_min, thrust_max]`. The clamp is where a torque the
/// motors cannot deliver saturates. The NumPy reference does the same with a general solve.
pub fn allocate(params: &DroneParams, thrust_cmd: f64, tau_des_b: Vec3) -> [f64; 4] {
    let (x_b, y_b, spin) = motor_layout(params);
    let d = params.motor_offset();
    let kappa = params.yaw_torque_coeff;
    let mut f_cmd = [0.0; 4];
    for i in 0..4 {
        let raw = thrust_cmd / 4.0 + y_b[i] * tau_des_b.x / (4.0 * d * d)
            - x_b[i] * tau_des_b.y / (4.0 * d * d)
            + spin[i] * tau_des_b.z / (4.0 * kappa);
        f_cmd[i] = raw.clamp(params.thrust_min, params.thrust_max);
    }
    f_cmd
}

/// Step 4: the exact solution of `f_dot = (f_cmd - f) / tau_m` when `f_cmd` is constant over `dt`.
pub fn motor_lag(f: &[f64; 4], f_cmd: &[f64; 4], dt: f64, time_constant: f64) -> [f64; 4] {
    let decay = (-dt / time_constant).exp();
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = f_cmd[i] + (f[i] - f_cmd[i]) * decay;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::test_params;

    #[test]
    fn equal_thrust_on_all_motors_gives_zero_torque() {
        let tau_b = body_torque(&test_params(), &[3.7; 4]);
        assert!(tau_b.norm() < 1e-15);
        assert!((total_thrust(&[3.7; 4]) - 4.0 * 3.7).abs() < 1e-12);
    }

    #[test]
    fn single_motor_torque_signs_match_the_section_6_table() {
        // Worked out by hand from the table above: (tau_x, tau_y, tau_z) signs per motor.
        let expected = [
            (-1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
        ];
        let p = test_params();
        for (motor, signs) in expected.iter().enumerate() {
            let mut f = [0.0; 4];
            f[motor] = 2.0;
            let tau_b = body_torque(&p, &f);
            assert_eq!(tau_b.x.signum(), signs.0, "motor {motor} roll");
            assert_eq!(tau_b.y.signum(), signs.1, "motor {motor} pitch");
            assert_eq!(tau_b.z.signum(), signs.2, "motor {motor} yaw");
        }
    }

    #[test]
    fn allocation_inverts_the_mixing_matrix_when_no_motor_saturates() {
        let p = test_params();
        let f = [1.5, 4.0, 2.5, 6.0];
        let tau_b = body_torque(&p, &f);
        let f_back = allocate(&p, total_thrust(&f), tau_b);
        for i in 0..4 {
            assert!(
                (f_back[i] - f[i]).abs() < 1e-12,
                "motor {i}: {} vs {}",
                f_back[i],
                f[i]
            );
        }
    }

    #[test]
    fn a_roll_torque_beyond_motor_capability_clamps_to_the_limits() {
        // Positive roll torque comes from the motors with y = +d (1 and 2).
        let p = test_params();
        let f_cmd = allocate(&p, p.mass * p.gravity, Vec3::new(1000.0, 0.0, 0.0));
        assert_eq!(
            f_cmd,
            [p.thrust_min, p.thrust_max, p.thrust_max, p.thrust_min]
        );
    }

    #[test]
    fn motor_lag_moves_a_fraction_of_the_way_to_the_command() {
        // One time constant of lag closes 1 - 1/e of the gap.
        let f = motor_lag(&[0.0; 4], &[1.0; 4], 0.03, 0.03);
        for fi in f {
            assert!((fi - (1.0 - (-1.0_f64).exp())).abs() < 1e-15);
        }
    }
}
