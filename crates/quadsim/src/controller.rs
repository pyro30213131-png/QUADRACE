//! The body-rate controller (handoff section 7, step 1). It runs at the physics rate, inside
//! every substep, so the policy only has to say which rates it wants.

use crate::math::Vec3;
use crate::params::DroneParams;

/// Desired body torque `tau_des_b`, all vectors in the body frame:
///
/// ```text
/// tau_des = J (Kp .* (omega_cmd - omega)) + omega x (J omega)
/// ```
///
/// The first term asks for an angular acceleration proportional to the rate error. The second
/// cancels the gyroscopic term in Euler's equation.
pub fn rate_controller(params: &DroneParams, omega_b: Vec3, omega_cmd_b: Vec3) -> Vec3 {
    let angular_accel_des_b = params.rate_gain_kp.mul_elementwise(omega_cmd_b - omega_b);
    let inertia_omega_b = params.inertia_diag_b.mul_elementwise(omega_b);
    params.inertia_diag_b.mul_elementwise(angular_accel_des_b) + omega_b.cross(inertia_omega_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::test_params;

    #[test]
    fn no_rate_error_and_no_spin_asks_for_no_torque() {
        let tau_b = rate_controller(&test_params(), Vec3::ZERO, Vec3::ZERO);
        assert_eq!(tau_b, Vec3::ZERO);
    }

    #[test]
    fn rate_error_on_roll_asks_for_roll_torque_only() {
        // J_xx * Kp_x * error = 0.0025 * 20 * 2 = 0.1 N m, nothing on the other axes.
        let tau_b = rate_controller(&test_params(), Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        assert!((tau_b.x - 0.1).abs() < 1e-15);
        assert_eq!((tau_b.y, tau_b.z), (0.0, 0.0));
    }
}
