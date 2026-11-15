//! Every physical constant the simulator uses, in one struct.
//!
//! No value is hard-coded here: the struct is filled from `configs/drone.yaml` by the
//! Python layer (`lib.rs` reads a dict made with `yaml.safe_load`). The one exception is
//! `test_params()` below, a fixture for Rust unit tests only. The parity tests in Python
//! use the real config.

use crate::math::Vec3;

/// Drone constants, mirroring `DroneParams` in `python/quadrace/ref_sim.py`.
#[derive(Clone, Debug)]
pub struct DroneParams {
    pub mass: f64,                // kg
    pub inertia_diag_b: Vec3,     // kg m^2, diagonal of J in the body frame
    pub arm_length: f64,          // m, centre to motor
    pub thrust_min: f64,          // N per motor
    pub thrust_max: f64,          // N per motor
    pub motor_time_constant: f64, // s, tau_m
    pub yaw_torque_coeff: f64,    // m, kappa
    pub max_rates_b: Vec3,        // rad/s, used by the action mapping
    pub rate_gain_kp: Vec3,       // 1/s
    pub drag_enabled: bool,
    pub drag_linear: Vec3, // N s/m, diagonal of D
    pub gravity: f64,      // m/s^2, acts along world -z
    pub physics_hz: f64,
    pub policy_hz: f64,
}

impl DroneParams {
    /// Physics substeps per policy step: `physics_hz / policy_hz` (10 for 500 / 50).
    /// `validate` guarantees the ratio is a whole number.
    pub fn substeps_per_action(&self) -> usize {
        (self.physics_hz / self.policy_hz).round() as usize
    }

    /// Length of one physics substep in seconds.
    pub fn dt(&self) -> f64 {
        1.0 / self.physics_hz
    }

    /// Each motor sits at `d = arm_length / sqrt(2)` on both the x and y axes (X layout).
    pub fn motor_offset(&self) -> f64 {
        self.arm_length / 2.0_f64.sqrt()
    }

    /// Check the values make physical sense, so bad configs fail with a message instead of NaNs.
    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "drone_cfg: {name} must be a positive number, got {value}"
                ))
            }
        };
        positive("body.mass", self.mass)?;
        positive("body.arm_length", self.arm_length)?;
        positive("motors.time_constant", self.motor_time_constant)?;
        positive("motors.yaw_torque_coeff", self.yaw_torque_coeff)?;
        positive("world.gravity", self.gravity)?;
        positive("timing.physics_hz", self.physics_hz)?;
        positive("timing.policy_hz", self.policy_hz)?;
        for (name, v) in [
            ("body.inertia_diag", self.inertia_diag_b),
            ("limits.max_rates", self.max_rates_b),
            ("controller.rate_gain_kp", self.rate_gain_kp),
        ] {
            positive(&format!("{name}[0]"), v.x)?;
            positive(&format!("{name}[1]"), v.y)?;
            positive(&format!("{name}[2]"), v.z)?;
        }
        if !self.drag_linear.is_finite() {
            return Err("drone_cfg: drag.linear must be finite".to_string());
        }
        if !(self.thrust_min.is_finite() && self.thrust_min >= 0.0) {
            return Err(format!(
                "drone_cfg: motors.thrust_min must be >= 0, got {}",
                self.thrust_min
            ));
        }
        if !(self.thrust_max.is_finite() && self.thrust_max > self.thrust_min) {
            return Err(
                "drone_cfg: motors.thrust_max must be greater than motors.thrust_min".to_string(),
            );
        }
        let ratio = self.physics_hz / self.policy_hz;
        if ratio < 1.0 || (ratio - ratio.round()).abs() > 1e-9 {
            return Err(format!(
                "drone_cfg: timing.physics_hz / timing.policy_hz must be a whole number, got {} / {}",
                self.physics_hz, self.policy_hz
            ));
        }
        Ok(())
    }
}

/// A fixture for Rust unit tests. Same numbers as `configs/drone.yaml` at the time of writing,
/// but nothing checks they stay equal; real runs always read the YAML.
#[cfg(test)]
pub fn test_params() -> DroneParams {
    DroneParams {
        mass: 0.75,
        inertia_diag_b: Vec3::new(0.0025, 0.0021, 0.0043),
        arm_length: 0.15,
        thrust_min: 0.0,
        thrust_max: 8.5,
        motor_time_constant: 0.03,
        yaw_torque_coeff: 0.02,
        max_rates_b: Vec3::new(10.0, 10.0, 4.0),
        rate_gain_kp: Vec3::new(20.0, 20.0, 10.0),
        drag_enabled: false,
        drag_linear: Vec3::new(0.3, 0.3, 0.3),
        gravity: 9.81,
        physics_hz: 500.0,
        policy_hz: 50.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_is_valid_and_has_ten_substeps_per_action() {
        let p = test_params();
        assert!(p.validate().is_ok());
        assert_eq!(p.substeps_per_action(), 10);
    }

    #[test]
    fn validate_rejects_a_non_positive_mass() {
        let mut p = test_params();
        p.mass = 0.0;
        assert!(p.validate().unwrap_err().contains("body.mass"));
    }

    #[test]
    fn validate_rejects_timing_that_is_not_a_whole_number_of_substeps() {
        let mut p = test_params();
        p.policy_hz = 60.0;
        assert!(p.validate().unwrap_err().contains("whole number"));
    }

    #[test]
    fn validate_rejects_thrust_max_not_above_thrust_min() {
        let mut p = test_params();
        p.thrust_max = p.thrust_min;
        assert!(p.validate().is_err());
    }
}
