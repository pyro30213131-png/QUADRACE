//! Many environments stepped together: one `Vec` of `Env`, one shared set of drone parameters.
//!
//! Single-threaded for now. Threads (rayon) are added after the parity test passes.

use crate::dynamics::{STATE_SIZE, State};
use crate::env::Env;
use crate::params::DroneParams;

/// `num_envs` independent drones that share one set of drone parameters.
pub struct Batch {
    params: DroneParams,
    envs: Vec<Env>,
}

impl Batch {
    /// Build `num_envs` drones, each level and motionless at the origin with hover thrust.
    pub fn new(num_envs: usize, params: DroneParams) -> Result<Batch, String> {
        params.validate()?;
        if num_envs == 0 {
            return Err("num_envs must be at least 1".to_string());
        }
        let hover_thrust_per_motor = params.mass * params.gravity / 4.0;
        let start = State::level_at(crate::math::Vec3::ZERO, hover_thrust_per_motor);
        Ok(Batch {
            params,
            envs: vec![Env::new(start); num_envs],
        })
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Step every environment once. `actions` holds `num_envs * 4` values, one action per
    /// environment laid out row by row, each in `[-1, 1]` (anything outside is clipped).
    pub fn step(&mut self, actions: &[f32]) -> Result<(), String> {
        if actions.len() != self.envs.len() * 4 {
            return Err(format!(
                "actions must hold num_envs * 4 = {} values, got {}",
                self.envs.len() * 4,
                actions.len()
            ));
        }
        let params = &self.params;
        for (env, a) in self.envs.iter_mut().zip(actions.chunks_exact(4)) {
            let action = [
                f64::from(a[0]),
                f64::from(a[1]),
                f64::from(a[2]),
                f64::from(a[3]),
            ];
            env.step_physics(params, action);
        }
        Ok(())
    }

    /// Why the first bad state is bad, if any environment has a NaN or a non-unit quaternion.
    pub fn check_states(&self) -> Result<(), String> {
        for (i, env) in self.envs.iter().enumerate() {
            env.state
                .validate()
                .map_err(|why| format!("environment {i}: {why}"))?;
        }
        Ok(())
    }

    /// Copy every state into `out`, `num_envs * 17` values laid out row by row.
    pub fn write_states(&self, out: &mut [f64]) -> Result<(), String> {
        if out.len() != self.envs.len() * STATE_SIZE {
            return Err(format!(
                "output must hold num_envs * {STATE_SIZE} = {} values, got {}",
                self.envs.len() * STATE_SIZE,
                out.len()
            ));
        }
        for (env, row) in self.envs.iter().zip(out.chunks_exact_mut(STATE_SIZE)) {
            env.state.write_to_slice(row);
        }
        Ok(())
    }

    /// Replace every state from `states`, `num_envs * 17` values laid out row by row.
    /// Nothing changes unless every state is finite and has a unit quaternion.
    pub fn set_states(&mut self, states: &[f64]) -> Result<(), String> {
        if states.len() != self.envs.len() * STATE_SIZE {
            return Err(format!(
                "states must hold num_envs * {STATE_SIZE} = {} values, got {}",
                self.envs.len() * STATE_SIZE,
                states.len()
            ));
        }
        let mut new_states = Vec::with_capacity(self.envs.len());
        for (i, row) in states.chunks_exact(STATE_SIZE).enumerate() {
            let state = State::from_slice(row).ok_or("state row has the wrong length")?;
            state
                .validate()
                .map_err(|why| format!("environment {i}: {why}"))?;
            new_states.push(state);
        }
        for (env, state) in self.envs.iter_mut().zip(new_states) {
            env.state = state;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::test_params;

    #[test]
    fn new_batch_has_the_requested_number_of_envs_at_hover() {
        let batch = Batch::new(3, test_params()).unwrap();
        assert_eq!(batch.num_envs(), 3);
        let mut flat = vec![0.0; 3 * STATE_SIZE];
        batch.write_states(&mut flat).unwrap();
        assert_eq!(flat[3], 1.0); // quaternion w of env 0: level
        assert_eq!(flat[6], 0.0); // quaternion z of env 0: no yaw
        assert_eq!(flat[13], test_params().mass * 9.81 / 4.0); // motor 0 at hover thrust
    }

    #[test]
    fn zero_envs_is_rejected() {
        assert!(Batch::new(0, test_params()).is_err());
    }

    #[test]
    fn step_rejects_the_wrong_number_of_actions() {
        let mut batch = Batch::new(2, test_params()).unwrap();
        assert!(batch.step(&[0.0; 7]).unwrap_err().contains("num_envs * 4"));
    }

    #[test]
    fn set_states_rejects_a_bad_state_and_leaves_the_batch_unchanged() {
        let mut batch = Batch::new(2, test_params()).unwrap();
        let mut before = vec![0.0; 2 * STATE_SIZE];
        batch.write_states(&mut before).unwrap();

        let mut bad = before.clone();
        bad[STATE_SIZE + 3] = 2.0; // env 1: quaternion w = 2, not unit length
        assert!(
            batch
                .set_states(&bad)
                .unwrap_err()
                .contains("environment 1")
        );

        let mut after = vec![0.0; 2 * STATE_SIZE];
        batch.write_states(&mut after).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn envs_do_not_affect_each_other() {
        // The same action on a batch of two equals stepping each drone alone.
        let p = test_params();
        let mut pair = Batch::new(2, p.clone()).unwrap();
        let mut solo = Batch::new(1, p).unwrap();
        let action = [0.1_f32, 0.2, -0.3, 0.4];
        pair.step(&[action, action].concat()).unwrap();
        solo.step(&action).unwrap();

        let mut pair_flat = vec![0.0; 2 * STATE_SIZE];
        let mut solo_flat = vec![0.0; STATE_SIZE];
        pair.write_states(&mut pair_flat).unwrap();
        solo.write_states(&mut solo_flat).unwrap();
        assert_eq!(&pair_flat[..STATE_SIZE], &solo_flat[..]);
        assert_eq!(&pair_flat[STATE_SIZE..], &solo_flat[..]);
    }
}
