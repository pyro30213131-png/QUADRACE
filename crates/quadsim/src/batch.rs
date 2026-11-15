//! Many environments stepped together: one `Vec` of `Env`, one shared set of drone parameters.
//!
//! `step` spreads the environments over a pool of threads (rayon). That cannot change any
//! result: each environment is stepped by the same code using only its own state and its own
//! action, so which thread runs it does not matter. `step_sequential` does the same work on the
//! calling thread, and the tests check that the two agree bit for bit.

use crate::dynamics::{STATE_SIZE, State};
use crate::env::Env;
use crate::params::DroneParams;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// `num_envs` independent drones that share one set of drone parameters.
pub struct Batch {
    params: DroneParams,
    envs: Vec<Env>,
    pool: ThreadPool,
}

/// One action as the f64 values the physics uses. `a` must hold at least 4 values.
fn action_from_f32(a: &[f32]) -> [f64; 4] {
    [
        f64::from(a[0]),
        f64::from(a[1]),
        f64::from(a[2]),
        f64::from(a[3]),
    ]
}

impl Batch {
    /// Build `num_envs` drones, each level and motionless at the origin with hover thrust.
    ///
    /// `num_threads` is the size of this batch's own thread pool; `None` lets rayon choose
    /// (one thread per logical CPU, or the `RAYON_NUM_THREADS` environment variable).
    pub fn new(
        num_envs: usize,
        params: DroneParams,
        num_threads: Option<usize>,
    ) -> Result<Batch, String> {
        params.validate()?;
        if num_envs == 0 {
            return Err("num_envs must be at least 1".to_string());
        }
        let mut builder = ThreadPoolBuilder::new();
        if let Some(n) = num_threads {
            if n == 0 {
                return Err("num_threads must be at least 1".to_string());
            }
            builder = builder.num_threads(n);
        }
        let pool = builder
            .build()
            .map_err(|e| format!("could not start the thread pool: {e}"))?;
        let hover_thrust_per_motor = params.mass * params.gravity / 4.0;
        let start = State::level_at(crate::math::Vec3::ZERO, hover_thrust_per_motor);
        Ok(Batch {
            params,
            envs: vec![Env::new(start); num_envs],
            pool,
        })
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// How many threads this batch steps on.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    fn check_action_count(&self, actions: &[f32]) -> Result<(), String> {
        if actions.len() != self.envs.len() * 4 {
            return Err(format!(
                "actions must hold num_envs * 4 = {} values, got {}",
                self.envs.len() * 4,
                actions.len()
            ));
        }
        Ok(())
    }

    /// Step every environment once, spread over the thread pool. `actions` holds
    /// `num_envs * 4` values, one action per environment laid out row by row, each in
    /// `[-1, 1]` (anything outside is clipped).
    pub fn step(&mut self, actions: &[f32]) -> Result<(), String> {
        self.check_action_count(actions)?;
        let Batch { params, envs, pool } = self;
        let params = &*params;
        pool.install(|| {
            envs.par_iter_mut()
                .zip(actions.par_chunks_exact(4))
                .for_each(|(env, a)| env.step_physics(params, action_from_f32(a)));
        });
        Ok(())
    }

    /// The same as `step`, on the calling thread only. It is the reference the thread-count
    /// tests compare against.
    pub fn step_sequential(&mut self, actions: &[f32]) -> Result<(), String> {
        self.check_action_count(actions)?;
        let params = &self.params;
        for (env, a) in self.envs.iter_mut().zip(actions.chunks_exact(4)) {
            env.step_physics(params, action_from_f32(a));
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
        let batch = Batch::new(3, test_params(), Some(2)).unwrap();
        assert_eq!(batch.num_envs(), 3);
        assert_eq!(batch.num_threads(), 2);
        let mut flat = vec![0.0; 3 * STATE_SIZE];
        batch.write_states(&mut flat).unwrap();
        assert_eq!(flat[3], 1.0); // quaternion w of env 0: level
        assert_eq!(flat[6], 0.0); // quaternion z of env 0: no yaw
        assert_eq!(flat[13], test_params().mass * 9.81 / 4.0); // motor 0 at hover thrust
    }

    #[test]
    fn zero_envs_is_rejected() {
        assert!(Batch::new(0, test_params(), Some(1)).is_err());
    }

    #[test]
    fn zero_threads_is_rejected() {
        let err = Batch::new(1, test_params(), Some(0)).err().unwrap();
        assert!(err.contains("num_threads"));
    }

    #[test]
    fn step_rejects_the_wrong_number_of_actions() {
        let mut batch = Batch::new(2, test_params(), Some(1)).unwrap();
        assert!(batch.step(&[0.0; 7]).unwrap_err().contains("num_envs * 4"));
        assert!(
            batch
                .step_sequential(&[0.0; 7])
                .unwrap_err()
                .contains("num_envs * 4")
        );
    }

    #[test]
    fn set_states_rejects_a_bad_state_and_leaves_the_batch_unchanged() {
        let mut batch = Batch::new(2, test_params(), Some(1)).unwrap();
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
        let mut pair = Batch::new(2, p.clone(), Some(2)).unwrap();
        let mut solo = Batch::new(1, p, Some(1)).unwrap();
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

    /// `n` different starting states, made from sines and cosines so no random generator is
    /// needed: varied positions, attitudes, velocities, rates and motor thrusts.
    fn varied_states(n: usize) -> Vec<f64> {
        let mut flat = vec![0.0; n * STATE_SIZE];
        for (i, row) in flat.chunks_exact_mut(STATE_SIZE).enumerate() {
            let a = i as f64;
            let q =
                crate::math::Quat::new(1.0 + a.cos(), a.sin(), (2.0 * a).sin(), (3.0 * a).cos())
                    .normalized();
            row[0..3].copy_from_slice(&[a.sin() * 3.0, a.cos() * 3.0, 1.0 + (0.5 * a).sin()]);
            row[3..7].copy_from_slice(&[q.w, q.x, q.y, q.z]);
            row[7..10].copy_from_slice(&[
                (1.3 * a).sin() * 2.0,
                (0.7 * a).cos() * 2.0,
                0.5 * a.sin(),
            ]);
            row[10..13].copy_from_slice(&[
                (0.9 * a).sin() * 3.0,
                (1.1 * a).cos() * 3.0,
                (0.3 * a).sin() * 2.0,
            ]);
            row[13..17].copy_from_slice(&[
                4.0 + a.sin() * 3.0,
                4.0 + a.cos() * 3.0,
                4.0 + (2.0 * a).sin() * 3.0,
                4.0 + (2.0 * a).cos() * 3.0,
            ]);
        }
        flat
    }

    /// Actions that differ per environment and per step, in and slightly beyond `[-1, 1]`.
    fn varied_actions(step: usize, n: usize) -> Vec<f32> {
        (0..n * 4)
            .map(|k| ((step * 7 + k) as f32 * 0.61).sin() * 1.2)
            .collect()
    }

    fn state_bits(batch: &Batch) -> Vec<u64> {
        let mut flat = vec![0.0; batch.num_envs() * STATE_SIZE];
        batch.write_states(&mut flat).unwrap();
        flat.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn results_are_identical_bit_for_bit_for_1_2_8_and_default_threads_and_sequential() {
        let n = 100;
        let start = varied_states(n);
        let mut sequential = Batch::new(n, test_params(), Some(1)).unwrap();
        sequential.set_states(&start).unwrap();
        let mut threaded: Vec<Batch> = [Some(1), Some(2), Some(8), None]
            .into_iter()
            .map(|threads| {
                let mut b = Batch::new(n, test_params(), threads).unwrap();
                b.set_states(&start).unwrap();
                b
            })
            .collect();

        for step in 0..50 {
            let actions = varied_actions(step, n);
            sequential.step_sequential(&actions).unwrap();
            for batch in threaded.iter_mut() {
                batch.step(&actions).unwrap();
            }
        }

        let reference = state_bits(&sequential);
        for batch in &threaded {
            assert_eq!(
                state_bits(batch),
                reference,
                "{} threads differ from the sequential result",
                batch.num_threads()
            );
        }
    }

    #[test]
    fn the_thread_pool_has_the_requested_size() {
        for threads in [1, 3, 8] {
            let batch = Batch::new(4, test_params(), Some(threads)).unwrap();
            assert_eq!(batch.num_threads(), threads);
        }
    }
}
