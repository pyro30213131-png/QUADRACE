//! Many environments stepped together: one `Vec` of `Env`, one shared set of drone parameters.
//!
//! `step` spreads the environments over a pool of threads (rayon). That cannot change any
//! result: each environment is stepped by the same code using only its own state and its own
//! action, so which thread runs it does not matter. `step_sequential` does the same work on the
//! calling thread, and the tests check that the two agree bit for bit.
//!
//! A batch can be built with or without a task (the hover task so far). Without one it only does
//! physics (`step`). With one, `reset_task` and `step_task` also give observations, rewards and
//! episode ends, and finished environments restart by themselves.

use crate::dynamics::{STATE_SIZE, State};
use crate::env::{Env, HOVER_OBS_DIM, HoverConfig, NUM_REWARD_TERMS, env_seed};
use crate::params::DroneParams;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// `num_envs` independent drones that share one set of drone parameters (and one task).
pub struct Batch {
    params: DroneParams,
    task: Option<HoverConfig>,
    envs: Vec<Env>,
    pool: ThreadPool,
}

/// Where `step_task` writes its results. Each slice is a preallocated output array, and each row
/// belongs to one environment: `obs` and `final_obs` hold `num_envs * obs_dim` values,
/// `reward_terms` holds `num_envs * 5`, and the others hold `num_envs`.
pub struct StepBuffers<'a> {
    pub obs: &'a mut [f32],
    pub reward: &'a mut [f32],
    pub terminated: &'a mut [bool],
    pub truncated: &'a mut [bool],
    pub final_obs: &'a mut [f32],
    pub reward_terms: &'a mut [f32],
    pub cause: &'a mut [u8],
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
    /// Build `num_envs` drones for physics only (no task), each level and motionless at the
    /// origin with hover thrust.
    ///
    /// `num_threads` is the size of this batch's own thread pool; `None` lets rayon choose
    /// (one thread per logical CPU, or the `RAYON_NUM_THREADS` environment variable).
    pub fn new(
        num_envs: usize,
        params: DroneParams,
        num_threads: Option<usize>,
    ) -> Result<Batch, String> {
        Batch::build(num_envs, params, None, 0, num_threads)
    }

    /// Like `new`, but the drones fly the hover `task`, and each has its own random generator
    /// seeded from `seed` and its index. Call `reset_task` before the first `step_task`.
    pub fn with_task(
        num_envs: usize,
        params: DroneParams,
        task: HoverConfig,
        seed: u64,
        num_threads: Option<usize>,
    ) -> Result<Batch, String> {
        task.validate(&params)?;
        Batch::build(num_envs, params, Some(task), seed, num_threads)
    }

    fn build(
        num_envs: usize,
        params: DroneParams,
        task: Option<HoverConfig>,
        seed: u64,
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
        let envs = (0..num_envs)
            .map(|i| {
                let mut env = Env::new(start, env_seed(seed, i));
                env.set_state(start, task.as_ref());
                env
            })
            .collect();
        Ok(Batch {
            params,
            task,
            envs,
            pool,
        })
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Length of one observation, or `None` for a physics-only batch.
    pub fn obs_dim(&self) -> Option<usize> {
        self.task.as_ref().map(|_| HOVER_OBS_DIM)
    }

    pub fn has_task(&self) -> bool {
        self.task.is_some()
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
        let Batch {
            params, envs, pool, ..
        } = self;
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

    /// Start a new episode in every environment and write the first observations into `obs_out`
    /// (`num_envs * obs_dim` values). With `Some(seed)` every environment's random generator is
    /// first restarted from it, so the same seed always gives the same episodes.
    pub fn reset_task(&mut self, seed: Option<u64>, obs_out: &mut [f32]) -> Result<(), String> {
        let task = self
            .task
            .as_ref()
            .ok_or("this batch has no task: build it with a track_cfg")?;
        if obs_out.len() != self.envs.len() * HOVER_OBS_DIM {
            return Err(format!(
                "obs output must hold num_envs * {HOVER_OBS_DIM} = {} values, got {}",
                self.envs.len() * HOVER_OBS_DIM,
                obs_out.len()
            ));
        }
        let Batch {
            params, envs, pool, ..
        } = self;
        let params = &*params;
        pool.install(|| {
            envs.par_iter_mut().enumerate().for_each(|(i, env)| {
                if let Some(seed) = seed {
                    env.reseed(env_seed(seed, i));
                }
                env.reset_hover(params, task);
            });
        });
        for (env, row) in self
            .envs
            .iter()
            .zip(obs_out.chunks_exact_mut(HOVER_OBS_DIM))
        {
            row.copy_from_slice(&env.out.obs);
        }
        Ok(())
    }

    fn check_step_buffers(&self, actions: &[f32], out: &StepBuffers) -> Result<(), String> {
        self.check_action_count(actions)?;
        let n = self.envs.len();
        let checks = [
            ("obs", out.obs.len(), n * HOVER_OBS_DIM),
            ("final_obs", out.final_obs.len(), n * HOVER_OBS_DIM),
            ("reward_terms", out.reward_terms.len(), n * NUM_REWARD_TERMS),
            ("reward", out.reward.len(), n),
            ("terminated", out.terminated.len(), n),
            ("truncated", out.truncated.len(), n),
            ("cause", out.cause.len(), n),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!(
                    "output '{name}' must hold {want} values, got {got}"
                ));
            }
        }
        Ok(())
    }

    /// Step every environment once, on the thread pool, and write what happened into `out`.
    /// Environments whose episode ended have already restarted: their row of `out.obs` is the
    /// first observation of the next episode and `out.final_obs` keeps the last one of the old.
    pub fn step_task(&mut self, actions: &[f32], out: &mut StepBuffers) -> Result<(), String> {
        self.check_step_buffers(actions, out)?;
        let Batch {
            params,
            task,
            envs,
            pool,
        } = self;
        let task = task
            .as_ref()
            .ok_or("this batch has no task: build it with a track_cfg")?;
        let params = &*params;
        pool.install(|| {
            envs.par_iter_mut()
                .zip(actions.par_chunks_exact(4))
                .for_each(|(env, a)| env.step_hover(params, task, action_from_f32(a)));
        });
        copy_step_outputs(&self.envs, out);
        Ok(())
    }

    /// The same as `step_task`, on the calling thread only: the reference for the thread tests.
    pub fn step_task_sequential(
        &mut self,
        actions: &[f32],
        out: &mut StepBuffers,
    ) -> Result<(), String> {
        self.check_step_buffers(actions, out)?;
        let task = self
            .task
            .as_ref()
            .ok_or("this batch has no task: build it with a track_cfg")?;
        let params = &self.params;
        for (env, a) in self.envs.iter_mut().zip(actions.chunks_exact(4)) {
            env.step_hover(params, task, action_from_f32(a));
        }
        copy_step_outputs(&self.envs, out);
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
        let task = self.task.as_ref();
        for (env, state) in self.envs.iter_mut().zip(new_states) {
            env.set_state(state, task);
        }
        Ok(())
    }
}

/// Copy what each environment reported in its latest step into the output arrays.
fn copy_step_outputs(envs: &[Env], out: &mut StepBuffers) {
    for (i, env) in envs.iter().enumerate() {
        let step = &env.out;
        out.obs[i * HOVER_OBS_DIM..(i + 1) * HOVER_OBS_DIM].copy_from_slice(&step.obs);
        out.final_obs[i * HOVER_OBS_DIM..(i + 1) * HOVER_OBS_DIM].copy_from_slice(&step.final_obs);
        out.reward_terms[i * NUM_REWARD_TERMS..(i + 1) * NUM_REWARD_TERMS]
            .copy_from_slice(&step.reward_terms);
        out.reward[i] = step.reward;
        out.terminated[i] = step.terminated;
        out.truncated[i] = step.truncated;
        out.cause[i] = step.cause;
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

    // ----- the hover task -----

    use crate::env::test_hover_config;

    /// Owned output arrays for `n` environments, so a test can call `step_task` and read results.
    struct Outputs {
        obs: Vec<f32>,
        final_obs: Vec<f32>,
        reward_terms: Vec<f32>,
        reward: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
        cause: Vec<u8>,
    }

    impl Outputs {
        fn new(n: usize) -> Outputs {
            Outputs {
                obs: vec![0.0; n * HOVER_OBS_DIM],
                final_obs: vec![0.0; n * HOVER_OBS_DIM],
                reward_terms: vec![0.0; n * NUM_REWARD_TERMS],
                reward: vec![0.0; n],
                terminated: vec![false; n],
                truncated: vec![false; n],
                cause: vec![0; n],
            }
        }

        fn buffers(&mut self) -> StepBuffers<'_> {
            StepBuffers {
                obs: &mut self.obs,
                reward: &mut self.reward,
                terminated: &mut self.terminated,
                truncated: &mut self.truncated,
                final_obs: &mut self.final_obs,
                reward_terms: &mut self.reward_terms,
                cause: &mut self.cause,
            }
        }
    }

    fn hover_batch(n: usize, seed: u64, threads: Option<usize>) -> Batch {
        Batch::with_task(n, test_params(), test_hover_config(), seed, threads).unwrap()
    }

    #[test]
    fn a_batch_with_a_task_reports_its_observation_size_and_one_without_does_not() {
        assert_eq!(hover_batch(2, 0, Some(1)).obs_dim(), Some(HOVER_OBS_DIM));
        assert_eq!(
            Batch::new(2, test_params(), Some(1)).unwrap().obs_dim(),
            None
        );
    }

    #[test]
    fn task_calls_on_a_batch_without_a_task_are_errors() {
        let mut batch = Batch::new(2, test_params(), Some(1)).unwrap();
        let mut out = Outputs::new(2);
        assert!(
            batch
                .reset_task(None, &mut out.obs)
                .unwrap_err()
                .contains("no task")
        );
        assert!(
            batch
                .step_task(&[0.0; 8], &mut out.buffers())
                .unwrap_err()
                .contains("no task")
        );
    }

    #[test]
    fn step_task_rejects_an_output_array_of_the_wrong_size() {
        let mut batch = hover_batch(3, 0, Some(1));
        let mut out = Outputs::new(3);
        out.reward = vec![0.0; 2];
        let err = batch.step_task(&[0.0; 12], &mut out.buffers()).unwrap_err();
        assert!(err.contains("reward") && err.contains("3"), "{err}");
        let mut obs = vec![0.0; 5];
        assert!(
            batch
                .reset_task(None, &mut obs)
                .unwrap_err()
                .contains("obs output")
        );
    }

    #[test]
    fn reset_task_fills_every_environments_first_observation() {
        let mut batch = hover_batch(4, 5, Some(1));
        let mut obs = vec![f32::NAN; 4 * HOVER_OBS_DIM];
        batch.reset_task(None, &mut obs).unwrap();
        assert!(obs.iter().all(|v| v.is_finite()));
        // Different environments start from different random states.
        assert_ne!(
            &obs[0..HOVER_OBS_DIM],
            &obs[HOVER_OBS_DIM..2 * HOVER_OBS_DIM]
        );
        // The previous-action part (last 4 values) of a fresh observation is zero.
        for row in obs.chunks_exact(HOVER_OBS_DIM) {
            assert_eq!(&row[18..22], &[0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn reset_with_the_same_seed_gives_the_same_episodes_and_a_new_seed_gives_new_ones() {
        let mut batch = hover_batch(8, 1, Some(2));
        let mut first = vec![0.0; 8 * HOVER_OBS_DIM];
        let mut again = vec![0.0; 8 * HOVER_OBS_DIM];
        let mut other = vec![0.0; 8 * HOVER_OBS_DIM];
        batch.reset_task(Some(77), &mut first).unwrap();
        batch.reset_task(Some(77), &mut again).unwrap();
        batch.reset_task(Some(78), &mut other).unwrap();
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    #[test]
    fn two_batches_with_the_same_seed_reset_identically() {
        let mut a = hover_batch(6, 9, Some(1));
        let mut b = hover_batch(6, 9, Some(4));
        let (mut obs_a, mut obs_b) = (vec![0.0; 6 * HOVER_OBS_DIM], vec![0.0; 6 * HOVER_OBS_DIM]);
        a.reset_task(None, &mut obs_a).unwrap();
        b.reset_task(None, &mut obs_b).unwrap();
        assert_eq!(obs_a, obs_b);
    }

    #[test]
    fn task_results_including_random_resets_are_identical_for_every_thread_count() {
        // A run long enough that many episodes end (crash or fly away) and restart at random, so
        // the random resets are exercised too. Compare every output bit for bit, step by step.
        let n = 64;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let mut sequential = hover_batch(n, 3, Some(1));
        let mut threaded: Vec<Batch> = [Some(1), Some(2), Some(8), None]
            .into_iter()
            .map(|t| hover_batch(n, 3, t))
            .collect();
        let mut obs = vec![0.0; n * HOVER_OBS_DIM];
        sequential.reset_task(None, &mut obs).unwrap();
        for b in threaded.iter_mut() {
            b.reset_task(None, &mut obs).unwrap();
        }

        let mut episodes_ended = 0;
        for step in 0..300 {
            let actions = wild_actions(step, n); // wild actions: episodes end often
            let mut reference = Outputs::new(n);
            sequential
                .step_task_sequential(&actions, &mut reference.buffers())
                .unwrap();
            episodes_ended += reference.terminated.iter().filter(|&&t| t).count();
            for batch in threaded.iter_mut() {
                let mut got = Outputs::new(n);
                batch.step_task(&actions, &mut got.buffers()).unwrap();
                let threads = batch.num_threads();
                assert_eq!(
                    bits(&got.obs),
                    bits(&reference.obs),
                    "obs, {threads} threads, step {step}"
                );
                assert_eq!(
                    bits(&got.final_obs),
                    bits(&reference.final_obs),
                    "final_obs, {threads} threads"
                );
                assert_eq!(
                    bits(&got.reward),
                    bits(&reference.reward),
                    "reward, {threads} threads"
                );
                assert_eq!(
                    bits(&got.reward_terms),
                    bits(&reference.reward_terms),
                    "terms, {threads} threads"
                );
                assert_eq!(got.terminated, reference.terminated);
                assert_eq!(got.truncated, reference.truncated);
                assert_eq!(got.cause, reference.cause);
            }
        }
        assert!(
            episodes_ended > 20,
            "only {episodes_ended} episodes ended: the test is not exercising resets"
        );
        // The final physics states agree as well.
        let final_bits = |b: &Batch| state_bits(b);
        for batch in &threaded {
            assert_eq!(
                final_bits(batch),
                final_bits(&sequential),
                "{} threads",
                batch.num_threads()
            );
        }
    }

    #[test]
    fn set_state_starts_a_fresh_episode_for_the_task() {
        let mut batch = hover_batch(2, 1, Some(1));
        let mut flat = vec![0.0; 2 * STATE_SIZE];
        batch.write_states(&mut flat).unwrap();
        // Put both drones level at the target with hover thrust, then step with the hover action:
        // the reward is the action-change penalty only (previous action was zeros).
        let target = test_hover_config().target_w;
        for row in flat.chunks_exact_mut(STATE_SIZE) {
            row[0..3].copy_from_slice(&[target.x, target.y, target.z]);
            row[3..7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            row[7..13].fill(0.0);
            row[13..17].fill(test_params().mass * 9.81 / 4.0);
        }
        batch.set_states(&flat).unwrap();
        let hover = 2.0 * test_params().mass * 9.81 / (4.0 * 8.5) - 1.0;
        let mut out = Outputs::new(2);
        batch
            .step_task(
                &[hover as f32, 0.0, 0.0, 0.0, hover as f32, 0.0, 0.0, 0.0],
                &mut out.buffers(),
            )
            .unwrap();
        assert!(out.reward_terms[0].abs() < 1e-6 && out.reward_terms[1].abs() < 1e-6);
        assert!(out.reward_terms[3] < 0.0);
        assert!(!out.terminated[0] && !out.truncated[0]);
    }

    /// Actions that differ per environment and per step, wide enough to make drones crash.
    fn wild_actions(step: usize, n: usize) -> Vec<f32> {
        (0..n * 4)
            .map(|k| ((step * 11 + k * 5) as f32 * 0.53).sin() * 1.5)
            .collect()
    }
}
