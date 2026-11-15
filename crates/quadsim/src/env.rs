//! One environment: one drone's physics state, plus the bookkeeping of the task it is flying.
//!
//! The task so far is **hover** (handoff section 8, "Hover task"): fly to a target point and stay
//! there. This file holds its observation, reward, episode ends and random resets. The gate-racing
//! task (milestone 4) will be added here too. Keeping everything one environment needs in one
//! struct is what lets `batch.rs` step many of them side by side.
//!
//! Each environment owns its own seeded random generator, and nothing else is shared, so the
//! result of a run does not depend on how many threads step the batch.
//!
//! Nothing here allocates on the heap while stepping: the outputs of the latest step are stored in
//! fixed-size arrays inside `Env`, and `batch.rs` copies them out.

use std::f64::consts::TAU;

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;

use crate::dynamics::{State, policy_step};
use crate::math::{Quat, Vec3};
use crate::params::DroneParams;

/// Number of values in the hover observation (handoff section 8).
pub const HOVER_OBS_DIM: usize = 22;
/// Number of reward terms reported per step: progress, distance penalty, rate penalty, action
/// penalty, crash.
pub const NUM_REWARD_TERMS: usize = 5;
/// Names of those terms, in order.
pub const HOVER_REWARD_TERM_NAMES: [&str; NUM_REWARD_TERMS] = [
    "progress",
    "distance_penalty",
    "rate_penalty",
    "action_penalty",
    "crash",
];

// Termination-cause codes (handoff section 8). Hover uses only these three.
pub const CAUSE_NONE: u8 = 0;
pub const CAUSE_ALTITUDE: u8 = 3; // altitude below the minimum: the drone hit the ground
pub const CAUSE_OUT_OF_BOUNDS: u8 = 4; // too far from the target
pub const CAUSE_TIME_LIMIT: u8 = 6; // the time limit ran out

/// The hover task's settings, read from `configs/task_hover.yaml`.
#[derive(Clone, Debug)]
pub struct HoverConfig {
    pub target_w: Vec3,          // m, world frame
    pub max_seconds: f64,        // episode time limit
    pub min_altitude: f64,       // m: below this the episode terminates
    pub max_distance: f64,       // m from the target: beyond this the episode terminates
    pub start_max_offset: f64,   // m: start within this distance of the target
    pub start_min_altitude: f64, // m: and at least this high
    pub start_max_tilt: f64,     // rad
    pub start_max_speed: f64,    // m/s
    pub start_max_rate: f64,     // rad/s
    pub weight_progress: f64,    // reward per metre closer to the target
    pub weight_distance: f64,    // penalty per metre of distance
    pub weight_body_rate: f64,   // penalty per (rad/s)^2
    pub weight_action_change: f64,
    pub crash_penalty: f64,
}

impl HoverConfig {
    /// Episode length in policy steps (15 s at 50 Hz is 750).
    pub fn max_steps(&self, params: &DroneParams) -> u32 {
        (self.max_seconds * params.policy_hz).round() as u32
    }

    /// Check the values make sense, so a bad config fails with a message instead of misbehaving.
    pub fn validate(&self, params: &DroneParams) -> Result<(), String> {
        if !self.target_w.is_finite() {
            return Err("track_cfg: target must be finite".to_string());
        }
        let non_negative = |name: &str, value: f64| {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "track_cfg: {name} must be a number >= 0, got {value}"
                ))
            }
        };
        non_negative("reset.max_offset", self.start_max_offset)?;
        non_negative("reset.min_altitude", self.start_min_altitude)?;
        non_negative("reset.max_tilt", self.start_max_tilt)?;
        non_negative("reset.max_speed", self.start_max_speed)?;
        non_negative("reset.max_body_rate", self.start_max_rate)?;
        non_negative("termination.min_altitude", self.min_altitude)?;
        for (name, value) in [
            ("reward.progress", self.weight_progress),
            ("reward.distance", self.weight_distance),
            ("reward.body_rate", self.weight_body_rate),
            ("reward.action_change", self.weight_action_change),
            ("reward.crash", self.crash_penalty),
        ] {
            if !value.is_finite() {
                return Err(format!(
                    "track_cfg: {name} must be a finite number, got {value}"
                ));
            }
        }
        if !(self.max_distance.is_finite() && self.max_distance > 0.0) {
            return Err("track_cfg: termination.max_distance must be greater than 0".to_string());
        }
        if !(self.max_seconds.is_finite() && self.max_steps(params) >= 1) {
            return Err(
                "track_cfg: episode.max_seconds must allow at least one policy step".to_string(),
            );
        }
        if self.start_max_tilt >= std::f64::consts::FRAC_PI_2 {
            return Err("track_cfg: reset.max_tilt must be less than pi/2 (a drone that starts on its side cannot recover)".to_string());
        }
        Ok(())
    }
}

/// What one environment reports after a step. `batch.rs` copies these into the output arrays.
#[derive(Clone, Copy, Debug)]
pub struct StepOut {
    pub reward: f32,
    pub reward_terms: [f32; NUM_REWARD_TERMS],
    pub terminated: bool,
    pub truncated: bool,
    pub cause: u8,
    /// What the agent sees next. If the episode just ended this is already the first
    /// observation of the next episode.
    pub obs: [f32; HOVER_OBS_DIM],
    /// The last observation of the episode that just ended (equal to `obs` if none ended).
    pub final_obs: [f32; HOVER_OBS_DIM],
}

impl StepOut {
    fn zeroed() -> StepOut {
        StepOut {
            reward: 0.0,
            reward_terms: [0.0; NUM_REWARD_TERMS],
            terminated: false,
            truncated: false,
            cause: CAUSE_NONE,
            obs: [0.0; HOVER_OBS_DIM],
            final_obs: [0.0; HOVER_OBS_DIM],
        }
    }
}

/// One simulated drone and its task bookkeeping.
#[derive(Clone, Debug)]
pub struct Env {
    pub state: State,
    rng: Pcg64Mcg,
    prev_action: [f64; 4], // the action applied on the previous step (zeros at a reset)
    steps: u32,            // policy steps taken in this episode
    prev_distance: f64,    // distance to the target before the latest step
    pub out: StepOut,
}

/// A seed for environment number `index` of a batch seeded with `seed`.
///
/// Every environment gets its own generator, seeded from the batch seed and its own index, so the
/// random numbers an environment sees never depend on which thread runs it or on the other
/// environments. (`seed_from_u64` scrambles its input, so neighbouring seeds give unrelated streams.)
pub fn env_seed(seed: u64, index: usize) -> u64 {
    seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

impl Env {
    /// A drone at `state`, with its random generator seeded by `seed`.
    pub fn new(state: State, seed: u64) -> Env {
        Env {
            state,
            rng: Pcg64Mcg::seed_from_u64(seed),
            prev_action: [0.0; 4],
            steps: 0,
            prev_distance: 0.0,
            out: StepOut::zeroed(),
        }
    }

    /// Restart this environment's random number stream from `seed`.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Pcg64Mcg::seed_from_u64(seed);
    }

    /// Advance this drone by one 50 Hz policy step with the given action in `[-1, 1]^4`,
    /// physics only: no reward, no episode bookkeeping.
    pub fn step_physics(&mut self, params: &DroneParams, action: [f64; 4]) {
        self.state = policy_step(params, &self.state, action);
    }

    /// Replace the state and start a fresh episode bookkeeping from it (used by `set_state`).
    pub fn set_state(&mut self, state: State, task: Option<&HoverConfig>) {
        self.state = state;
        self.begin_episode(task);
    }

    /// Zero the episode counters for the current state.
    fn begin_episode(&mut self, task: Option<&HoverConfig>) {
        self.steps = 0;
        self.prev_action = [0.0; 4];
        self.prev_distance = match task {
            Some(task) => (task.target_w - self.state.pos_w).norm(),
            None => 0.0,
        };
    }

    /// Start a new hover episode from a random state (handoff section 8, "Resets"), and write its
    /// first observation into `out.obs`.
    pub fn reset_hover(&mut self, params: &DroneParams, task: &HoverConfig) {
        // Position: uniformly within `start_max_offset` of the target and at least
        // `start_min_altitude` above the ground. Try a few times, then fall back to clamping.
        let mut pos_w = task.target_w;
        for attempt in 0..100 {
            pos_w = task.target_w + uniform_in_ball(&mut self.rng, task.start_max_offset);
            if pos_w.z >= task.start_min_altitude {
                break;
            }
            if attempt == 99 {
                pos_w.z = task.start_min_altitude;
            }
        }

        // Attitude: tilted by an angle up to `start_max_tilt` about a random horizontal axis, yaw 0.
        let tilt = self.rng.random_range(0.0..=task.start_max_tilt);
        let azimuth = self.rng.random_range(0.0..TAU);
        let (sin_half, cos_half) = (0.5 * tilt).sin_cos();
        let quat = Quat::new(
            cos_half,
            sin_half * azimuth.cos(),
            sin_half * azimuth.sin(),
            0.0,
        );

        // Velocity (world frame) and body rates: a random direction times a random size up to the maximum.
        let vel_w =
            random_unit_vector(&mut self.rng) * self.rng.random_range(0.0..=task.start_max_speed);
        let omega_b =
            random_unit_vector(&mut self.rng) * self.rng.random_range(0.0..=task.start_max_rate);

        // Motors start at hover thrust: each carries a quarter of the weight.
        let thrust = [params.mass * params.gravity / 4.0; 4];

        self.state = State {
            pos_w,
            quat,
            vel_w,
            omega_b,
            thrust,
        };
        self.begin_episode(Some(task));
        self.out.obs = hover_observation(&self.state, task.target_w, &self.prev_action);
    }

    /// Advance one 50 Hz policy step of the hover task and record everything in `out`.
    ///
    /// Order: physics, then reward (using the distance before and after), then the episode-end
    /// checks, then the observation. If the episode ended, the environment resets itself: `out.obs`
    /// becomes the first observation of the next episode and `out.final_obs` keeps the last one of
    /// the episode that just ended.
    pub fn step_hover(&mut self, params: &DroneParams, task: &HoverConfig, action: [f64; 4]) {
        // The action the drone actually flies on: out-of-range values are clipped (handoff is
        // silent on this; `action_to_commands` clips the same way).
        let action = action.map(|v| v.clamp(-1.0, 1.0));
        self.state = policy_step(params, &self.state, action);
        self.steps += 1;

        let distance = (task.target_w - self.state.pos_w).norm();
        let omega_b = self.state.omega_b;
        let mut action_change_sq = 0.0;
        for i in 0..4 {
            let change = action[i] - self.prev_action[i];
            action_change_sq += change * change;
        }

        // Episode ends. Termination: the task is over (hit the ground, flew away). Truncation:
        // the time limit ran out. A step that terminates is not also reported as truncated.
        let hit_ground = self.state.pos_w.z < task.min_altitude;
        let flew_away = distance > task.max_distance;
        let terminated = hit_ground || flew_away;
        let truncated = !terminated && self.steps >= task.max_steps(params);
        let cause = if hit_ground {
            CAUSE_ALTITUDE
        } else if flew_away {
            CAUSE_OUT_OF_BOUNDS
        } else if truncated {
            CAUSE_TIME_LIMIT
        } else {
            CAUSE_NONE
        };

        // Reward terms, already weighted, so that they add up to the step's reward.
        let terms = [
            task.weight_progress * (self.prev_distance - distance), // metres gained toward the target
            -task.weight_distance * distance,
            -task.weight_body_rate * omega_b.dot(omega_b),
            -task.weight_action_change * action_change_sq,
            if terminated { -task.crash_penalty } else { 0.0 },
        ];
        let reward: f64 = terms.iter().sum();

        self.prev_action = action;
        self.prev_distance = distance;

        let obs = hover_observation(&self.state, task.target_w, &self.prev_action);
        self.out.reward = reward as f32;
        self.out.reward_terms = terms.map(|t| t as f32);
        self.out.terminated = terminated;
        self.out.truncated = truncated;
        self.out.cause = cause;
        self.out.obs = obs;
        self.out.final_obs = obs;

        if terminated || truncated {
            self.reset_hover(params, task); // overwrites out.obs with the next episode's first observation
        }
    }
}

/// A random point uniformly inside a ball of the given radius (rejection sampling in a cube).
fn uniform_in_ball(rng: &mut Pcg64Mcg, radius: f64) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        if v.dot(v) <= 1.0 {
            return v * radius;
        }
    }
}

/// A random direction, uniform over the sphere.
fn random_unit_vector(rng: &mut Pcg64Mcg) -> Vec3 {
    loop {
        let v = uniform_in_ball(rng, 1.0);
        let norm = v.norm();
        if norm > 1e-6 {
            return v / norm;
        }
    }
}

/// The hover observation, 22 values, everything relative to the drone (handoff section 8):
///
/// | part | size |
/// | --- | --- |
/// | velocity in the body frame | 3 |
/// | rotation matrix, body to world, row-major | 9 |
/// | body rates | 3 |
/// | target position relative to the drone, in the body frame | 3 |
/// | previous action | 4 |
pub fn hover_observation(
    state: &State,
    target_w: Vec3,
    prev_action: &[f64; 4],
) -> [f32; HOVER_OBS_DIM] {
    let rot = state.quat.to_rotation_matrix(); // body to world
    let vel_b = rot.transpose_mul_vec(state.vel_w); // R^T v_w
    let target_b = rot.transpose_mul_vec(target_w - state.pos_w); // R^T (target - position)

    let mut obs = [0.0_f32; HOVER_OBS_DIM];
    obs[0] = vel_b.x as f32;
    obs[1] = vel_b.y as f32;
    obs[2] = vel_b.z as f32;
    for row in 0..3 {
        for col in 0..3 {
            obs[3 + 3 * row + col] = rot.rows[row][col] as f32;
        }
    }
    obs[12] = state.omega_b.x as f32;
    obs[13] = state.omega_b.y as f32;
    obs[14] = state.omega_b.z as f32;
    obs[15] = target_b.x as f32;
    obs[16] = target_b.y as f32;
    obs[17] = target_b.z as f32;
    for i in 0..4 {
        obs[18 + i] = prev_action[i] as f32;
    }
    obs
}

/// A hover config for Rust unit tests. Same numbers as `configs/task_hover.yaml` at the time of
/// writing; nothing checks they stay equal, and real runs always read the YAML.
#[cfg(test)]
pub fn test_hover_config() -> HoverConfig {
    HoverConfig {
        target_w: Vec3::new(0.0, 0.0, 2.0),
        max_seconds: 15.0,
        min_altitude: 0.1,
        max_distance: 5.0,
        start_max_offset: 1.5,
        start_min_altitude: 0.5,
        start_max_tilt: 0.5,
        start_max_speed: 2.0,
        start_max_rate: 2.0,
        weight_progress: 1.0,
        weight_distance: 0.05,
        weight_body_rate: 0.0002,
        weight_action_change: 0.002,
        crash_penalty: 5.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::test_params;

    fn hover_action(params: &DroneParams) -> [f64; 4] {
        [
            2.0 * params.mass * params.gravity / (4.0 * params.thrust_max) - 1.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    fn hovering_at_target(params: &DroneParams, task: &HoverConfig, seed: u64) -> Env {
        let mut env = Env::new(
            State::level_at(task.target_w, params.mass * params.gravity / 4.0),
            seed,
        );
        env.set_state(env.state, Some(task));
        env
    }

    // ----- observation -----

    #[test]
    fn observation_of_a_level_drone_at_rest_below_the_target() {
        let task = test_hover_config();
        let state = State::level_at(Vec3::new(0.0, 0.0, 1.0), 1.0); // 1 m below the target
        let obs = hover_observation(&state, task.target_w, &[0.0; 4]);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(&obs[0..3], &[0.0, 0.0, 0.0]); // velocity
        assert_eq!(&obs[3..12], &identity); // rotation matrix, row-major
        assert_eq!(&obs[12..15], &[0.0, 0.0, 0.0]); // body rates
        assert_eq!(&obs[15..18], &[0.0, 0.0, 1.0]); // the target is 1 m up, in the body frame too
        assert_eq!(&obs[18..22], &[0.0, 0.0, 0.0, 0.0]); // previous action
    }

    #[test]
    fn observation_target_and_velocity_are_in_the_body_frame() {
        // Yawed 90 degrees: body x points along world y. A target one metre along world x is then
        // one metre to the drone's right, which is body -y. Likewise for velocity.
        let task = test_hover_config();
        let half = 0.5_f64.sqrt();
        let mut state = State::level_at(Vec3::new(-1.0, 0.0, 2.0), 1.0);
        state.quat = Quat::new(half, 0.0, 0.0, half);
        state.vel_w = Vec3::new(3.0, 0.0, 0.0);
        let obs = hover_observation(&state, task.target_w, &[0.0; 4]);
        let close = |a: f32, b: f64| (f64::from(a) - b).abs() < 1e-6;
        assert!(
            close(obs[0], 0.0) && close(obs[1], -3.0) && close(obs[2], 0.0),
            "velocity {:?}",
            &obs[0..3]
        );
        assert!(
            close(obs[15], 0.0) && close(obs[16], -1.0) && close(obs[17], 0.0),
            "target {:?}",
            &obs[15..18]
        );
    }

    #[test]
    fn observation_carries_the_rates_and_the_previous_action() {
        let task = test_hover_config();
        let mut state = State::level_at(task.target_w, 1.0);
        state.omega_b = Vec3::new(0.5, -0.25, 2.0);
        let obs = hover_observation(&state, task.target_w, &[0.1, -0.2, 0.3, -0.4]);
        assert_eq!(&obs[12..15], &[0.5, -0.25, 2.0]);
        assert_eq!(&obs[18..22], &[0.1_f32, -0.2, 0.3, -0.4]);
    }

    // ----- resets -----

    #[test]
    fn resets_follow_the_ranges_in_the_config() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), 42);
        for _ in 0..2000 {
            env.reset_hover(&params, &task);
            let s = env.state;
            assert!((s.pos_w - task.target_w).norm() <= task.start_max_offset + 1e-12);
            assert!(s.pos_w.z >= task.start_min_altitude);
            let tilt = s.quat.to_rotation_matrix().rows[2][2]
                .clamp(-1.0, 1.0)
                .acos();
            assert!(tilt <= task.start_max_tilt + 1e-9, "tilt {tilt}");
            assert_eq!(s.quat.z, 0.0, "yaw starts at 0");
            assert!(s.vel_w.norm() <= task.start_max_speed + 1e-12);
            assert!(s.omega_b.norm() <= task.start_max_rate + 1e-12);
            assert_eq!(s.thrust, [params.mass * params.gravity / 4.0; 4]);
            assert!(s.validate().is_ok());
        }
    }

    #[test]
    fn resets_actually_vary() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), 1);
        env.reset_hover(&params, &task);
        let first = env.state;
        env.reset_hover(&params, &task);
        assert_ne!(first.pos_w, env.state.pos_w);
        assert_ne!(first.vel_w, env.state.vel_w);
    }

    #[test]
    fn the_same_seed_gives_the_same_resets_and_another_seed_does_not() {
        let (params, task) = (test_params(), test_hover_config());
        let start = |seed: u64| {
            let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), seed);
            env.reset_hover(&params, &task);
            env.state
        };
        assert_eq!(start(7), start(7));
        assert_ne!(start(7), start(8));
    }

    #[test]
    fn reseeding_restarts_the_random_stream() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), 5);
        env.reset_hover(&params, &task);
        let first = env.state;
        env.reset_hover(&params, &task);
        env.reseed(5);
        env.reset_hover(&params, &task);
        assert_eq!(first, env.state);
    }

    #[test]
    fn the_first_observation_after_a_reset_matches_the_state_and_a_zero_previous_action() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), 9);
        env.reset_hover(&params, &task);
        assert_eq!(
            env.out.obs,
            hover_observation(&env.state, task.target_w, &[0.0; 4])
        );
    }

    // ----- stepping: reward -----

    #[test]
    fn hovering_exactly_at_the_target_earns_zero_reward() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = hovering_at_target(&params, &task, 1);
        env.step_hover(&params, &task, hover_action(&params));
        // The drone holds still at the target: no progress, no distance, no body rate. Only the
        // action-change term is non-zero on this first step, because the previous action was
        // zeros and the hover action's thrust value is not zero.
        let change_sq = hover_action(&params)[0] * hover_action(&params)[0];
        assert!(env.out.reward_terms[0].abs() < 1e-9);
        assert!(env.out.reward_terms[1].abs() < 1e-9);
        assert!(env.out.reward_terms[2].abs() < 1e-9);
        assert!(
            (f64::from(env.out.reward_terms[3]) + task.weight_action_change * change_sq).abs()
                < 1e-7
        );
        assert_eq!(env.out.reward_terms[4], 0.0);
        // The second step repeats the same action: now nothing is paid at all.
        env.step_hover(&params, &task, hover_action(&params));
        assert!(env.out.reward.abs() < 1e-9, "reward {}", env.out.reward);
    }

    #[test]
    fn flying_toward_the_target_pays_progress_and_the_terms_add_up_to_the_reward() {
        let (params, task) = (test_params(), test_hover_config());
        // Start 1 m below the target, moving up at 2 m/s with hover thrust: the distance shrinks.
        let mut state =
            State::level_at(Vec3::new(0.0, 0.0, 1.0), params.mass * params.gravity / 4.0);
        state.vel_w = Vec3::new(0.0, 0.0, 2.0);
        let mut env = Env::new(state, 1);
        env.set_state(state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        let terms = env.out.reward_terms;
        assert!(terms[0] > 0.0, "progress should be positive: {}", terms[0]);
        let moved = 1.0 - (task.target_w - env.state.pos_w).norm();
        assert!((f64::from(terms[0]) - moved).abs() < 1e-6);
        assert!(terms[1] < 0.0 && terms[2] <= 0.0 && terms[3] <= 0.0);
        let sum: f32 = terms.iter().sum();
        assert!((sum - env.out.reward).abs() < 1e-6);
    }

    #[test]
    fn the_distance_penalty_is_proportional_to_the_distance() {
        let (params, task) = (test_params(), test_hover_config());
        // Hovering (zero velocity) 2 m from the target, so the distance barely changes in one step.
        let mut env = Env::new(
            State::level_at(Vec3::new(2.0, 0.0, 2.0), params.mass * params.gravity / 4.0),
            1,
        );
        env.set_state(env.state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        assert!((f64::from(env.out.reward_terms[1]) + 0.05 * 2.0).abs() < 1e-4);
    }

    #[test]
    fn the_rate_penalty_is_the_weight_times_the_squared_body_rate() {
        let (params, task) = (test_params(), test_hover_config());
        let mut state = State::level_at(task.target_w, params.mass * params.gravity / 4.0);
        state.omega_b = Vec3::new(3.0, 0.0, 0.0); // the rate controller will start to reduce it
        let mut env = Env::new(state, 1);
        env.set_state(state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        let rate = env.state.omega_b;
        let expected = -task.weight_body_rate * rate.dot(rate);
        assert!((f64::from(env.out.reward_terms[2]) - expected).abs() < 1e-7);
        assert!(env.out.reward_terms[2] < 0.0);
    }

    #[test]
    fn the_action_change_penalty_uses_the_clipped_actions() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = hovering_at_target(&params, &task, 1);
        env.step_hover(&params, &task, [50.0, -50.0, 0.0, 0.0]); // clipped to (1, -1, 0, 0)
        let expected = -task.weight_action_change * (1.0 + 1.0);
        assert!((f64::from(env.out.reward_terms[3]) - expected).abs() < 1e-7);
    }

    // ----- stepping: episode ends -----

    #[test]
    fn falling_to_the_ground_terminates_with_the_altitude_cause_and_the_crash_penalty_once() {
        let (params, task) = (test_params(), test_hover_config());
        // Motors off, 0.3 m up: it reaches the 0.1 m limit within a few steps.
        let mut env = Env::new(State::level_at(Vec3::new(0.0, 0.0, 0.3), 0.0), 1);
        env.set_state(env.state, Some(&task));
        let mut crash_penalties = 0;
        let mut ended_at = None;
        for step in 1..=30 {
            env.step_hover(&params, &task, [-1.0, 0.0, 0.0, 0.0]);
            if env.out.reward_terms[4] != 0.0 {
                crash_penalties += 1;
                assert_eq!(env.out.reward_terms[4] as f64, -task.crash_penalty);
            }
            if env.out.terminated {
                ended_at = Some(step);
                break;
            }
        }
        assert!(ended_at.is_some(), "the drone should have hit the ground");
        assert_eq!(crash_penalties, 1);
        assert_eq!(env.out.cause, CAUSE_ALTITUDE);
        assert!(
            !env.out.truncated,
            "a terminated step is not also truncated"
        );
    }

    #[test]
    fn flying_too_far_from_the_target_terminates_with_the_out_of_bounds_cause() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(
            State::level_at(Vec3::new(5.5, 0.0, 2.0), params.mass * params.gravity / 4.0),
            1,
        );
        env.set_state(env.state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        assert!(env.out.terminated);
        assert_eq!(env.out.cause, CAUSE_OUT_OF_BOUNDS);
    }

    #[test]
    fn the_episode_is_truncated_at_exactly_750_steps_and_not_before() {
        let (params, task) = (test_params(), test_hover_config());
        assert_eq!(task.max_steps(&params), 750);
        let mut env = hovering_at_target(&params, &task, 1);
        for step in 1..=750 {
            env.step_hover(&params, &task, hover_action(&params));
            assert!(!env.out.terminated, "step {step}");
            assert_eq!(env.out.truncated, step == 750, "step {step}");
        }
        assert_eq!(env.out.cause, CAUSE_TIME_LIMIT);
    }

    #[test]
    fn a_step_that_does_not_end_the_episode_reports_no_cause() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = hovering_at_target(&params, &task, 1);
        env.step_hover(&params, &task, hover_action(&params));
        assert_eq!(
            (env.out.terminated, env.out.truncated, env.out.cause),
            (false, false, CAUSE_NONE)
        );
    }

    // ----- automatic reset -----

    #[test]
    fn when_an_episode_ends_the_environment_resets_and_final_obs_keeps_the_last_observation() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(
            State::level_at(Vec3::new(5.5, 0.0, 2.0), params.mass * params.gravity / 4.0),
            3,
        );
        env.set_state(env.state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        assert!(env.out.terminated);
        // obs is the first observation of a fresh episode: previous action zero, a start state
        // within 1.5 m of the target.
        assert_eq!(&env.out.obs[18..22], &[0.0, 0.0, 0.0, 0.0]);
        assert!((env.state.pos_w - task.target_w).norm() <= task.start_max_offset + 1e-12);
        // final_obs is the last observation of the ended episode: far from the target, with the
        // action that was just taken as the previous action.
        let target_distance = |obs: &[f32; HOVER_OBS_DIM]| {
            (f64::from(obs[15]).powi(2) + f64::from(obs[16]).powi(2) + f64::from(obs[17]).powi(2))
                .sqrt()
        };
        assert!(target_distance(&env.out.final_obs) > task.max_distance);
        assert!(target_distance(&env.out.obs) <= task.start_max_offset + 1e-6);
        assert_ne!(env.out.final_obs[18], 0.0);
    }

    #[test]
    fn after_a_reset_the_episode_counters_start_again() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = hovering_at_target(&params, &task, 1);
        for _ in 0..750 {
            env.step_hover(&params, &task, hover_action(&params));
        }
        assert!(env.out.truncated);
        // The truncating step reset the environment: a fresh step count, a zero previous action,
        // and the distance of the NEW start state (so the first step's progress is measured from it).
        assert_eq!(env.steps, 0);
        assert_eq!(env.prev_action, [0.0; 4]);
        assert_eq!(env.prev_distance, (task.target_w - env.state.pos_w).norm());
    }

    #[test]
    fn the_crash_penalty_is_not_paid_when_the_episode_is_only_cut_off_by_the_time_limit() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = hovering_at_target(&params, &task, 1);
        for _ in 0..750 {
            env.step_hover(&params, &task, hover_action(&params));
        }
        assert!(env.out.truncated);
        assert_eq!(env.out.reward_terms[4], 0.0);
    }

    #[test]
    fn a_step_that_terminates_exactly_at_the_time_limit_is_terminated_and_not_truncated() {
        let (params, task) = (test_params(), test_hover_config());
        let mut env = Env::new(
            State::level_at(Vec3::new(5.5, 0.0, 2.0), params.mass * params.gravity / 4.0),
            1,
        );
        env.set_state(env.state, Some(&task));
        env.steps = 749; // the next step is the 750th: the time limit and the flight-away coincide
        env.step_hover(&params, &task, hover_action(&params));
        assert!(env.out.terminated);
        assert!(!env.out.truncated);
        assert_eq!(env.out.cause, CAUSE_OUT_OF_BOUNDS);
    }

    #[test]
    fn progress_is_measured_from_the_previous_step_not_from_the_start_of_the_episode() {
        let (params, task) = (test_params(), test_hover_config());
        let mut state =
            State::level_at(Vec3::new(0.0, 0.0, 1.0), params.mass * params.gravity / 4.0);
        state.vel_w = Vec3::new(0.0, 0.0, 1.0); // coasting up toward the target
        let mut env = Env::new(state, 1);
        env.set_state(state, Some(&task));
        env.step_hover(&params, &task, hover_action(&params));
        let distance_after_first = (task.target_w - env.state.pos_w).norm();
        env.step_hover(&params, &task, hover_action(&params));
        let distance_after_second = (task.target_w - env.state.pos_w).norm();
        let expected = distance_after_first - distance_after_second;
        assert!((f64::from(env.out.reward_terms[0]) - expected).abs() < 1e-6);
    }

    #[test]
    fn the_observation_rotation_matrix_is_row_major_and_maps_body_to_world() {
        // Yawed 90 degrees, body x points along world y: R = [[0,-1,0],[1,0,0],[0,0,1]].
        let half = 0.5_f64.sqrt();
        let mut state = State::level_at(Vec3::new(0.0, 0.0, 2.0), 1.0);
        state.quat = Quat::new(half, 0.0, 0.0, half);
        let obs = hover_observation(&state, Vec3::new(0.0, 0.0, 2.0), &[0.0; 4]);
        let expected = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (f64::from(obs[3 + i]) - want).abs() < 1e-6,
                "entry {i}: {}",
                obs[3 + i]
            );
        }
    }

    #[test]
    fn resets_respect_the_minimum_altitude_even_when_the_start_ball_reaches_below_it() {
        // A target 1 m up with a 1.5 m ball reaches down to -0.5 m, so the 0.5 m floor must bite.
        let params = test_params();
        let mut task = test_hover_config();
        task.target_w = Vec3::new(0.0, 0.0, 1.0);
        let mut env = Env::new(State::level_at(Vec3::ZERO, 1.0), 4);
        let mut lowest: f64 = f64::MAX;
        for _ in 0..3000 {
            env.reset_hover(&params, &task);
            assert!(env.state.pos_w.z >= task.start_min_altitude);
            assert!((env.state.pos_w - task.target_w).norm() <= task.start_max_offset + 1e-12);
            lowest = lowest.min(env.state.pos_w.z);
        }
        assert!(
            lowest < 0.6,
            "starts should reach down close to the 0.5 m floor, lowest was {lowest}"
        );
    }

    #[test]
    fn a_non_finite_or_bad_config_value_is_rejected() {
        let params = test_params();
        let mut task = test_hover_config();
        assert!(task.validate(&params).is_ok());
        task.start_max_tilt = 2.0;
        assert!(task.validate(&params).unwrap_err().contains("max_tilt"));
        let mut task = test_hover_config();
        task.max_seconds = 0.0;
        assert!(task.validate(&params).unwrap_err().contains("max_seconds"));
        let mut task = test_hover_config();
        task.weight_progress = f64::NAN;
        assert!(task.validate(&params).is_err());
    }
}
