//! One environment. For now that is just one drone's physics state.
//!
//! Observations, rewards, gates and resets arrive in later milestones (handoff section 8), and
//! will be added to this struct and file. Keeping the per-environment data in one struct is what
//! lets `batch.rs` step many of them side by side.

use crate::dynamics::{State, policy_step};
use crate::params::DroneParams;

/// One simulated drone.
#[derive(Clone, Debug)]
pub struct Env {
    pub state: State,
}

impl Env {
    pub fn new(state: State) -> Env {
        Env { state }
    }

    /// Advance this drone by one 50 Hz policy step with the given action in `[-1, 1]^4`.
    pub fn step_physics(&mut self, params: &DroneParams, action: [f64; 4]) {
        self.state = policy_step(params, &self.state, action);
    }
}
