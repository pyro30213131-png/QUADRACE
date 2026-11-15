//! quadsim: the Rust half of quadrace. A batched quadrotor physics simulator, exposed to
//! Python through PyO3.
//!
//! How the pieces fit:
//!   * `math` is a hand-written `Vec3`, `Quat` and `Mat3`.
//!   * `params` holds every physical constant, filled from `configs/drone.yaml`.
//!   * `motors`, `controller` and `dynamics` are the physics, ported line by line from the
//!     NumPy reference (`python/quadrace/ref_sim.py`), which is the oracle for parity tests.
//!   * `env` is one drone; `batch` steps many of them.
//!   * The `_quadsim` module at the bottom is the thin PyO3 layer. It turns NumPy arrays into
//!     Rust slices, calls the plain Rust code, and turns a Rust error into a Python exception.
//!     Nothing above it knows about Python, so `cargo test` and the benchmark run without it.
//!
//! Observations, rewards and gates (handoff sections 5 and 8) arrive in later milestones.

pub mod batch;
pub mod controller;
pub mod dynamics;
pub mod env;
pub mod math;
pub mod motors;
pub mod params;

use numpy::ndarray::{Array1, ArrayView1};

/// Element-wise sum of two vectors. A milestone 0 toolchain check, kept as a tiny example.
///
/// Returns an error message (not a panic) if the lengths differ, so the Python
/// caller gets a normal `ValueError` instead of a crashed interpreter.
fn add_views(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "add_arrays: length mismatch, a has {} elements and b has {}",
            a.len(),
            b.len()
        ));
    }
    // `&a + &b` builds a new array and leaves a and b untouched.
    Ok(&a + &b)
}

/// The Python module. Its name must match `[lib] name` in Cargo.toml and the
/// last part of `module-name` in pyproject.toml. Every `#[pyfunction]` and `#[pyclass]`
/// inside this block is added to the module automatically.
#[pyo3::pymodule]
mod _quadsim {
    use crate::batch::{Batch, StepBuffers};
    use crate::dynamics::STATE_SIZE;
    use crate::env::{HOVER_REWARD_TERM_NAMES, HoverConfig, NUM_REWARD_TERMS};
    use crate::math::Vec3;
    use crate::params::DroneParams;
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyUntypedArrayMethods,
    };
    use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyTuple};
    use std::borrow::Cow;

    /// Add two 1-D float64 NumPy arrays and return a new float64 array.
    ///
    /// Boundary checks (CLAUDE.md: validate at the PyO3 boundary):
    ///   * wrong dtype or wrong number of dimensions: PyO3 raises `TypeError`
    ///     before this body runs, because the argument type is
    ///     `PyReadonlyArray1<f64>`;
    ///   * different lengths: we raise `ValueError` ourselves.
    #[pyfunction]
    fn add_arrays<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let sum = super::add_views(a.as_array(), b.as_array()).map_err(PyValueError::new_err)?;
        Ok(sum.into_pyarray(py))
    }

    // ----- reading a config dict (from yaml.safe_load of configs/drone.yaml or task_hover.yaml) -----

    /// A config dict plus its name, so error messages say which config has the problem.
    struct Cfg<'a, 'py> {
        dict: &'a Bound<'py, PyDict>,
        label: &'static str, // "drone_cfg" or "track_cfg"
    }

    impl<'py> Cfg<'_, 'py> {
        /// `cfg[name]`, which must be a dict.
        fn section(&self, name: &str) -> PyResult<Bound<'py, PyDict>> {
            let label = self.label;
            let item = self.dict.get_item(name)?.ok_or_else(|| {
                PyKeyError::new_err(format!("{label} is missing the section '{name}'"))
            })?;
            item.cast_into::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("{label}['{name}'] must be a dict")))
        }

        /// `cfg[section][key]`, as an untyped Python object.
        fn field(&self, section_name: &str, key: &str) -> PyResult<Bound<'py, PyAny>> {
            let label = self.label;
            self.section(section_name)?.get_item(key)?.ok_or_else(|| {
                PyKeyError::new_err(format!("{label} is missing '{section_name}.{key}'"))
            })
        }

        /// `cfg[key]`, a value at the top level of the config.
        fn top(&self, key: &str) -> PyResult<Bound<'py, PyAny>> {
            let label = self.label;
            self.dict
                .get_item(key)?
                .ok_or_else(|| PyKeyError::new_err(format!("{label} is missing '{key}'")))
        }

        fn f64(&self, section_name: &str, key: &str) -> PyResult<f64> {
            let label = self.label;
            self.field(section_name, key)?
                .extract::<f64>()
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{label}: '{section_name}.{key}' must be a number"
                    ))
                })
        }

        fn vec3(&self, section_name: &str, key: &str) -> PyResult<Vec3> {
            let label = self.label;
            self.field(section_name, key)?
                .extract::<[f64; 3]>()
                .map(Vec3::from_array)
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{label}: '{section_name}.{key}' must be a list of 3 numbers"
                    ))
                })
        }

        fn bool(&self, section_name: &str, key: &str) -> PyResult<bool> {
            let label = self.label;
            self.field(section_name, key)?
                .extract::<bool>()
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{label}: '{section_name}.{key}' must be true or false"
                    ))
                })
        }
    }

    /// Build `DroneParams` from the nested dict, using the same keys as `configs/drone.yaml`.
    fn params_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<DroneParams> {
        let cfg = Cfg {
            dict,
            label: "drone_cfg",
        };
        Ok(DroneParams {
            mass: cfg.f64("body", "mass")?,
            inertia_diag_b: cfg.vec3("body", "inertia_diag")?,
            arm_length: cfg.f64("body", "arm_length")?,
            thrust_min: cfg.f64("motors", "thrust_min")?,
            thrust_max: cfg.f64("motors", "thrust_max")?,
            motor_time_constant: cfg.f64("motors", "time_constant")?,
            yaw_torque_coeff: cfg.f64("motors", "yaw_torque_coeff")?,
            max_rates_b: cfg.vec3("limits", "max_rates")?,
            rate_gain_kp: cfg.vec3("controller", "rate_gain_kp")?,
            drag_enabled: cfg.bool("drag", "enabled")?,
            drag_linear: cfg.vec3("drag", "linear")?,
            gravity: cfg.f64("world", "gravity")?,
            physics_hz: cfg.f64("timing", "physics_hz")?,
            policy_hz: cfg.f64("timing", "policy_hz")?,
        })
    }

    /// Build the hover task's settings from the nested dict, using the keys of
    /// `configs/task_hover.yaml`. Only `task: hover` exists so far; the racing task is milestone 4.
    fn hover_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<HoverConfig> {
        let cfg = Cfg {
            dict,
            label: "track_cfg",
        };
        let task: String = cfg
            .top("task")?
            .extract()
            .map_err(|_| PyTypeError::new_err("track_cfg: 'task' must be a string"))?;
        if task != "hover" {
            return Err(PyValueError::new_err(format!(
                "track_cfg: task '{task}' is not available yet (only 'hover'; racing arrives in milestone 4)"
            )));
        }
        let target_w = cfg
            .top("target")?
            .extract::<[f64; 3]>()
            .map(Vec3::from_array)
            .map_err(|_| PyTypeError::new_err("track_cfg: 'target' must be a list of 3 numbers"))?;
        Ok(HoverConfig {
            target_w,
            max_seconds: cfg.f64("episode", "max_seconds")?,
            min_altitude: cfg.f64("termination", "min_altitude")?,
            max_distance: cfg.f64("termination", "max_distance")?,
            start_max_offset: cfg.f64("reset", "max_offset")?,
            start_min_altitude: cfg.f64("reset", "min_altitude")?,
            start_max_tilt: cfg.f64("reset", "max_tilt")?,
            start_max_speed: cfg.f64("reset", "max_speed")?,
            start_max_rate: cfg.f64("reset", "max_body_rate")?,
            weight_progress: cfg.f64("reward", "progress")?,
            weight_distance: cfg.f64("reward", "distance")?,
            weight_body_rate: cfg.f64("reward", "body_rate")?,
            weight_action_change: cfg.f64("reward", "action_change")?,
            crash_penalty: cfg.f64("reward", "crash")?,
        })
    }

    // ----- shared checks at the Python boundary -----

    fn no_task_error() -> PyErr {
        PyValueError::new_err(
            "this QuadSim was built without a track_cfg, so it has no task: pass track_cfg \
             (for example yaml.safe_load of configs/task_hover.yaml), or use step_physics",
        )
    }

    fn not_contiguous_error<E>(_: E) -> PyErr {
        PyValueError::new_err("internal error: a new output array is not contiguous")
    }

    /// The actions as one flat slice of `n * 4` values, checked. A C-contiguous array is used in
    /// place; anything else (a strided view) is copied once. Raises `ValueError` for the wrong
    /// shape or a NaN or infinity.
    fn checked_actions<'a>(
        actions: &'a PyReadonlyArray2<'_, f32>,
        n: usize,
    ) -> PyResult<Cow<'a, [f32]>> {
        let expected = [n, 4];
        if actions.shape() != expected {
            return Err(PyValueError::new_err(format!(
                "actions must have shape {:?}, got {:?}",
                expected,
                actions.shape()
            )));
        }
        let flat: Cow<'a, [f32]> = match actions.as_slice() {
            Ok(slice) => Cow::Borrowed(slice),
            Err(_) => Cow::Owned(actions.as_array().iter().copied().collect()),
        };
        if flat.iter().any(|a| !a.is_finite()) {
            return Err(PyValueError::new_err("actions contain NaN or infinity"));
        }
        Ok(flat)
    }

    // ----- the simulator -----

    /// A batch of `num_envs` drones stepped together.
    ///
    /// `drone_cfg` is the dict from `yaml.safe_load` of `configs/drone.yaml`. `track_cfg` is the
    /// dict from `yaml.safe_load` of a task file such as `configs/task_hover.yaml`; with it the
    /// batch is an environment (`reset`, `step`, `obs_dim`); without it, only physics is
    /// available (`step_physics`). `seed` seeds one random generator per drone, so the same seed
    /// always gives the same episodes, whatever `num_threads` is.
    ///
    /// Only the hover task exists so far (milestone 3); the racing task arrives in milestone 4.
    #[pyclass]
    struct QuadSim {
        batch: Batch,
    }

    #[pymethods]
    impl QuadSim {
        /// `num_threads` is how many CPU threads step the drones; `None` (the default) uses one
        /// per logical CPU. It never changes the results, only the speed.
        #[new]
        #[pyo3(signature = (num_envs, drone_cfg, track_cfg=None, seed=0, num_threads=None))]
        fn new(
            num_envs: usize,
            drone_cfg: &Bound<'_, PyDict>,
            track_cfg: Option<&Bound<'_, PyDict>>,
            seed: u64,
            num_threads: Option<usize>,
        ) -> PyResult<QuadSim> {
            let params = params_from_dict(drone_cfg)?;
            let batch = match track_cfg {
                Some(track_dict) => {
                    let task = hover_from_dict(track_dict)?;
                    Batch::with_task(num_envs, params, task, seed, num_threads)
                }
                None => Batch::new(num_envs, params, num_threads),
            }
            .map_err(PyValueError::new_err)?;
            Ok(QuadSim { batch })
        }

        #[getter]
        fn num_envs(&self) -> usize {
            self.batch.num_envs()
        }

        #[getter]
        fn num_threads(&self) -> usize {
            self.batch.num_threads()
        }

        /// Length of one observation (22 for the hover task), or `None` if built without a task.
        #[getter]
        fn obs_dim(&self) -> Option<usize> {
            self.batch.obs_dim()
        }

        /// The task name (`"hover"`), or `None` if built without a task.
        #[getter]
        fn task(&self) -> Option<&'static str> {
            self.batch.has_task().then_some("hover")
        }

        /// Names of the reward terms reported in `info["reward_terms"]`, in column order.
        #[getter]
        fn reward_term_names(&self) -> Vec<&'static str> {
            HOVER_REWARD_TERM_NAMES.to_vec()
        }

        /// Start a new episode in every drone and return the first observations, a float32 array
        /// of shape `[num_envs, obs_dim]`. With `seed`, every drone's random generator is first
        /// restarted from it, so the same seed always gives the same episodes.
        #[pyo3(signature = (seed=None))]
        fn reset<'py>(
            &mut self,
            py: Python<'py>,
            seed: Option<u64>,
        ) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let dim = self.batch.obs_dim().ok_or_else(no_task_error)?;
            let obs = PyArray2::<f32>::zeros(py, [self.batch.num_envs(), dim], false);
            {
                let mut writable = obs.readwrite();
                let slice = writable.as_slice_mut().map_err(not_contiguous_error)?;
                self.batch
                    .reset_task(seed, slice)
                    .map_err(PyValueError::new_err)?;
            }
            self.batch.check_states().map_err(PyValueError::new_err)?;
            Ok(obs)
        }

        /// Advance every drone by one 50 Hz policy step of the task.
        ///
        /// `actions` is a float32 array of shape `[num_envs, 4]`, each value nominally in [-1, 1]
        /// (anything outside is clipped). Returns `(obs, reward, terminated, truncated, final_obs,
        /// info)`:
        ///
        /// - `obs` `[N, obs_dim]` float32: what each drone sees next. A drone whose episode just
        ///   ended has already restarted, so its row is the first observation of the new episode.
        /// - `reward` `[N]` float32, `terminated` and `truncated` `[N]` bool.
        /// - `final_obs` `[N, obs_dim]` float32: the last observation of the episode that just
        ///   ended (only meaningful for drones where `terminated | truncated`).
        /// - `info`: a dict with `reward_terms` `[N, 5]` float32 (progress, distance penalty, rate
        ///   penalty, action penalty, crash; they add up to `reward`) and `termination_cause`
        ///   `[N]` uint8 (0 none, 3 altitude below the minimum, 4 too far from the target, 6 time limit).
        ///
        /// Raises `ValueError` for the wrong shape or a NaN action and `TypeError` for the wrong
        /// dtype. The GIL is released while stepping.
        fn step<'py>(
            &mut self,
            py: Python<'py>,
            actions: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<Bound<'py, PyTuple>> {
            let dim = self.batch.obs_dim().ok_or_else(no_task_error)?;
            let n = self.batch.num_envs();
            let flat = checked_actions(&actions, n)?;
            let actions_slice: &[f32] = &flat;

            let obs = PyArray2::<f32>::zeros(py, [n, dim], false);
            let final_obs = PyArray2::<f32>::zeros(py, [n, dim], false);
            let terms = PyArray2::<f32>::zeros(py, [n, NUM_REWARD_TERMS], false);
            let reward = PyArray1::<f32>::zeros(py, n, false);
            let terminated = PyArray1::<bool>::zeros(py, n, false);
            let truncated = PyArray1::<bool>::zeros(py, n, false);
            let cause = PyArray1::<u8>::zeros(py, n, false);
            {
                let (mut w_obs, mut w_final) = (obs.readwrite(), final_obs.readwrite());
                let (mut w_terms, mut w_reward) = (terms.readwrite(), reward.readwrite());
                let (mut w_term, mut w_trunc) = (terminated.readwrite(), truncated.readwrite());
                let mut w_cause = cause.readwrite();
                let mut buffers = StepBuffers {
                    obs: w_obs.as_slice_mut().map_err(not_contiguous_error)?,
                    final_obs: w_final.as_slice_mut().map_err(not_contiguous_error)?,
                    reward_terms: w_terms.as_slice_mut().map_err(not_contiguous_error)?,
                    reward: w_reward.as_slice_mut().map_err(not_contiguous_error)?,
                    terminated: w_term.as_slice_mut().map_err(not_contiguous_error)?,
                    truncated: w_trunc.as_slice_mut().map_err(not_contiguous_error)?,
                    cause: w_cause.as_slice_mut().map_err(not_contiguous_error)?,
                };
                let batch = &mut self.batch;
                py.detach(|| batch.step_task(actions_slice, &mut buffers))
                    .map_err(PyValueError::new_err)?;
            }
            self.batch.check_states().map_err(PyValueError::new_err)?;

            let info = PyDict::new(py);
            info.set_item("reward_terms", terms)?;
            info.set_item("termination_cause", cause)?;
            PyTuple::new(
                py,
                [
                    obs.into_any(),
                    reward.into_any(),
                    terminated.into_any(),
                    truncated.into_any(),
                    final_obs.into_any(),
                    info.into_any(),
                ],
            )
        }

        /// Advance every drone by one 50 Hz policy step (10 physics substeps).
        ///
        /// `actions` is a float32 array of shape `[num_envs, 4]`, each value nominally in
        /// [-1, 1] (anything outside is clipped). Raises `ValueError` for the wrong shape or a
        /// NaN, and `TypeError` for the wrong dtype. The GIL is released while stepping.
        fn step_physics<'py>(
            &mut self,
            py: Python<'py>,
            actions: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<()> {
            let flat = checked_actions(&actions, self.batch.num_envs())?;
            let batch = &mut self.batch;
            py.detach(|| batch.step(&flat))
                .map_err(PyValueError::new_err)?;
            self.batch.check_states().map_err(PyValueError::new_err)
        }

        /// The full 17-value state of every drone, as a new float64 array of shape `[num_envs, 17]`:
        /// position (3, world), quaternion (4, w x y z), velocity (3, world), body rates (3),
        /// motor thrusts (4).
        fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
            let out = PyArray2::<f64>::zeros(py, [self.batch.num_envs(), STATE_SIZE], false);
            {
                let mut writable = out.readwrite();
                let slice = writable.as_slice_mut().map_err(|_| {
                    PyValueError::new_err("internal error: new state array is not contiguous")
                })?;
                self.batch
                    .write_states(slice)
                    .map_err(PyValueError::new_err)?;
            }
            Ok(out)
        }

        /// Replace every drone's state from a float64 array of shape `[num_envs, 17]` (same layout
        /// as `get_state`). Raises `ValueError`, changing nothing, if any state has a NaN or a
        /// quaternion that is not unit length.
        fn set_state(&mut self, states: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
            let expected = [self.batch.num_envs(), STATE_SIZE];
            if states.shape() != expected {
                return Err(PyValueError::new_err(format!(
                    "states must have shape {:?}, got {:?}",
                    expected,
                    states.shape()
                )));
            }
            let copied: Vec<f64>;
            let flat: &[f64] = match states.as_slice() {
                Ok(slice) => slice,
                Err(_) => {
                    copied = states.as_array().iter().copied().collect();
                    &copied
                }
            };
            self.batch.set_states(flat).map_err(PyValueError::new_err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_views_adds_element_by_element() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let sum = add_views(a.view(), b.view()).unwrap();
        assert_eq!(sum.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_views_rejects_length_mismatch() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0]);
        assert!(add_views(a.view(), b.view()).is_err());
    }
}
