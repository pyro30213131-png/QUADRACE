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
    use crate::batch::Batch;
    use crate::dynamics::STATE_SIZE;
    use crate::math::Vec3;
    use crate::params::DroneParams;
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyUntypedArrayMethods,
    };
    use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

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

    // ----- reading drone_cfg (the dict from yaml.safe_load("configs/drone.yaml")) -----

    /// `cfg[name]`, which must be a dict.
    fn section<'py>(cfg: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let item = cfg.get_item(name)?.ok_or_else(|| {
            PyKeyError::new_err(format!("drone_cfg is missing the section '{name}'"))
        })?;
        item.cast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("drone_cfg['{name}'] must be a dict")))
    }

    /// `cfg[section][key]`, as an untyped Python object.
    fn field<'py>(
        cfg: &Bound<'py, PyDict>,
        section_name: &str,
        key: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        section(cfg, section_name)?.get_item(key)?.ok_or_else(|| {
            PyKeyError::new_err(format!("drone_cfg is missing '{section_name}.{key}'"))
        })
    }

    fn get_f64(cfg: &Bound<'_, PyDict>, section_name: &str, key: &str) -> PyResult<f64> {
        field(cfg, section_name, key)?
            .extract::<f64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "drone_cfg: '{section_name}.{key}' must be a number"
                ))
            })
    }

    fn get_vec3(cfg: &Bound<'_, PyDict>, section_name: &str, key: &str) -> PyResult<Vec3> {
        field(cfg, section_name, key)?
            .extract::<[f64; 3]>()
            .map(Vec3::from_array)
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "drone_cfg: '{section_name}.{key}' must be a list of 3 numbers"
                ))
            })
    }

    fn get_bool(cfg: &Bound<'_, PyDict>, section_name: &str, key: &str) -> PyResult<bool> {
        field(cfg, section_name, key)?
            .extract::<bool>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "drone_cfg: '{section_name}.{key}' must be true or false"
                ))
            })
    }

    /// Build `DroneParams` from the nested dict, using the same keys as `configs/drone.yaml`.
    fn params_from_dict(cfg: &Bound<'_, PyDict>) -> PyResult<DroneParams> {
        Ok(DroneParams {
            mass: get_f64(cfg, "body", "mass")?,
            inertia_diag_b: get_vec3(cfg, "body", "inertia_diag")?,
            arm_length: get_f64(cfg, "body", "arm_length")?,
            thrust_min: get_f64(cfg, "motors", "thrust_min")?,
            thrust_max: get_f64(cfg, "motors", "thrust_max")?,
            motor_time_constant: get_f64(cfg, "motors", "time_constant")?,
            yaw_torque_coeff: get_f64(cfg, "motors", "yaw_torque_coeff")?,
            max_rates_b: get_vec3(cfg, "limits", "max_rates")?,
            rate_gain_kp: get_vec3(cfg, "controller", "rate_gain_kp")?,
            drag_enabled: get_bool(cfg, "drag", "enabled")?,
            drag_linear: get_vec3(cfg, "drag", "linear")?,
            gravity: get_f64(cfg, "world", "gravity")?,
            physics_hz: get_f64(cfg, "timing", "physics_hz")?,
            policy_hz: get_f64(cfg, "timing", "policy_hz")?,
        })
    }

    // ----- the simulator -----

    /// A batch of `num_envs` drones stepped together.
    ///
    /// `drone_cfg` is the dict from `yaml.safe_load` of `configs/drone.yaml`. Every drone
    /// starts level and motionless at the origin with hover thrust; use `set_state` to place them.
    ///
    /// This is the physics-only version (milestone 2). Observations, rewards, gates and the
    /// `track_cfg` and `seed` arguments of handoff section 5 arrive with the environment, in
    /// milestones 3 and 4, because nothing here is random or needs a track yet.
    #[pyclass]
    struct QuadSim {
        batch: Batch,
    }

    #[pymethods]
    impl QuadSim {
        /// `num_threads` is how many CPU threads step the drones; `None` (the default) uses one
        /// per logical CPU. It never changes the results, only the speed.
        #[new]
        #[pyo3(signature = (num_envs, drone_cfg, num_threads=None))]
        fn new(
            num_envs: usize,
            drone_cfg: &Bound<'_, PyDict>,
            num_threads: Option<usize>,
        ) -> PyResult<QuadSim> {
            let params = params_from_dict(drone_cfg)?;
            let batch = Batch::new(num_envs, params, num_threads).map_err(PyValueError::new_err)?;
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
            let expected = [self.batch.num_envs(), 4];
            if actions.shape() != expected {
                return Err(PyValueError::new_err(format!(
                    "actions must have shape {:?}, got {:?}",
                    expected,
                    actions.shape()
                )));
            }
            // A C-contiguous array is used in place; anything else (a strided view) is copied once.
            let copied: Vec<f32>;
            let flat: &[f32] = match actions.as_slice() {
                Ok(slice) => slice,
                Err(_) => {
                    copied = actions.as_array().iter().copied().collect();
                    &copied
                }
            };
            if flat.iter().any(|a| !a.is_finite()) {
                return Err(PyValueError::new_err("actions contain NaN or infinity"));
            }
            let batch = &mut self.batch;
            py.detach(|| batch.step(flat))
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
