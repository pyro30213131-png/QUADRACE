//! quadsim: the Rust half of quadrace.
//!
//! Milestone 0 only proves the toolchain: one function, `add_arrays`, that adds
//! two NumPy arrays. The simulator itself arrives in milestone 2.
//!
//! How the pieces fit:
//!   * `add_views` is plain Rust. It knows nothing about Python, so it can be
//!     unit-tested with `cargo test`.
//!   * The `_quadsim` module at the bottom is the thin PyO3 layer. It converts
//!     NumPy arrays into Rust views, calls the plain function, and turns a Rust
//!     error into a Python exception.

use numpy::ndarray::{Array1, ArrayView1};

/// Element-wise sum of two vectors.
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
/// last part of `module-name` in pyproject.toml. Every `#[pyfunction]` inside
/// this block is added to the module automatically.
#[pyo3::pymodule]
mod _quadsim {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::{Bound, PyResult, Python, exceptions::PyValueError, pyfunction};

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
