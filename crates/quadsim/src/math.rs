//! Hand-written 3-vector, quaternion and 3x3 matrix. No nalgebra, on purpose: the whole
//! simulator should be readable from this file down.
//!
//! Conventions (handoff section 6):
//! - Quaternion is scalar first, `(w, x, y, z)`, Hamilton product, and rotates body to world.
//! - `q` and `-q` are the same rotation.
//! - Names carry their frame: `_w` is the world frame, `_b` is the body frame.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3-vector of f64. Which frame it is in is told by the variable name, not the type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn from_array(a: [f64; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }

    pub fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product `self x other`.
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Component-wise product, for a diagonal matrix times a vector.
    pub fn mul_elementwise(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    /// Component-wise quotient, for the inverse of a diagonal matrix times a vector.
    pub fn div_elementwise(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

/// A quaternion `(w, x, y, z)`, scalar first. As an attitude it is body to world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Quat {
        Quat { w, x, y, z }
    }

    /// Hamilton product `self (x) other`. Not commutative.
    pub fn multiply(self, other: Quat) -> Quat {
        Quat::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }

    pub fn norm(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Scaled to unit length: the same rotation with the numerical drift removed.
    pub fn normalized(self) -> Quat {
        let n = self.norm();
        Quat::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Rotation matrix that maps body-frame vectors to world-frame vectors.
    ///
    /// The quaternion is normalized first. RK4 evaluates the dynamics at intermediate
    /// quaternions that are very slightly off unit length, and these formulas are only a
    /// rotation for unit quaternions. The NumPy reference does the same (`ref_sim.py`).
    pub fn to_rotation_matrix(self) -> Mat3 {
        let q = self.normalized();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        Mat3 {
            rows: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
        }
    }

    /// `vec_w = R(q) vec_b`.
    pub fn rotate_body_to_world(self, vec_b: Vec3) -> Vec3 {
        self.to_rotation_matrix().mul_vec(vec_b)
    }
}

impl Add for Quat {
    type Output = Quat;
    fn add(self, other: Quat) -> Quat {
        Quat::new(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }
}

impl Mul<f64> for Quat {
    type Output = Quat;
    fn mul(self, s: f64) -> Quat {
        Quat::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

/// A 3x3 matrix, stored by rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub rows: [[f64; 3]; 3],
}

impl Mat3 {
    /// `M v`.
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let r = &self.rows;
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    }

    /// `M^T v`. For a rotation matrix this is the inverse rotation (world to body).
    pub fn transpose_mul_vec(&self, v: Vec3) -> Vec3 {
        let r = &self.rows;
        Vec3::new(
            r[0][0] * v.x + r[1][0] * v.y + r[2][0] * v.z,
            r[0][1] * v.x + r[1][1] * v.y + r[2][1] * v.z,
            r[0][2] * v.x + r[1][2] * v.y + r[2][2] * v.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn ninety_degree_yaw_quaternion_maps_body_x_to_world_y() {
        let half = 0.5_f64.sqrt();
        let v_w = Quat::new(half, 0.0, 0.0, half).rotate_body_to_world(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(v_w.x, 0.0) && close(v_w.y, 1.0) && close(v_w.z, 0.0));
    }

    #[test]
    fn identity_quaternion_leaves_a_vector_unchanged() {
        let v_b = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(Quat::IDENTITY.rotate_body_to_world(v_b), v_b);
    }

    #[test]
    fn quaternion_product_with_identity_changes_nothing() {
        let q = Quat::new(0.5, 0.5, -0.5, 0.5);
        assert_eq!(q.multiply(Quat::IDENTITY), q);
        assert_eq!(Quat::IDENTITY.multiply(q), q);
    }

    #[test]
    fn quaternion_product_is_not_commutative() {
        // Rotating 90 degrees about x then about y differs from the other order.
        let half = 0.5_f64.sqrt();
        let about_x = Quat::new(half, half, 0.0, 0.0);
        let about_y = Quat::new(half, 0.0, half, 0.0);
        assert_ne!(about_x.multiply(about_y), about_y.multiply(about_x));
    }

    #[test]
    fn rotation_matrix_is_inverted_by_its_transpose() {
        let q = Quat::new(0.3, -0.5, 0.7, 0.2);
        let r = q.to_rotation_matrix();
        let v = Vec3::new(0.4, -1.2, 2.5);
        let round_trip = r.transpose_mul_vec(r.mul_vec(v));
        assert!(close(round_trip.x, v.x) && close(round_trip.y, v.y) && close(round_trip.z, v.z));
    }

    #[test]
    fn cross_product_follows_the_right_hand_rule() {
        let z = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn normalizing_a_quaternion_gives_unit_length() {
        assert!(close(
            Quat::new(2.0, 0.0, 0.0, 0.0).normalized().norm(),
            1.0
        ));
    }
}
