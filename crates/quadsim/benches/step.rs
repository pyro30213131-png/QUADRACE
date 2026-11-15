//! Simulator throughput benchmark (milestone 2). Run it with `cargo bench -p quadsim`.
//!
//! One "env step" is one 50 Hz policy step of one drone: 10 physics substeps, each with the rate
//! controller, allocation, motor lag and an RK4 step. The numbers criterion prints as
//! `thrpt` ("elements per second") are env steps per second.
//!
//! This measures the Rust simulator only, with no Python in the loop.
//!
//! The drone constants below are a copy of `configs/drone.yaml` (the Rust code has no YAML
//! reader; the Python layer reads the file). They only set how much work is timed, and the work
//! does not depend on their exact values.

use _quadsim::batch::Batch;
use _quadsim::math::Vec3;
use _quadsim::params::DroneParams;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

fn bench_params() -> DroneParams {
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

/// 64 different action sets for `n` drones: near hover with small wiggles, so the drones keep
/// flying about instead of all doing the same thing. Cycled through, one per iteration.
fn action_sets(n: usize) -> Vec<Vec<f32>> {
    (0..64)
        .map(|set| {
            (0..n * 4)
                .map(|k| {
                    let wiggle = ((set * 13 + k) as f32 * 0.37).sin() * 0.3;
                    if k % 4 == 0 {
                        -0.57 + wiggle * 0.3
                    } else {
                        wiggle
                    }
                })
                .collect()
        })
        .collect()
}

/// One drone, on the calling thread: the cost of a single env step with no threading at all.
fn single_env(c: &mut Criterion) {
    let mut batch = Batch::new(1, bench_params(), Some(1)).unwrap();
    let actions = action_sets(1);
    let mut i = 0;
    let mut group = c.benchmark_group("single_env");
    group.throughput(Throughput::Elements(1));
    group.bench_function("step_sequential", |b| {
        b.iter(|| {
            batch.step_sequential(black_box(&actions[i % 64])).unwrap();
            i += 1;
        })
    });
    group.finish();
}

/// Many drones, on the calling thread, then spread over 2, 4, 8 and 16 threads.
fn batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_step");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for n in [1024_usize, 4096, 16384] {
        let actions = action_sets(n);
        group.throughput(Throughput::Elements(n as u64));

        let mut sequential = Batch::new(n, bench_params(), Some(1)).unwrap();
        let mut i = 0;
        group.bench_function(BenchmarkId::new("sequential", n), |b| {
            b.iter(|| {
                sequential
                    .step_sequential(black_box(&actions[i % 64]))
                    .unwrap();
                i += 1;
            })
        });

        for threads in [2_usize, 4, 8, 16] {
            let mut batch = Batch::new(n, bench_params(), Some(threads)).unwrap();
            let mut i = 0;
            group.bench_function(BenchmarkId::new(format!("{threads}_threads"), n), |b| {
                b.iter(|| {
                    batch.step(black_box(&actions[i % 64])).unwrap();
                    i += 1;
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, single_env, batched);
criterion_main!(benches);
