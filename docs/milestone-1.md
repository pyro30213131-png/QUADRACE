# Milestone 1: NumPy reference simulator

> **Exit test met locally: all non-parity tests in handoff section 7 pass.** CI has not run on these commits yet, because they have not been pushed. The handoff says CI must be green before a milestone is reported done, so treat this milestone as done once the push is made and both CI jobs are green. See `docs/STATUS.md`.

## What was built

- `python/quadrace/ref_sim.py`: a single quadrotor simulated in plain NumPy float64, following handoff section 7 step by step. It is slow on purpose and written to be read. It is the oracle: in milestone 2 the Rust simulator is ported from it, and the parity test requires the two to agree to 1e-9.
- `tests/test_ref_sim.py`: 31 test cases. Sixteen cover the bullets of the handoff's list of tests the simulator must pass before any learning (a few bullets are parametrized, and the saturation bullet is split into three tests). The other fifteen check the pieces those tests lean on (config loading, action mapping, drag, input checks, the substep order, and that a roll command leaves pitch and yaw alone).

The parity bullet in that list belongs to milestone 2, because it needs the Rust side.

## How to run it

```powershell
.venv\Scripts\Activate.ps1
pytest tests/test_ref_sim.py -q     # just this milestone
pytest -q                           # everything
```

No Rust build is needed for this milestone, but `import quadrace` needs the project installed once with `maturin develop --release`, as in milestone 0.

## How the code is organised

`ref_sim.py` is a set of small functions with no module-level state. Each takes the drone parameters and returns new arrays instead of changing its inputs.

| Part | Functions |
| --- | --- |
| Config | `DroneParams`, `load_drone_params`, `substeps_per_action` |
| Quaternions | `quat_multiply`, `quat_normalize`, `quat_to_rotation_matrix`, `rotate_body_to_world` |
| Motors | `motor_layout`, `mixing_matrix`, `total_thrust`, `body_torque` |
| Rigid body | `rigid_body_derivative` (the equations of motion), `rk4_step` |
| Control | `rate_controller`, `allocate` (invert the mixing matrix, then clamp), `motor_lag` |
| Stepping | `physics_substep` (the six steps of section 7, in order), `action_to_commands`, `policy_step` (10 substeps) |
| Helpers | `hover_action`, `hover_state`, `validate_state` |

The state is one 17-value array. `POS`, `QUAT`, `VEL`, `RATES`, `THRUST` are the slices that say where each part lives. Every vector name ends in `_w` or `_b` to say which frame it is in, and each physics equation has a comment naming it.

`physics_substep` is the function the Rust code has to reproduce. In order: the rate controller makes a desired torque, the mixing matrix is inverted to get four thrust commands, each is clamped to the motor limits, the motors lag toward those commands, RK4 advances the body with the (now lagged) thrusts held constant, and the quaternion is renormalized.

## Choices worth knowing

These bind the Rust port, so they are written down here.

1. **The rotation matrix is built from a normalized quaternion, even in RK4's intermediate stages.** RK4 evaluates the equations at slightly non-unit quaternions, and the rotation-matrix formula is only a rotation for unit ones. The Rust code must normalize in the same place or parity will drift.
2. **RK4 sees the lagged thrust.** Step 4 (motor lag) runs before step 5 (RK4), so the integrator uses the thrust after this substep's lag, not before. A test pins this order: a deliberately wrong version passed every other test.
3. **Actions outside [-1, 1] are clipped.** The handoff does not say what to do with them. I chose clipping, because the action mapping is only defined on that range. Changing it later is a one-line change plus a test.
4. **The thrust clamp uses `thrust_min` from the config** (0.0), not a hard-coded zero.
5. **"Reached within 10% in 0.2 s" is read as:** at t = 0.2 s the roll rate is between 4.5 and 5.5 rad/s. The handoff could also mean "reaches 4.5 at some point before 0.2 s"; the response here satisfies both readings.
6. **How the saturation bullet is tested.** At hover thrust, the largest roll or pitch command the action mapping allows (10 rad/s) needs 0.5 and 0.42 N m, and the motors can supply more, so it does not saturate (checked by running the allocation). Saturation tests therefore call `physics_substep` directly with a 100 rad/s roll command, which no action can produce. Two properties are checked: the thrust commands land exactly on the limits (motors 1 and 2 at maximum, 0 and 3 at zero), and the roll rate grows no faster than the torque limit allows and slower than an unlimited controller would.
7. **A full-scale yaw command does saturate at hover thrust.** Measured: at hover collective thrust, the 4 rad/s yaw command asks for two motors at about -0.31 N, which clamps to 0. So yaw authority is limited in normal flight, and the policy will meet this when it learns. No action is needed now; it is recorded so it is not a surprise in milestone 3.

## Results

Measured on 2026-11-15 on the owner's Windows machine (Python 3.12.10, NumPy 2.5.4):

| Check | Result |
| --- | --- |
| `pytest tests/test_ref_sim.py -q` | 31 passed |
| `pytest -q` (everything) | 43 passed |
| `cargo test -p quadsim`, `cargo fmt --check`, `cargo clippy -- -D warnings` | pass (Rust code unchanged this milestone) |
| Free fall, motors off, worst difference from `z0 - 4.905 t^2` over 1 s | 1.3e-14 m |
| Hover from hover thrust, worst altitude change over 5 s | 0 m (limit 1 mm) |
| Roll rate for a 5 rad/s command at 0.2 s | 5.23 rad/s (4.6% high; peak 5.35; limit +-10%) |
| Saturated roll rate (100 rad/s command) at 0.05 s | 19.1 rad/s, against a torque limit of 36.1 and an unlimited controller's 63.2 |
| World-frame angular momentum, worst relative drift over 1 s of tumbling | 7.2e-11 (the body rates changed by 6 rad/s in that time) |
| Exact quaternion formula, constant rate, equal inertias | 3e-15 after 1 step, 2e-12 after 500 steps |

**Mutation check.** To see whether the tests would notice a wrong simulator, I broke `ref_sim.py` in 15 different ways, one at a time (a flipped quaternion sign, a dropped gyroscopic term, a flipped pitch torque sign, a swapped yaw spin sign, no thrust clamp, no renormalization, RK4 replaced by Euler, a wrong lag order, and others), and ran the tests each time. 14 were caught at once. One was not (using the thrust from before the motor lag in RK4), so I added the test described in choice 2, and then all 15 were caught. The file was restored after each, and the check script is not part of the repo.

No test was weakened, skipped or deleted. Tolerances were chosen from the maths before the tests were run; the comments in the test file say how.

## Five questions you should be able to answer

1. **What is `ref_sim.py` for, if Rust is the real simulator?**
   It is the answer key. It is written first, slowly and plainly, and checked against physics we can work out by hand (free fall, conservation of angular momentum, exact rotation formulas). The Rust simulator is then ported from it and must agree to 1e-9. If the two disagree, there is a bug in the Rust port, because the Python was already checked. Without it, a sign error in fast Rust code would be invisible until training failed in some confusing way.

2. **Why does angular momentum being constant prove anything?**
   With no torque and no drag, a real body's angular momentum in the world frame cannot change, even though its body-frame rates do (the body tumbles). If the quaternion product, the rotation matrix, or Euler's equation had a sign wrong, `R J omega` would drift. The test caught three different deliberate bugs this way.

3. **What happens if you ask for a body rate the motors cannot deliver?**
   The rate controller asks for a torque. Inverting the mixing matrix may give a negative thrust for some motor, or one above the maximum. Each command is clamped to the motor limits, so the real torque is smaller than requested and the rate rises more slowly. That is saturation.

4. **Why does the order of the substep matter?**
   The motor lag runs before RK4, so the integrator uses the thrust after this substep's lag. If the order were swapped, the drone would respond one substep (2 ms) later. That is small but it adds up, and the Rust code would then disagree with this one. A test pins the order.

5. **Why is the quaternion renormalized after every substep?**
   RK4 on the quaternion does not keep its length exactly 1, and a non-unit quaternion is not a pure rotation, so errors would grow. Dividing by the length after each substep removes the drift. The test `test_quaternion_stays_unit_length_over_many_policy_steps` checks the length stays within 1e-12 over 2,000 substeps.
