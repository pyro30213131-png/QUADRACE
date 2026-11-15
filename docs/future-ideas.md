# Future ideas (parked)

> **PARKED. Do not work on anything on this page.** These are ideas the owner wants to remember, not tasks. They are not part of any milestone. Nothing here may be started until milestones 0 to 8 are finished **and** the owner asks. Some of it is on the handoff's out-of-scope list, and turning any of it into work means proposing a change to `AGENT_HANDOFF.md` and bumping its version after the owner approves (see `CLAUDE.md`). No code, config keys, API arguments or hooks should be added now to prepare for it.
>
> Written on 2026-11-15 from a brainstorm with the owner. Nothing below has been built, tested or measured.

## Idea 1: a collaborative design-and-reward challenge

### The idea

Friends of the owner each design their own drone, and/or their own reward function, within limits enforced by the Rust simulator. Each trains locally on their own machine. Everyone is compared by lap time on the same track, using the same official evaluation.

### What the project already has that would help

- A drone is already a YAML file (`configs/drone.yaml`; the Swift drone gets `drone_swift.yaml` in milestone 6), so a friend's drone would be one more file of the same kind.
- Lap time is already defined by the simulator's gate logic (handoff section 8: the time between two consecutive passes of gate 0, evaluation starting at rest). It does not depend on the reward, so a friend cannot raise their score by exploiting their reward.
- The handoff has the simulator return the reward terms separately (progress, gate bonus, rate penalty, action penalty, crash). A reward could later be a recombination of those terms in Python, with no Rust changes.
- `DroneParams::validate()` in `crates/quadsim/src/params.rs` already rejects impossible values, and could be extended to design limits.

### Design limits, enforced in Rust (to decide)

- The simulator would refuse an out-of-range drone with a clear error, the way it already refuses a negative mass.
- The simulator models only mass, inertia, arm length, motor thrust, motor time constant, yaw coefficient, rate limits and drag. Battery voltage sag and propeller aerodynamics are out of scope, so "real constraints" can only mean limits on those values.
- Two options:
  - **Ranges and trade-offs.** Allowed ranges per parameter, plus consistency rules (for example, inertia must be plausible for the mass and arm length, and more thrust per motor costs mass).
  - **A parts catalog.** Motors, frames and batteries, each with a mass and thrust from real datasheets. A drone is a choice of parts, and its physical values follow from the choice. This is fairer by construction and gives real trade-offs, but the catalog data would have to be sourced and verified (handoff section 13 says not to invent values).
- Whether the rate-controller gains and the max body rates count as part of the design or are fixed is undecided.
- Current limitation: the closed-form motor allocation in `motors.rs` assumes the X layout, so designs would be quads with different parameters. A different motor layout would mean rewriting that function.

### Trust

Limits compiled into the simulator stop honest mistakes, not cheating: anyone can edit their own copy. The score that counts would come from the owner re-running the official evaluation on a clean build, using the submitted drone file and trained policy (with its observation-normalization statistics). The evaluation does not need the contributor's reward function.

### Training locally

- The owner's setup needed a Rust toolchain, Visual Studio Build Tools and a workaround for the space in the Windows username. Asking every friend to repeat that is the biggest barrier.
- A possible fix: CI builds ready-made wheels (installable packages, `maturin build` already works) for Windows and Linux, so friends need only Python and PyTorch. The limits would be compiled in, and the owner would verify on the same build. Publishing builds would need a licence decision first (an "ask the owner" item in `CLAUDE.md`).
- Friends' hardware differs. PyTorch training with CUDA needs an NVIDIA GPU; whether CPU training is practical is unknown.

### Making comparisons fair

- Fix the training budget in simulator steps, not hours or wall-clock time.
- Use several seeds per entry (about 3), because one training run differs from another by a few percent. A gap smaller than the seed-to-seed spread means nothing.
- Change one thing at a time, in leagues:
  - **Drone league:** same reward and PPO settings, different drones.
  - **Reward league:** same drone, different rewards.
  - **Open league:** anything within the limits.
- Rank by lap time, with a minimum lap-completion rate (the handoff's targets are 95% and 90%).

### Risks and open questions

- Reward exploits: rewards that rise while lap completion stays flat. Ranking by lap time and completion rate avoids rewarding them.
- Running a friend's Python reward code means running untrusted code. Declarative reward weights are safer than scripts.
- `CLAUDE.md` says not to commit checkpoints, so submitted policies would need another home.
- The licence question (above) must be answered by the owner.
- Training time per run is not measured yet, and the handoff says to ask before any run over an hour.
- The drone numbers in `drone.yaml` are the handoff's approximate values, not from a verified source.

### A possible staging (a suggestion, not a decision)

1. **Owner only:** run sweeps. Fix the reward and vary thrust-to-weight, mass and motor lag, and plot lap time. Then fix the drone and vary the reward weights. This gives a real result even if no friend takes part.
2. **Entries:** one folder per entry with a drone file, a check that rejects out-of-range designs, a smoke test in CI, and a leaderboard script that re-runs the official evaluation on a submitted policy.
3. **Custom rewards:** only after the earlier steps work.

## Idea 2: racing several drones at once on the same course

Explicitly a post-project idea.

- **Cheap version: ghost racing.** Every entry flies the course alone, and the replay shows them all together as ghosts, side by side. There is no physics change. The handoff's training-progress replay already plans side-by-side and ghost-trail views.
- **Real version: interactive multi-drone racing.** It needs collisions between drones, observations of other drones and multi-agent training. The handoff lists multi-drone racing as out of scope. It would be a separate phase with its own design, and nothing should be built now to prepare for it.

## Questions to answer when this is picked up

- Few free parameters, or a parts catalog with real trade-offs?
- Do friends train on their own machines, or do some only send a drone file?
- A friendly competition, or a public benchmark?
- Which tracks count: the figure-8, the Split-S, or both?

## What is already compatible without any work

- Every physical constant lives in YAML (handoff rule 8).
- Reward terms are returned separately (handoff section 5).
- Lap time comes from the simulator, not the reward (handoff section 8).
