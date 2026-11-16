"""See a trained policy fly: record a rollout and view it in 3D.

    python -m quadrace.viz --checkpoint runs/hover1/checkpoint_000381.pt           # GIF + still image
    python -m quadrace.viz --checkpoint runs/hover1/checkpoint_000381.pt --rerun   # also the Rerun viewer

Two ways to look at the same recording:

- **Rerun** (`--rerun`): an interactive 3D viewer. Drag to rotate and zoom, scrub the timeline,
  and watch the distance plots. The recording is also saved as an `.rrd` file you can reopen later
  with `rerun <file>`.
- **A GIF and a PNG** (matplotlib): a fixed animation of a 3D scene next to a plot of the distance
  to the target, plus a still of the flight paths. Easy to share and needs no viewer.

This is a simple viewer for the hover task. The handoff plans Rerun for milestone 4 (a live view of
the policy during gate-racing training, and a replay) and videos for milestone 8; this lets the
hover policy be watched now. matplotlib and rerun are imported only when drawing, so recording a
rollout needs neither.

The rollout is flown on its own small simulator (a few drones), never on the training environments
(CLAUDE.md: never render the training environments).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from quadrace import ppo, ref_sim

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Rollout:
    """What the drones did. `T` frames at the policy rate, `N` drones.

    `pos_w` is the position in the world frame, `quat` the attitude (w, x, y, z, body to world) and
    `thrust` the four motor thrusts. A drone whose episode ended early (it hit the ground or flew
    away) is frozen at its last state from then on; `ended_at[i]` is the frame where that happened,
    or -1 if drone i flew the whole rollout.
    """

    time_s: np.ndarray  # (T,)
    pos_w: np.ndarray  # (T, N, 3)
    quat: np.ndarray  # (T, N, 4)
    thrust: np.ndarray  # (T, N, 4)
    target_w: np.ndarray  # (3,)
    ended_at: np.ndarray  # (N,) int

    @property
    def distance_to_target(self) -> np.ndarray:
        """Distance from each drone to the target at every frame, shape (T, N)."""
        return np.linalg.norm(self.pos_w - self.target_w, axis=2)

    @property
    def start_tilt_deg(self) -> np.ndarray:
        """How far each drone's body z axis starts from straight up, in degrees, shape (N,)."""
        x, y = self.quat[0, :, 1], self.quat[0, :, 2]
        return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def select_drones(rollout: Rollout, indices: np.ndarray) -> Rollout:
    """A rollout with only the drones at `indices`, in that order."""
    return Rollout(rollout.time_s, rollout.pos_w[:, indices], rollout.quat[:, indices], rollout.thrust[:, indices],
                   rollout.target_w, rollout.ended_at[indices])


def record_rollout(
    policy: ppo.Policy,
    drone_cfg: dict,
    task_cfg: dict,
    num_drones: int = 3,
    seed: int = 2026,
    seconds: float | None = None,
    hardest_of: int | None = None,
) -> Rollout:
    """Fly `num_drones` drones with the deterministic policy and record every policy step.

    Each drone starts from the task's random reset distribution (drone i uses the i-th stream of
    `seed`), so the starts are not hand-picked. `seconds` defaults to the task's episode length.

    With `hardest_of=K` (K >= num_drones), K drones are flown from random starts and the
    `num_drones` that START MOST TILTED are kept, most tilted first. The starts are still drawn from
    the task's own distribution; only the choice among them is made for drama, so show it as such.
    """
    import quadsim  # imported here so the rest of this module loads without the compiled extension

    policy_hz = drone_cfg["timing"]["policy_hz"]
    frames = round((seconds if seconds is not None else task_cfg["episode"]["max_seconds"]) * policy_hz)
    flown = max(num_drones, hardest_of) if hardest_of else num_drones
    sim = quadsim.QuadSim(flown, drone_cfg, task_cfg, seed=seed, num_threads=1)
    obs = sim.reset(seed=seed)
    num_drones_flown = num_drones
    num_drones = flown

    pos = np.zeros((frames, num_drones, 3))
    quat = np.zeros((frames, num_drones, 4))
    thrust = np.zeros((frames, num_drones, 4))
    ended_at = np.full(num_drones, -1, dtype=np.int64)

    def store(frame: int, state: np.ndarray) -> None:
        alive = ended_at < 0
        pos[frame, alive] = state[alive, ref_sim.POS]
        quat[frame, alive] = state[alive, ref_sim.QUAT]
        thrust[frame, alive] = state[alive, ref_sim.THRUST]
        if frame > 0:  # a drone that has ended stays where it was
            pos[frame, ~alive] = pos[frame - 1, ~alive]
            quat[frame, ~alive] = quat[frame - 1, ~alive]
            thrust[frame, ~alive] = thrust[frame - 1, ~alive]

    store(0, sim.get_state())
    for frame in range(1, frames):
        obs, _, terminated, _, _, _ = sim.step(policy.act(obs, deterministic=True))
        # A drone that just terminated has already restarted inside `step`, so its state now is a new
        # episode's start: note the frame and freeze it at its last real state (`store` does that).
        # The rollout is one step shorter than the time limit, so no drone is ever cut off by it.
        ended_at[terminated & (ended_at < 0)] = frame
        store(frame, sim.get_state())
    time_s = np.arange(frames) / policy_hz
    rollout = Rollout(time_s, pos, quat, thrust, np.array(task_cfg["target"], dtype=np.float64), ended_at)
    if flown > num_drones_flown:
        rollout = select_drones(rollout, np.argsort(-rollout.start_tilt_deg, kind="stable")[:num_drones_flown])
    return rollout


def display_offsets(num: int, spacing: float = 2.4) -> np.ndarray:
    """Where to draw each drone's copy of the scene, shape (num, 3).

    Every drone flies the same task to the same target point, so drawn in one place they would pile
    up on top of each other. Each is shifted sideways along x instead. This moves only the picture:
    the flights are unchanged.
    """
    offsets = np.zeros((num, 3))
    offsets[:, 0] = (np.arange(num) - (num - 1) / 2.0) * spacing
    return offsets


def drone_geometry(pos_w: np.ndarray, quat: np.ndarray, arm_length: float, scale: float = 1.0) -> dict:
    """Points to draw one drone, in the world frame. `scale` enlarges the drone (real drones are
    tiny next to the room, so the views draw them bigger; the position is not scaled).

    Returns `motors` (4, 3): where the four motors are; `nose` (3,): a point ahead of the drone along
    its body x axis, so you can see which way it faces; `up` (3,): a point above it along body z.
    """
    rot = ref_sim.quat_to_rotation_matrix(np.asarray(quat, dtype=np.float64))  # body to world
    x_b, y_b, _ = ref_sim.motor_layout(arm_length)
    motors_b = scale * np.stack([x_b, y_b, np.zeros(4)], axis=1)  # (4, 3) in the body frame
    motors_w = pos_w + motors_b @ rot.T
    nose_w = pos_w + rot @ np.array([1.6 * scale * arm_length, 0.0, 0.0])
    up_w = pos_w + rot @ np.array([0.0, 0.0, 0.8 * scale * arm_length])
    return {"motors": motors_w, "nose": nose_w, "up": up_w}


def _circle(center: np.ndarray, rot: np.ndarray, radius: float, points: int = 16) -> np.ndarray:
    """A circle of the given radius around `center` in the drone's own horizontal plane, (points, 3)."""
    angle = np.linspace(0.0, 2.0 * np.pi, points)
    ring_b = radius * np.stack([np.cos(angle), np.sin(angle), np.zeros(points)], axis=1)
    return center + ring_b @ rot.T


# --------------------------------------------------------------------------
# matplotlib: a GIF and a still image
# --------------------------------------------------------------------------


def render_gif(
    rollout: Rollout,
    path: Path,
    arm_length: float,
    fps: int = 25,
    stride: int = 2,
    scale: float = 5.0,
    spacing: float = 2.4,
    tolerance_m: float = 0.1,
    window_s: float = 10.0,
    title: str = "Hover policy, deterministic",
    dpi: int = 70,
) -> None:
    """Draw the rollout as an animated GIF: a 3D scene on the left (one drone per slot, side by
    side) and the distance to the target over time on the right. `stride` keeps every `stride`-th
    policy step (2 gives real time at 25 fps from the 50 Hz policy)."""
    import matplotlib

    matplotlib.use("Agg")  # draw to memory; no window
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = list(range(0, len(rollout.time_s), stride))
    num = rollout.pos_w.shape[1]
    offsets = display_offsets(num, spacing)
    colors = plt.cm.tab10(np.arange(num) % 10)
    distance = rollout.distance_to_target

    fig = plt.figure(figsize=(12.0, 5.6), dpi=dpi)
    grid_spec = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], left=0.0, right=0.98, top=0.9, bottom=0.11, wspace=0.1)
    scene = fig.add_subplot(grid_spec[0, 0], projection="3d")
    plot = fig.add_subplot(grid_spec[0, 1])
    fig.suptitle(title, fontsize=12)

    # --- the scene: ground, then per drone a target marker, trail, arms, rotors and nose ---
    half_x = spacing * (num - 1) / 2.0 + 1.7
    x_lim, y_lim, z_lim = (-half_x, half_x), (-1.7, 1.7), (0.0, 3.4)
    scene.set_xlim(*x_lim)
    scene.set_ylim(*y_lim)
    scene.set_zlim(*z_lim)
    scene.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))
    scene.set_xticks([])
    scene.set_yticks([])
    scene.set_zticks([0, 1, 2, 3])
    scene.set_zlabel("height (m)")
    for gx in np.arange(np.ceil(x_lim[0]), x_lim[1] + 0.01, 1.0):
        scene.plot([gx, gx], list(y_lim), [0, 0], color="0.85", lw=0.6)
    for gy in np.arange(np.ceil(y_lim[0]), y_lim[1] + 0.01, 1.0):
        scene.plot(list(x_lim), [gy, gy], [0, 0], color="0.85", lw=0.6)
    for i in range(num):
        target = rollout.target_w + offsets[i]
        scene.scatter(*target, s=110, c="red", marker="x", linewidths=2.5, depthshade=False)
        scene.plot([target[0]] * 2, [target[1]] * 2, [0, target[2]], color="red", lw=0.8, ls=":")
        ring = _circle(target, np.eye(3), tolerance_m)
        scene.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="red", lw=1.0)

    trails, arms, rotors, noses, shadows = [], [], [], [], []
    for i in range(num):
        (trail,) = scene.plot([], [], [], color=colors[i], lw=1.2, alpha=0.8)
        (shadow,) = scene.plot([], [], [], color="0.55", lw=1.0, alpha=0.5)  # the trail projected on the ground
        arms.append([scene.plot([], [], [], color="0.15", lw=2.4)[0] for _ in range(2)])  # the X's two diagonals
        rotors.append([scene.plot([], [], [], color=colors[i], lw=1.8)[0] for _ in range(4)])
        (nose,) = scene.plot([], [], [], color="black", lw=3.2)
        trails.append(trail)
        shadows.append(shadow)
        noses.append(nose)
    clock = scene.text2D(0.03, 0.93, "", transform=scene.transAxes, fontsize=11)
    scene.text2D(0.03, 0.03, f"each drone flies its own copy of the task; drawn {scale:g}x life size, "
                 "red x = target, red ring = 0.1 m", transform=scene.transAxes, fontsize=7.5, color="0.35")

    # --- the plot: distance to the target over time, one line per drone ---
    window_start = max(rollout.time_s[-1] - window_s, 0.0)
    plot.axvspan(window_start, rollout.time_s[-1], color="green", alpha=0.08, label=f"final {window_s:.0f} s window")
    plot.axhline(tolerance_m, color="red", ls="--", lw=1.0, label=f"{tolerance_m} m tolerance")
    for i in range(num):
        plot.plot(rollout.time_s, np.maximum(distance[:, i], 1e-4), color=colors[i], lw=1.4,
                  label=f"drone {i + 1} (start tilt {rollout.start_tilt_deg[i]:.0f}°)")
    plot.set_yscale("log")
    plot.set_ylim(5e-4, 3.0)
    plot.set_xlim(0.0, rollout.time_s[-1])
    plot.set_xlabel("time (s)")
    plot.set_ylabel("distance to target (m, log scale)")
    plot.legend(loc="upper right", fontsize=8)
    plot.grid(True, which="both", alpha=0.25)
    cursor = plot.axvline(0.0, color="black", lw=1.0)

    def draw(frame_number: int) -> None:
        t = frames[frame_number]
        for i in range(num):
            trail_to = rollout.pos_w[: t + 1, i] + offsets[i]
            trails[i].set_data_3d(trail_to[:, 0], trail_to[:, 1], trail_to[:, 2])
            shadows[i].set_data_3d(trail_to[:, 0], trail_to[:, 1], np.zeros(len(trail_to)))
            centre = rollout.pos_w[t, i] + offsets[i]
            geo = drone_geometry(centre, rollout.quat[t, i], arm_length, scale)
            motors = geo["motors"]
            for diagonal, (a, b) in enumerate([(0, 1), (2, 3)]):  # motors 0-1 and 2-3 are opposite corners
                arms[i][diagonal].set_data_3d(motors[[a, b], 0], motors[[a, b], 1], motors[[a, b], 2])
            rot = ref_sim.quat_to_rotation_matrix(rollout.quat[t, i])
            for m in range(4):
                ring = _circle(motors[m], rot, 0.45 * scale * arm_length)
                rotors[i][m].set_data_3d(ring[:, 0], ring[:, 1], ring[:, 2])
            noses[i].set_data_3d([centre[0], geo["nose"][0]], [centre[1], geo["nose"][1]], [centre[2], geo["nose"][2]])
        scene.view_init(elev=17, azim=-80 + 22 * frame_number / max(len(frames) - 1, 1))
        ended = int((rollout.ended_at >= 0).sum())
        clock.set_text(f"t = {rollout.time_s[t]:5.2f} s" + (f"   ({ended} crashed)" if ended else ""))
        cursor.set_xdata([rollout.time_s[t], rollout.time_s[t]])

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_overview_png(rollout: Rollout, path: Path, tolerance_m: float = 0.1) -> None:
    """A still picture: where each drone went (top and side views, coloured by time)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num = rollout.pos_w.shape[1]
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=90)
    for i in range(num):
        pos = rollout.pos_w[:, i]
        top.scatter(pos[:, 0], pos[:, 1], c=rollout.time_s, s=5, cmap="viridis")
        top.plot(pos[0, 0], pos[0, 1], "ko", ms=5)
        side.scatter(pos[:, 0], pos[:, 2], c=rollout.time_s, s=5, cmap="viridis")
        side.plot(pos[0, 0], pos[0, 2], "ko", ms=5)
    target = rollout.target_w
    for axis, (a, b), label in ((top, (0, 1), "top view (x, y)"), (side, (0, 2), "side view (x, height)")):
        axis.plot(target[a], target[b], "rx", ms=12, mew=2.5)
        axis.add_patch(plt.Circle((target[a], target[b]), tolerance_m, fill=False, color="red"))
        axis.set_aspect("equal")
        axis.set_title(label)
        axis.grid(alpha=0.3)
        axis.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    side.set_ylabel("height (m)")
    side.axhline(0.0, color="0.4", lw=2)
    fig.suptitle("Flight paths from random starts (black dot = start, red x = target; colour = time, 0 to 15 s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Rerun: an interactive 3D viewer
# --------------------------------------------------------------------------

_DRONE_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189), (140, 86, 75), (227, 119, 194),
                 (127, 127, 127), (188, 189, 34)]  # matplotlib's "tab10" blues, oranges, greens, ...


def log_to_rerun(
    rollout: Rollout,
    arm_length: float,
    path: Path,
    scale: float = 5.0,
    spacing: float = 2.4,
    tolerance_m: float = 0.1,
    application_id: str = "quadrace_hover",
) -> None:
    """Write the rollout as a Rerun recording (`.rrd`), to be opened with the Rerun viewer.

    The 3D view shows a ground grid, and for each drone its target and 0.1 m ring, its path so far,
    and the drone itself drawn `scale` times life size (arms, rotor discs, a black nose). The plots
    show the distance to the target. Drag to rotate and zoom; drag the timeline to scrub through the
    flight. The recording goes to a file and is not streamed, so the viewer can be opened any time.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    num = rollout.pos_w.shape[1]
    offsets = display_offsets(num, spacing)
    distance = rollout.distance_to_target

    rr.init(application_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(path))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="hover (drones drawn side by side)", origin="world"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="distance to target (m)", origin="distance"),
                    rrb.TimeSeriesView(name="log10 of the distance (m)", origin="distance_log10"),
                ),
                column_shares=[2, 1],
            ),
        )
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # our world: x forward, y left, z up

    # Everything that does not change: the ground grid, and each drone's target and tolerance ring.
    x_extent = spacing * (num - 1) / 2.0 + 1.7
    grid_lines = [[[g, -1.7, 0.0], [g, 1.7, 0.0]] for g in np.arange(np.ceil(-x_extent), x_extent + 0.01, 1.0)]
    grid_lines += [[[-x_extent, g, 0.0], [x_extent, g, 0.0]] for g in np.arange(-1.0, 1.01, 1.0)]
    rr.log("world/ground", rr.LineStrips3D(grid_lines, colors=[190, 190, 190], radii=0.004), static=True)
    for i in range(num):
        target = rollout.target_w + offsets[i]
        ring = _circle(target, np.eye(3), tolerance_m, points=32)
        rr.log(f"world/drone_{i + 1}/target", rr.Points3D([target], colors=[220, 30, 30], radii=0.05), static=True)
        rr.log(f"world/drone_{i + 1}/tolerance_ring", rr.LineStrips3D([ring], colors=[220, 30, 30], radii=0.006), static=True)
    for i in range(num):
        name = f"drone {i + 1} (start tilt {rollout.start_tilt_deg[i]:.0f} deg)"
        rr.log(f"distance/drone_{i + 1}", rr.SeriesLines(colors=_DRONE_COLORS[i % 8], names=name), static=True)
        rr.log(f"distance_log10/drone_{i + 1}", rr.SeriesLines(colors=_DRONE_COLORS[i % 8], names=name), static=True)
    rr.log("distance/tolerance", rr.SeriesLines(colors=[220, 30, 30], names=f"{tolerance_m} m tolerance"), static=True)
    rr.log("distance_log10/tolerance", rr.SeriesLines(colors=[220, 30, 30], names=f"{tolerance_m} m tolerance"), static=True)

    for frame, time_s in enumerate(rollout.time_s):
        rr.set_time("time", duration=float(time_s))
        for i in range(num):
            color = _DRONE_COLORS[i % 8]
            centre = rollout.pos_w[frame, i] + offsets[i]
            geo = drone_geometry(centre, rollout.quat[frame, i], arm_length, scale)
            motors = geo["motors"]
            rot = ref_sim.quat_to_rotation_matrix(rollout.quat[frame, i])
            base = f"world/drone_{i + 1}"
            rr.log(f"{base}/arms", rr.LineStrips3D([motors[[0, 1]], motors[[2, 3]]], colors=[40, 40, 40], radii=0.014))
            rr.log(f"{base}/rotors", rr.LineStrips3D(
                [_circle(motors[m], rot, 0.45 * scale * arm_length) for m in range(4)], colors=color, radii=0.009))
            rr.log(f"{base}/nose", rr.LineStrips3D([[centre, geo["nose"]]], colors=[0, 0, 0], radii=0.022))
            rr.log(f"{base}/trail", rr.LineStrips3D([rollout.pos_w[: frame + 1, i] + offsets[i]], colors=color, radii=0.006))
            rr.log(f"distance/drone_{i + 1}", rr.Scalars(float(distance[frame, i])))
            rr.log(f"distance_log10/drone_{i + 1}", rr.Scalars(float(np.log10(max(distance[frame, i], 1e-4)))))
        rr.log("distance/tolerance", rr.Scalars(tolerance_m))
        rr.log("distance_log10/tolerance", rr.Scalars(float(np.log10(tolerance_m))))
    rr.disconnect()  # flush and close the file


def rerun_command(*arguments: str) -> list[str]:
    """The command that runs the `rerun` program (the viewer and its tools) from this environment,
    followed by `arguments`."""
    import sys

    exe = Path(sys.executable).parent / ("rerun.exe" if sys.platform == "win32" else "rerun")
    base = [str(exe)] if exe.exists() else [sys.executable, "-m", "rerun"]
    return base + list(arguments)


def open_in_rerun_viewer(path: Path) -> None:
    """Start the Rerun viewer on a recording, as a separate window that keeps running after this
    script ends."""
    import subprocess

    subprocess.Popen(rerun_command(str(path)), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="View a trained hover policy: GIF and PNG, and optionally Rerun")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drone", type=Path, default=REPO_ROOT / "configs" / "drone.yaml")
    parser.add_argument("--task", type=Path, default=REPO_ROOT / "configs" / "task_hover.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "renders" / "hover_demo.gif",
                        help="the GIF to write; the PNG and the .rrd go next to it")
    parser.add_argument("--drones", type=int, default=3, help="how many drones to fly and draw together (1 to 8)")
    parser.add_argument("--seed", type=int, default=2026, help="seed of the random starts")
    parser.add_argument("--scale", type=float, default=5.0, help="how much larger than life the drones are drawn")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2, help="draw every n-th policy step (2 = real time at 25 fps)")
    parser.add_argument("--hardest-of", type=int, default=None, metavar="K",
                        help="fly K random starts and show the --drones that start most tilted (a stress demo)")
    parser.add_argument("--no-gif", action="store_true", help="skip the (slow) GIF and PNG")
    parser.add_argument("--rerun", action="store_true", help="also write a Rerun recording and open the viewer on it")
    parser.add_argument("--no-viewer", action="store_true", help="with --rerun: write the .rrd file but do not open the viewer")
    args = parser.parse_args()
    if not 1 <= args.drones <= 8:
        raise SystemExit("--drones must be between 1 and 8 (this is a viewer, not the training environment)")

    with open(args.drone, encoding="utf-8") as f:
        drone_cfg = yaml.safe_load(f)
    with open(args.task, encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)
    policy, info = ppo.load_checkpoint(args.checkpoint, device="cpu")

    rollout = record_rollout(policy, drone_cfg, task_cfg, num_drones=args.drones, seed=args.seed,
                             hardest_of=args.hardest_of)
    final = rollout.distance_to_target[-1]
    print(f"{args.drones} drones, {rollout.time_s[-1]:.1f} s, seed {args.seed}: "
          f"start tilts {np.array2string(rollout.start_tilt_deg, precision=1)} deg, "
          f"final distance to target {np.array2string(final, precision=4)} m, "
          f"ended early: {int((rollout.ended_at >= 0).sum())}")
    which = (f"the {args.drones} most tilted starts of {args.hardest_of} random ones" if args.hardest_of
             else "random starts, not hand-picked")
    label = f"Hover policy ({args.checkpoint.parent.name}, update {info['update']}), deterministic; {which}"

    if not args.no_gif:
        render_gif(rollout, args.out, drone_cfg["body"]["arm_length"], fps=args.fps, stride=args.stride,
                   scale=args.scale, title=label)
        png = args.out.with_suffix(".png")
        render_overview_png(rollout, png)
        print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) and {png}")
    if args.rerun:
        rrd = args.out.with_suffix(".rrd")
        log_to_rerun(rollout, drone_cfg["body"]["arm_length"], rrd, scale=args.scale)
        print(f"saved the Rerun recording {rrd} ({rrd.stat().st_size / 1e6:.1f} MB); reopen it any time with:  rerun {rrd}")
        if not args.no_viewer:
            open_in_rerun_viewer(rrd)
            print("opened the Rerun viewer")


if __name__ == "__main__":
    main()
