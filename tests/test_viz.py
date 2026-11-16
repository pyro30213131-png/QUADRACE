"""Tests for the viewer (python/quadrace/viz.py): the drawing geometry, the rollout recorder, and
that the GIF and PNG are really written. The viewer's pictures are judged by eye, so these tests
check what can be checked: the points it draws, what it records, and the files it produces.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import CrashPolicy, PDHover

from quadrace import viz

REPO_ROOT = Path(__file__).resolve().parent.parent
ARM = 0.15


@pytest.fixture(scope="module")
def drone_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "drone.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def task_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "task_hover.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# Where the drone is drawn
# --------------------------------------------------------------------------


def test_a_level_drone_has_its_motors_at_the_corners_of_the_x_from_the_handoff_table() -> None:
    d = ARM / np.sqrt(2.0)
    geo = viz.drone_geometry(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), ARM)
    expected = np.array([[+d, -d, 0.0], [-d, +d, 0.0], [+d, +d, 0.0], [-d, -d, 0.0]])  # motors 0 to 3, section 6
    np.testing.assert_allclose(geo["motors"], expected, atol=1e-12)


def test_the_nose_is_ahead_along_body_x_and_up_is_above_along_body_z() -> None:
    geo = viz.drone_geometry(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]), ARM)
    np.testing.assert_allclose(geo["nose"], [1.0 + 1.6 * ARM, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(geo["up"], [1.0, 2.0, 3.0 + 0.8 * ARM], atol=1e-12)


def test_a_drone_yawed_90_degrees_has_its_nose_along_world_y_and_motors_rotated_with_it() -> None:
    half = np.sqrt(0.5)
    d = ARM / np.sqrt(2.0)
    geo = viz.drone_geometry(np.zeros(3), np.array([half, 0.0, 0.0, half]), ARM)
    np.testing.assert_allclose(geo["nose"], [0.0, 1.6 * ARM, 0.0], atol=1e-12)
    np.testing.assert_allclose(geo["motors"][0], [d, d, 0.0], atol=1e-12)  # motor 0 was at (+d, -d)


def test_a_drone_rolled_90_degrees_has_its_up_point_sideways() -> None:
    half = np.sqrt(0.5)
    geo = viz.drone_geometry(np.zeros(3), np.array([half, half, 0.0, 0.0]), ARM)  # 90 degrees about x
    np.testing.assert_allclose(geo["up"], [0.0, -0.8 * ARM, 0.0], atol=1e-12)


def test_the_scale_enlarges_the_drone_around_its_position_without_moving_the_position() -> None:
    pos = np.array([1.0, -1.0, 2.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    small = viz.drone_geometry(pos, quat, ARM, scale=1.0)
    big = viz.drone_geometry(pos, quat, ARM, scale=5.0)
    np.testing.assert_allclose(big["motors"] - pos, 5.0 * (small["motors"] - pos), atol=1e-12)
    np.testing.assert_allclose(big["motors"].mean(axis=0), pos, atol=1e-12)  # still centred on the drone


def test_drones_are_shifted_sideways_so_they_do_not_overlap_and_the_set_is_centred() -> None:
    offsets = viz.display_offsets(3, spacing=2.4)
    np.testing.assert_allclose(offsets[:, 0], [-2.4, 0.0, 2.4])
    np.testing.assert_array_equal(offsets[:, 1:], 0.0)
    np.testing.assert_allclose(viz.display_offsets(1)[:, 0], [0.0])
    np.testing.assert_allclose(viz.display_offsets(4, spacing=1.0)[:, 0].mean(), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Recording a rollout
# --------------------------------------------------------------------------


def test_a_recorded_rollout_has_one_frame_per_policy_step_for_the_episode_length(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=3, seed=1)
    assert rollout.pos_w.shape == (750, 3, 3) and rollout.quat.shape == (750, 3, 4) and rollout.thrust.shape == (750, 3, 4)
    assert rollout.time_s[0] == 0.0 and rollout.time_s[-1] == pytest.approx(749 / 50)
    np.testing.assert_allclose(np.linalg.norm(rollout.quat, axis=2), 1.0, atol=1e-9)
    np.testing.assert_array_equal(rollout.target_w, [0.0, 0.0, 2.0])


def test_a_rollout_can_be_shorter_than_the_episode(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=2, seconds=2.0)
    assert rollout.pos_w.shape[0] == 100


def test_the_first_frame_is_a_random_start_within_the_reset_range_of_the_task(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=6, seed=5, seconds=1.0)
    start_distance = np.linalg.norm(rollout.pos_w[0] - rollout.target_w, axis=1)
    assert (start_distance <= task_cfg["reset"]["max_offset"] + 1e-9).all() and start_distance.min() > 0.05
    assert len({tuple(p) for p in rollout.pos_w[0]}) == 6  # six different starts


def test_a_competent_controller_ends_at_the_target_in_the_recording_and_nobody_ends_early(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=3, seed=2)
    assert (rollout.ended_at == -1).all()
    assert rollout.distance_to_target[-1].max() < 0.01
    assert rollout.distance_to_target[0].min() > 0.05  # they did not start there


def test_a_drone_that_crashes_is_marked_and_frozen_where_it_was_instead_of_showing_the_restart(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(CrashPolicy(), drone_cfg, task_cfg, num_drones=3, seed=3)
    assert (rollout.ended_at > 0).all()
    for i in range(3):
        frame = rollout.ended_at[i]
        np.testing.assert_array_equal(rollout.pos_w[frame:, i], np.tile(rollout.pos_w[frame - 1, i], (750 - frame, 1)))
        assert rollout.pos_w[frame - 1, i, 2] < 0.3  # it had fallen almost to the ground, not been reset to 0.5+ m up


def test_the_same_seed_records_the_same_flight_and_another_seed_a_different_one(drone_cfg: dict, task_cfg: dict) -> None:
    policy = PDHover(drone_cfg)
    a = viz.record_rollout(policy, drone_cfg, task_cfg, num_drones=2, seed=7, seconds=1.0)
    b = viz.record_rollout(policy, drone_cfg, task_cfg, num_drones=2, seed=7, seconds=1.0)
    c = viz.record_rollout(policy, drone_cfg, task_cfg, num_drones=2, seed=8, seconds=1.0)
    np.testing.assert_array_equal(a.pos_w, b.pos_w)
    assert not np.array_equal(a.pos_w, c.pos_w)


def test_the_start_tilt_is_the_angle_of_the_body_z_axis_from_vertical() -> None:
    half = np.sqrt(0.5)
    quat = np.zeros((2, 3, 4))
    quat[:, 0] = [1.0, 0.0, 0.0, 0.0]  # level
    quat[:, 1] = [half, half, 0.0, 0.0]  # 90 degrees about x
    quat[:, 2] = [np.cos(0.25), 0.0, np.sin(0.25), 0.0]  # 0.5 rad about y
    rollout = viz.Rollout(np.zeros(2), np.zeros((2, 3, 3)), quat, np.zeros((2, 3, 4)), np.zeros(3), np.full(3, -1))
    np.testing.assert_allclose(rollout.start_tilt_deg, [0.0, 90.0, np.degrees(0.5)], atol=1e-9)


def test_selecting_drones_keeps_only_those_in_the_order_given() -> None:
    n = 4
    rollout = viz.Rollout(np.arange(3.0), np.arange(3 * n * 3.0).reshape(3, n, 3), np.zeros((3, n, 4)),
                          np.zeros((3, n, 4)), np.zeros(3), np.arange(n) - 1)
    chosen = viz.select_drones(rollout, np.array([2, 0]))
    np.testing.assert_array_equal(chosen.pos_w[:, 0], rollout.pos_w[:, 2])
    np.testing.assert_array_equal(chosen.pos_w[:, 1], rollout.pos_w[:, 0])
    np.testing.assert_array_equal(chosen.ended_at, [1, -1])
    assert chosen.pos_w.shape == (3, 2, 3)


def test_hardest_of_keeps_the_most_tilted_starts_most_tilted_first_and_they_are_real_random_starts(
    drone_cfg: dict, task_cfg: dict
) -> None:
    policy = PDHover(drone_cfg)
    pool = viz.record_rollout(policy, drone_cfg, task_cfg, num_drones=32, seed=9, seconds=0.5)
    hardest = viz.record_rollout(policy, drone_cfg, task_cfg, num_drones=3, seed=9, seconds=0.5, hardest_of=32)
    top3 = np.sort(pool.start_tilt_deg)[::-1][:3]
    np.testing.assert_allclose(hardest.start_tilt_deg, top3, atol=1e-9)  # the 3 largest of the 32, in order
    assert hardest.start_tilt_deg.min() > np.median(pool.start_tilt_deg)
    assert hardest.start_tilt_deg.max() <= np.degrees(task_cfg["reset"]["max_tilt"]) + 1e-6  # within the task's range
    for i, drone in enumerate(hardest.pos_w[0]):  # each kept drone is one of the pool's drones
        assert any(np.array_equal(drone, other) for other in pool.pos_w[0]), i


def test_the_distance_to_target_is_the_distance_between_position_and_target(drone_cfg: dict, task_cfg: dict) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=2, seed=4, seconds=1.0)
    expected = np.linalg.norm(rollout.pos_w - np.array([0.0, 0.0, 2.0]), axis=2)
    np.testing.assert_allclose(rollout.distance_to_target, expected)


# --------------------------------------------------------------------------
# The files it writes
# --------------------------------------------------------------------------


def test_the_gif_is_written_with_the_expected_number_of_frames(drone_cfg: dict, task_cfg: dict, tmp_path: Path) -> None:
    from PIL import Image

    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=2, seed=1, seconds=1.0)  # 50 frames
    path = tmp_path / "out" / "demo.gif"  # the folder does not exist yet: the viewer must create it
    viz.render_gif(rollout, path, ARM, fps=10, stride=10, dpi=30)
    assert path.read_bytes()[:6] == b"GIF89a"
    with Image.open(path) as image:
        assert image.n_frames == 5  # 50 frames, every 10th
        assert image.size[0] > image.size[1] > 100


def test_the_overview_png_is_written(drone_cfg: dict, task_cfg: dict, tmp_path: Path) -> None:
    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=2, seed=1, seconds=1.0)
    path = tmp_path / "overview.png"
    viz.render_overview_png(rollout, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_rerun_recording_is_a_valid_file_that_contains_what_the_viewer_draws(
    drone_cfg: dict, task_cfg: dict, tmp_path: Path
) -> None:
    import subprocess

    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=2, seed=1, seconds=1.0)
    path = tmp_path / "out" / "demo.rrd"  # the folder does not exist yet: the viewer code must create it
    viz.log_to_rerun(rollout, ARM, path)

    assert path.stat().st_size > 50_000
    verified = subprocess.run(viz.rerun_command("rrd", "verify", str(path)), capture_output=True, text=True)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    listing = subprocess.run(viz.rerun_command("rrd", "print", str(path)), capture_output=True, text=True).stdout
    for entity in ("/world/ground", "/world/drone_1/arms", "/world/drone_1/rotors", "/world/drone_1/nose",
                   "/world/drone_1/trail", "/world/drone_2/trail", "/world/drone_1/target",
                   "/world/drone_2/tolerance_ring", "/distance/drone_1", "/distance/drone_2",
                   "/distance/tolerance", "/distance_log10/drone_1"):
        assert entity in listing, entity


def test_the_rerun_recording_has_one_logged_step_per_frame(drone_cfg: dict, task_cfg: dict, tmp_path: Path) -> None:
    import re
    import subprocess

    rollout = viz.record_rollout(PDHover(drone_cfg), drone_cfg, task_cfg, num_drones=1, seed=1, seconds=1.0)  # 50 frames
    path = tmp_path / "one.rrd"
    viz.log_to_rerun(rollout, ARM, path)
    listing = subprocess.run(viz.rerun_command("rrd", "print", str(path)), capture_output=True, text=True).stdout

    def rows(entity: str, data_column: str) -> list[int]:
        pattern = rf"with (\d+) rows .* - {re.escape(entity)} - data columns: \[{data_column}"
        return [int(n) for n in re.findall(pattern, listing)]

    # Entities that change over time hold one row per frame; the static ones (ground, target) one row.
    assert rows("/distance/drone_1", "Scalars") == [50]
    assert rows("/world/drone_1/trail", "LineStrips3D") == [50]
    assert rows("/world/drone_1/rotors", "LineStrips3D") == [50]
    assert rows("/world/drone_1/target", "Points3D") == [1]
    assert rows("/world/ground", "LineStrips3D") == [1]


def test_asking_for_too_many_drones_is_refused_because_this_is_a_viewer_not_the_training_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["viz", "--checkpoint", "nowhere.pt", "--drones", "9"])
    with pytest.raises(SystemExit, match="between 1 and 8"):
        viz.main()
