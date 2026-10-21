"""Tests for the world package: geometry, and that sampled scenes respect their rules."""

import numpy as np
import pytest

from hummingbird_rl.world.collision import clearance, collides, nearest_obstacles, segment_point_distance
from hummingbird_rl.world.sampling import ALTITUDE_RANGE, path_exists, sample_scene
from hummingbird_rl.world.scene import START, Obstacle, Scene

R_BIRD = 0.02
N_SCENES = 30


def test_scene_round_trips_through_dict():
    scene = Scene(targets=[np.array([0.1, 0.6, 0.2])], obstacles=[Obstacle(np.array([0.3, 0.5, 0.0]), 0.05)])
    back = Scene.from_dict(scene.to_dict())
    assert np.allclose(back.start, scene.start)
    assert np.allclose(back.targets[0], scene.targets[0])
    assert np.allclose(back.obstacles[0].center, scene.obstacles[0].center)
    assert back.obstacles[0].radius == scene.obstacles[0].radius


def test_clearance_and_collision():
    o = Obstacle(np.array([1.0, 0.0, 0.0]), 0.1)
    assert clearance(np.zeros(3), o, R_BIRD) == pytest.approx(1.0 - 0.1 - R_BIRD)
    assert not collides(np.array([0.87, 0.0, 0.0]), [o], R_BIRD)      # 1 cm gap
    assert collides(np.array([0.89, 0.0, 0.0]), [o], R_BIRD)          # 1 cm overlap


def test_nearest_obstacles_sorted_by_clearance_and_limited_in_range():
    pos = np.zeros(3)
    big_far = Obstacle(np.array([0.5, 0.0, 0.0]), 0.3)        # centre farther, but surface closer
    small_near = Obstacle(np.array([0.3, 0.0, 0.0]), 0.02)
    out_of_range = Obstacle(np.array([3.0, 0.0, 0.0]), 0.05)
    seen = nearest_obstacles(pos, [small_near, out_of_range, big_far], R_BIRD, k=5, max_range=1.0)
    assert seen == [big_far, small_near]


def test_segment_point_distance():
    a, b = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert segment_point_distance(a, b, np.array([0.5, 0.2, 0.0])) == pytest.approx(0.2)
    assert segment_point_distance(a, b, np.array([2.0, 0.0, 0.0])) == pytest.approx(1.0)   # beyond the end


def test_path_exists_open_space_and_around_one_sphere():
    goal = START + np.array([0.6, 0.0, 0.0])
    assert path_exists([START, goal], [], R_BIRD)
    assert path_exists([START, goal], [Obstacle(START + np.array([0.3, 0.0, 0.0]), 0.1)], R_BIRD)


def test_path_exists_detects_a_wall_and_a_buried_target():
    goal = START + np.array([0.6, 0.0, 0.0])
    # A wall of spheres across the whole search box at x = 0.3, from the ground to the ceiling.
    wall = [Obstacle(np.array([0.3, y, z]), 0.06)
            for y in np.arange(0.0, 1.25, 0.05) for z in np.arange(-0.4, 0.45, 0.05)]
    assert not path_exists([START, goal], wall, R_BIRD)
    assert not path_exists([START, goal], [Obstacle(goal, 0.05)], R_BIRD)


def test_hover_scene_target_is_the_start():
    scene = sample_scene("hover", np.random.default_rng(0))
    assert np.allclose(scene.targets[0], START) and not scene.obstacles


def test_goto_scenes():
    rng = np.random.default_rng(1)
    behind = 0
    for _ in range(N_SCENES):
        target = sample_scene("goto", rng).targets[0]
        assert 0.10 <= np.linalg.norm(target - START) <= 0.50
        assert ALTITUDE_RANGE[0] <= target[1] <= ALTITUDE_RANGE[1]
        behind += target[0] < START[0]
    assert 5 < behind < N_SCENES - 5       # directions really are spread around the bird


def test_obstacle_scenes():
    rng = np.random.default_rng(2)
    for _ in range(N_SCENES):
        scene = sample_scene("obstacles", rng)
        goal = scene.targets[0]
        assert 0.50 <= np.linalg.norm(goal - START) <= 0.80 + 1e-9
        assert 3 <= len(scene.obstacles) <= 6
        assert any(segment_point_distance(START, goal, o.center) < o.radius + R_BIRD for o in scene.obstacles)
        for o in scene.obstacles:
            assert clearance(START, o, 0.0) >= 0.10 - 1e-9 and clearance(goal, o, 0.0) >= 0.10 - 1e-9
        assert path_exists([START, goal], scene.obstacles, R_BIRD)


def test_navigation_scenes():
    rng = np.random.default_rng(3)
    for _ in range(N_SCENES // 3):
        scene = sample_scene("navigation", rng)
        course = [START] + scene.targets
        assert len(scene.targets) == 3
        for a, b in zip(course[:-1], course[1:]):
            assert 0.25 - 1e-9 <= np.linalg.norm(b - a) <= 0.50 + 1e-9
        assert all(np.hypot(w[0], w[2]) <= 1.0 for w in scene.targets)
        assert 3 <= len(scene.obstacles) <= 6
        assert not any(collides(p, scene.obstacles, 0.0) for p in course)


def test_sampling_is_reproducible():
    a = sample_scene("navigation", np.random.default_rng(7)).to_dict()
    b = sample_scene("navigation", np.random.default_rng(7)).to_dict()
    assert a == b


def test_number_of_obstacles_can_be_changed():
    rng = np.random.default_rng(4)
    for task in ("obstacles", "navigation"):
        counts = {len(sample_scene(task, rng, n_obstacles=(2, 3)).obstacles) for _ in range(10)}
        assert counts <= {2, 3}
