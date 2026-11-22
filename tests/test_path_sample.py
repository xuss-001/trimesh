import numpy as np


def test_resample_original():
    # check to see if `include_original` works

    from shapely.geometry import Polygon

    from trimesh.path.traversal import resample_path

    ori = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)

    re = resample_path(ori, step=0.25, include_original=True)

    a, b = Polygon(ori), Polygon(re)
    assert np.isclose(a.area, b.area)
    assert np.isclose(a.length, b.length)


def test_resample_original_case():
    # check a case that was failing to insert the original points

    from scipy.spatial import KDTree
    from shapely.geometry import LineString

    from trimesh.path.traversal import PathSample

    test_path = np.array([[0, 0], [1, 0], [1.1, 0], [2, 0], [3, 0]], dtype=float)

    ps = PathSample(test_path)
    # note the length of the original path is 3, so this is sampling past the end
    a = ps.sample(np.arange(0, 5, 1.5), include_original=True)

    # get the distance to the closest point from our test path to the sample
    radius, index = KDTree(test_path).query(a)

    # assert sampling is increasing in index
    assert ((index[1:] - index[:-1]) >= 0).all()

    # the sample path includes points that didn't exist in the original
    # so we need to check that every index has at least one zero-radius
    radius_ok = np.zeros(len(test_path), dtype=bool)
    radius_ok[index[radius < 1e-12]] = True
    assert radius_ok.all(), "not every point was inserted!"

    # run a few more quick checks to make sure the path wasn't trashed
    a = LineString(test_path)
    b = LineString(a)

    assert np.isclose(a.length, b.length)
    assert np.allclose(a.bounds, b.bounds)


if __name__ == "__main__":
    test_resample_original()
    test_resample_original_case()


def test_sample_include_original_keeps_length():
    from trimesh.path.traversal import PathSample

    # segments of length 1, 2 and 3 so the path is 6 long
    path = np.array([[0, 0], [1, 0], [1, 2], [4, 2]], dtype=np.float64)
    sampler = PathSample(path)
    for distances in ([0.0, 2.0], [2.0], [1.0, 3.0, 6.0]):
        result = sampler.sample(np.array(distances), include_original=True)
        length = np.linalg.norm(np.diff(result, axis=0), axis=1).sum()
        assert np.isclose(length, 6.0)

    # every original vertex is kept and the new sample is in order
    result = sampler.sample(np.array([0.0, 2.0]), include_original=True)
    assert np.allclose(result, [[0, 0], [1, 0], [1, 1], [1, 2], [4, 2]])


def _mixed_contour():
    """
    Open contour: line, half-circle arc, line.
    Total analytic length is ``4 + pi``.
    """
    from trimesh.path.entities import Arc, Line

    vertices = np.array(
        [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0], [6.0, 0.0]]
    )
    entities = [
        Line(points=[0, 1]),
        Arc(points=[1, 2, 3]),
        Line(points=[3, 4]),
    ]
    return entities, vertices


def _global_distances(sampler, entity_ids, local):
    # reconstruct global arc-length from the per-entity local values
    position = {
        int(entity_id): index
        for index, entity_id in enumerate(sampler._entity_ids)
    }
    return np.array(
        [
            sampler._bounds[position[int(entity_id)]] + distance
            for entity_id, distance in zip(entity_ids, local)
        ]
    )


def test_curve_sample_mixed_open_contour():
    from trimesh.path.entities import Arc
    from trimesh.path.traversal import CurveSample

    entities, vertices = _mixed_contour()
    sampler = CurveSample(entities, vertices, [0, 1, 2])

    # exact analytic length: two unit lines plus a half circle of radius 1
    assert np.isclose(sampler.length, 4.0 + np.pi)

    points, entity_ids, local = sampler.sample(0.3)
    distances = _global_distances(sampler, entity_ids, local)

    # constant spacing except for the short final segment
    assert np.allclose(np.diff(distances)[:-1], 0.3)
    assert np.diff(distances)[-1] <= 0.3 + 1e-12

    # open contour keeps both endpoints
    assert np.allclose(points[0], [0.0, 0.0])
    assert np.allclose(points[-1], [6.0, 0.0])
    assert entity_ids[-1] == 2
    assert np.isclose(local[-1], 2.0)

    # arc samples stay on the analytic circle, not a coarse polyline
    arc = entities[1]
    fit = arc.center(vertices)
    radii = np.linalg.norm(points[entity_ids == 1] - fit.center, axis=1)
    assert np.allclose(radii, fit.radius)
    assert np.all(local[entity_ids == 1] >= 0.0)
    assert np.all(local[entity_ids == 1] <= arc.length(vertices) + 1e-12)


def test_curve_sample_junction_belongs_to_later_entity():
    from trimesh.path.traversal import CurveSample

    entities, vertices = _mixed_contour()
    sampler = CurveSample(entities, vertices, [0, 1, 2])

    # step equal to the first line length lands on the line/arc junction
    points, entity_ids, local = sampler.sample(2.0)
    junction = np.argmin(np.linalg.norm(points - [2.0, 0.0], axis=1))
    assert np.allclose(points[junction], [2.0, 0.0])
    assert entity_ids[junction] == 1
    assert np.isclose(local[junction], 0.0)


def test_curve_sample_reversed_traversal():
    from trimesh.path.traversal import CurveSample

    entities, vertices = _mixed_contour()
    sampler = CurveSample(entities, vertices, [2, 1, 0])

    points, entity_ids, local = sampler.sample(0.3)
    distances = _global_distances(sampler, entity_ids, local)
    assert np.allclose(np.diff(distances)[:-1], 0.3)

    # traversal now runs from the last line back to the first line
    assert np.allclose(points[0], [6.0, 0.0])
    assert np.allclose(points[-1], [0.0, 0.0])

    # the reversed arc still follows the same circle and its local
    # arc-length increases from the new traversal start
    fit = entities[1].center(vertices)
    arc_points = points[entity_ids == 1]
    radii = np.linalg.norm(arc_points - fit.center, axis=1)
    assert np.allclose(radii, fit.radius)
    assert np.all(np.diff(local[entity_ids == 1]) > 0.0)

    # junction at global distance 2 belongs to the arc, the entity
    # encountered second in the reversed traversal
    junction_points, junction_ids, junction_local = sampler.sample(2.0)
    hit = np.argmin(np.linalg.norm(junction_points - [4.0, 0.0], axis=1))
    assert junction_ids[hit] == 1
    assert np.isclose(junction_local[hit], 0.0)


def test_curve_sample_small_arc():
    from trimesh.path.entities import Arc
    from trimesh.path.traversal import CurveSample

    # tiny arc of radius 1 spanning 0.02 radians
    vertices = np.array(
        [
            [1.0, 0.0],
            [np.cos(0.01), np.sin(0.01)],
            [np.cos(0.02), np.sin(0.02)],
        ]
    )
    arc = Arc(points=[0, 1, 2])
    sampler = CurveSample([arc], vertices, [0])

    assert np.isclose(sampler.length, 0.02, rtol=1e-9)
    points, entity_ids, local = sampler.sample(0.005)

    # five evenly spaced samples including the exact end point
    assert len(points) == 5
    assert np.allclose(np.diff(local), 0.005)
    assert np.allclose(points[-1], vertices[-1], atol=1e-12)
    assert np.isclose(local[-1], sampler.length)
    assert (entity_ids == 0).all()


def test_curve_sample_closed_circle():
    from trimesh.path.entities import Arc
    from trimesh.path.traversal import CurveSample

    vertices = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    circle = Arc(points=[0, 1, 2], closed=True)
    sampler = CurveSample([circle], vertices, [0], closed=True)

    points, entity_ids, local = sampler.sample(np.pi / 2.0)

    # closed contour does not repeat the start point
    assert len(points) == 4
    assert not np.allclose(points[-1], points[0])
    assert np.allclose(points[0], vertices[0])
    assert np.allclose(np.diff(local), np.pi / 2.0)
    assert (entity_ids == 0).all()

    # a step larger than the circumference keeps a single start sample
    single, _, _ = sampler.sample(100.0)
    assert len(single) == 1
    assert np.allclose(single[0], vertices[0])


def test_curve_sample_closed_mixed_racetrack():
    from trimesh.path.entities import Arc, Line
    from trimesh.path.traversal import CurveSample

    # two semicircles of radius 1 joined by two length-2 lines
    vertices = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, -2.0],
            [0.0, -3.0],
            [-1.0, -2.0],
        ]
    )
    entities = [
        Arc(points=[0, 1, 2]),
        Line(points=[2, 3]),
        Arc(points=[3, 4, 5]),
        Line(points=[5, 0]),
    ]
    sampler = CurveSample(entities, vertices, [0, 1, 2, 3], closed=True)
    assert np.isclose(sampler.length, 2.0 * np.pi + 4.0)

    points, entity_ids, local = sampler.sample(0.5)
    distances = _global_distances(sampler, entity_ids, local)
    assert np.allclose(np.diff(distances), 0.5)
    assert len(points) == int(np.floor(sampler.length / 0.5))
    assert not np.allclose(points[-1], points[0])

    # the arc/line junction at global distance pi belongs to the line
    junction_points, junction_ids, junction_local = sampler.sample(np.pi)
    hit = np.argmin(
        np.linalg.norm(junction_points - [1.0, 0.0], axis=1)
    )
    assert junction_ids[hit] == 1
    assert np.isclose(junction_local[hit], 0.0)

    # reversing a closed contour preserves count and analytic geometry
    reversed_sampler = CurveSample(
        entities, vertices, [3, 2, 1, 0], closed=True
    )
    reversed_points, reversed_ids, _ = reversed_sampler.sample(0.5)
    assert reversed_points.shape == points.shape
    for entity_id in np.unique(reversed_ids):
        entity = entities[int(entity_id)]
        if isinstance(entity, Arc):
            fit = entity.center(vertices)
            chosen = reversed_points[reversed_ids == entity_id]
            radii = np.linalg.norm(chosen - fit.center, axis=1)
            assert np.allclose(radii, fit.radius)


def test_curve_sample_invalid_step():
    from trimesh.path.traversal import CurveSample

    entities, vertices = _mixed_contour()
    sampler = CurveSample(entities, vertices, [0, 1, 2])

    for invalid in (0, -1.0, np.nan, np.inf, -np.inf):
        with np.testing.assert_raises(ValueError):
            sampler.sample(invalid)


def test_curve_sample_does_not_mutate_input():
    from trimesh.path.traversal import CurveSample

    entities, vertices = _mixed_contour()
    points_before = [entity.points.copy() for entity in entities]
    vertices_before = vertices.copy()

    CurveSample(entities, vertices, [2, 1, 0]).sample(0.7)

    for before, entity in zip(points_before, entities):
        assert np.array_equal(before, entity.points)
    assert np.array_equal(vertices_before, vertices)
    assert not any(hasattr(entity, "_direction") for entity in entities)


def test_sample_arc_length_function():
    from trimesh.path.traversal import sample_arc_length

    entities, vertices = _mixed_contour()
    points, entity_ids, local = sample_arc_length(
        entities, vertices, [0, 1, 2], step=1.0
    )
    assert points.shape[0] == entity_ids.shape[0] == local.shape[0]
    assert np.allclose(points[0], [0.0, 0.0])
    assert np.allclose(points[-1], [6.0, 0.0])
