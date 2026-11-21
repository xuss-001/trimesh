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


def _mixed_path(reverse=False):
    # a unit line from the origin followed by a quarter
    # circle of radius one centered at (1, 1)
    from trimesh.path.path import Path2D
    from trimesh.path.entities import Arc, Line

    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0 + 2**0.5 / 2, 1.0 - 2**0.5 / 2], [2.0, 1.0]]
    )
    if reverse:
        # store the entities opposite to the traversal direction
        entities = [Line([1, 0]), Arc([3, 2, 1])]
        for entity in entities:
            entity.reverse()
    else:
        entities = [Line([0, 1]), Arc([1, 2, 3])]
    return Path2D(entities=entities, vertices=vertices)


def test_sample_at_interval_mixed():
    # mixed line and arc sampled at exact arc-length intervals
    path = _mixed_path()

    # exact lengths: unit line and quarter circle of radius one
    lengths = np.array([1.0, np.pi / 2])
    assert np.allclose([e.length(path.vertices) for e in path.entities], lengths)

    step = 0.5
    points, entity_index, distance = path.sample_at_interval(step)

    # global arc length of every sample from the start of the path
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    global_distance = distance + cumulative[entity_index]
    # samples are evenly spaced and the open end point is kept
    expected = np.append(np.arange(0.0, lengths.sum(), step), lengths.sum())
    assert np.allclose(global_distance, expected)
    # the final segment is allowed to be shorter than the step
    assert 0.0 < np.diff(global_distance)[-1] < step

    # the sample exactly on the junction belongs to the later entity
    (junction,) = np.nonzero(np.isclose(global_distance, 1.0))
    assert len(junction) == 1
    assert entity_index[junction[0]] == 1
    assert np.isclose(distance[junction[0]], 0.0)
    assert np.allclose(points[junction[0]], [1.0, 0.0])

    # the end point of the open path belongs to the last entity
    assert entity_index[-1] == 1
    assert np.isclose(distance[-1], lengths[1])
    assert np.allclose(points[-1], [2.0, 1.0])

    # line samples are on the line and arc samples on the circle
    on_line = entity_index == 0
    assert np.allclose(points[on_line][:, 0], distance[on_line])
    assert np.allclose(points[on_line][:, 1], 0.0)
    radius = np.linalg.norm(points[~on_line] - [1.0, 1.0], axis=1)
    assert np.allclose(radius, 1.0)

    # the input entities were not modified
    assert path.entities[0].points.tolist() == [0, 1]
    assert path.entities[1].points.tolist() == [1, 2, 3]


def test_sample_at_interval_reversed():
    # entities stored against the traversal direction and
    # reversed in place must sample identically
    forward = _mixed_path()
    reversed_path = _mixed_path(reverse=True)

    step = 0.3
    result = forward.sample_at_interval(step)
    result_reversed = reversed_path.sample_at_interval(step)

    for a, b in zip(result, result_reversed):
        assert np.allclose(a, b)

    # the reversed line runs from the origin to (1, 0)
    points, entity_index, distance = result_reversed
    on_line = entity_index == 0
    assert np.allclose(points[on_line][:, 0], distance[on_line])
    assert np.allclose(points[on_line][:, 1], 0.0)


def test_sample_at_interval_small_arc():
    # a tiny half circle where a coarse polyline approximation
    # of the length would be very inaccurate
    from trimesh.path.path import Path2D
    from trimesh.path.entities import Arc

    radius = 1e-3
    vertices = np.array([[radius, 0.0], [0.0, radius], [-radius, 0.0]])
    path = Path2D(entities=[Arc([0, 1, 2])], vertices=vertices)

    total = np.pi * radius
    assert np.isclose(path.entities[0].length(path.vertices), total)

    step = total / 10.0
    points, entity_index, distance = path.sample_at_interval(step)

    # single entity keeps both ends of the open arc
    assert (entity_index == 0).all()
    assert np.isclose(distance[-1], total)
    assert np.allclose(points[0], [radius, 0.0])
    assert np.allclose(points[-1], [-radius, 0.0])
    # every sample is on the exact circle at arc-length spacing
    assert np.allclose(np.linalg.norm(points, axis=1), radius)
    assert np.allclose(np.diff(distance), step)


def test_sample_at_interval_closed():
    # a closed loop of a half circle and a diameter line
    from trimesh.path.path import Path2D
    from trimesh.path.entities import Arc, Line

    vertices = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    entities = [Arc([0, 1, 2]), Line([2, 0])]
    path = Path2D(entities=entities, vertices=vertices)

    lengths = np.array([np.pi, 2.0])
    total = lengths.sum()
    step = 0.7
    points, entity_index, distance = path.sample_at_interval(step)

    # the start point is not repeated at the end of a closed path
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    global_distance = distance + cumulative[entity_index]
    expected = np.arange(0.0, total, step)
    assert np.allclose(global_distance, expected)
    assert global_distance[-1] < total
    assert not np.allclose(points[-1], points[0])
    # the gap wrapping back to the start is shorter than the step
    assert 0.0 < total - global_distance[-1] < step

    # a closed circle whose length is an exact multiple of the
    # step must not duplicate the start point either
    circle = Path2D(
        entities=[Arc([0, 1, 2], closed=True)],
        vertices=np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]),
    )
    points, entity_index, distance = circle.sample_at_interval(np.pi / 2)
    assert len(points) == 8
    assert (entity_index == 0).all()
    assert np.allclose(np.diff(distance), np.pi / 2)
    assert np.allclose(np.linalg.norm(points, axis=1), 2.0)
    assert np.allclose(points[0], [2.0, 0.0])


def test_sample_at_interval_open_ends():
    # an open path shorter than the step keeps both end points
    from trimesh.path.path import Path2D
    from trimesh.path.entities import Line

    path = Path2D(
        entities=[Line([0, 1])], vertices=np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    points, entity_index, distance = path.sample_at_interval(5.0)
    assert np.allclose(points, [[0.0, 0.0], [1.0, 0.0]])
    assert (entity_index == 0).all()
    assert np.allclose(distance, [0.0, 1.0])


def test_sample_at_interval_invalid():
    # non-positive or non-finite steps and unsupported entities
    import pytest

    from trimesh.path.path import Path2D
    from trimesh.path.entities import Bezier, Line

    path = _mixed_path()
    for bad in [0.0, -1.0, np.inf, -np.inf, np.nan]:
        with pytest.raises(ValueError):
            path.sample_at_interval(bad)

    spline = Path2D(
        entities=[Line([0, 1]), Bezier([1, 2, 3])],
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]),
    )
    with pytest.raises(ValueError):
        spline.sample_at_interval(0.5)
