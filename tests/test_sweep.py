import numpy as np
import pytest
from pyinstrument import Profiler
from shapely.geometry import Point, Polygon

import trimesh
from trimesh.creation import sweep_polygon

trimesh.tol.strict = True


def arc_2d(final_angle, d=1, start_angle=0, splits=20):
    xs = np.linspace(start_angle, final_angle, splits)
    return np.stack((d / 2 * np.cos(xs), d / 2 * np.sin(xs)), axis=1)


def test_simple_closed(h1=1, w=2, r=4.4):
    # a simple closed path
    square = Polygon([(-0.01, -h1 / 2), (w, -h1 / 2), (w, +h1 / 2), (-0.01, +h1 / 2)])
    circle = Point([0, 0]).buffer(1.0)

    # a simple square of side `r`
    path = [[0, 0, 0], [r, 0, 0], [r, r, 0], [0, r, 0], [0, 0, 0]]

    a = sweep_polygon(square, path)
    assert a.is_volume

    aa = sweep_polygon(square, path, connect=False)
    assert aa.is_volume

    # should have the same bounds but no longer a volume
    aaa = sweep_polygon(square, path, connect=False, cap=False)
    assert np.allclose(aa.bounds, aaa.bounds)
    assert not aaa.is_volume

    b = sweep_polygon(circle, path)
    assert b.is_volume


def test_simple_extrude(height=10):
    # make sure sweep behaves okay on a simple single segment path
    # this should be identical to the extrude operation
    circle = Point([0, 0]).buffer(1.0)

    # a simple extrusion
    path = [[0, 0, 0], [0, 0, height]]
    # will be running asserts inside function
    b = sweep_polygon(circle, path)

    # should be a straight extrude along Z
    expected = np.append(np.ptp(np.reshape(circle.bounds, (2, 2)), axis=0), height)
    assert np.allclose(expected, b.extents)

    # should be well constructed
    assert b.is_volume
    # volume should correspond to expected cylinder area
    assert np.isclose(b.volume, circle.area * height)


def test_simple_open():
    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


def test_spline_3D():
    circle = Point([0, 0]).buffer(1.0)
    # make a 3D noodle using spline smoothing
    path = trimesh.path.simplify.resample_spline(
        [[0, 0, 0], [4, 4, 0], [5, 5, 0], [10, 0, 10], [0, 20, 0]], smooth=0.25, count=100
    )

    a = sweep_polygon(circle, path)
    assert a.is_volume
    assert a.body_count == 1


def test_screw():
    h = 200
    d = 8
    h1 = 1
    w = 2
    d = d - w
    h2 = h1

    lead = 2
    splits_per_turn = 4 * 20
    polygon = np.array([(-0.01, -h1 / 2), (w, -h2 / 2), (w, +h2 / 2), (-0.01, +h1 / 2)])
    h_ = h + h1
    n = int(np.ceil(splits_per_turn * h_ / lead))
    turns = n / splits_per_turn
    xys = arc_2d(2 * np.pi * turns, d, splits=n)
    zs = np.linspace(0, h_, n)[:, None]
    spin_path = np.concatenate((xys, zs), axis=1)

    with Profiler() as P:
        x = sweep_polygon(Polygon(polygon), spin_path)
    P.print()

    assert x.is_volume
    # should have produced a result corresponding to the input dimensions
    assert np.allclose(x.extents[:2], d + w * 2, atol=0.01)
    assert np.allclose(x.extents[2], h + h1 * 2, atol=0.01)

    # check without capping: should be the same size just without caps
    o = sweep_polygon(Polygon(polygon), spin_path, cap=False, connect=False)
    assert np.allclose(x.bounds, o.bounds)
    assert not o.is_volume

    # try spinning along an angle
    sweep_polygon(
        Polygon(polygon), spin_path, angles=np.linspace(0.0, np.pi / 2, len(spin_path))
    )

    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 100)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10

    a = sweep_polygon(circle, path)
    assert a.is_volume


def test_scale_straight():
    # anisotropic per-point scaling on a straight Z path
    # a quarter-turn roll aligns the slice frame with the
    # world axes so slice coordinates are exactly scaled
    square = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    path = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [2.0, 0.5], [3.0, 2.0]])
    angles = np.full(len(path), np.pi / 2)

    m = sweep_polygon(square, path, angles=angles, scale=scale)
    assert m.is_volume
    assert m.is_winding_consistent

    stride = len(m.vertices) // len(path)
    slices = m.vertices.reshape((len(path), stride, 3))
    # each slice should be centered on its path point and
    # have extents of exactly twice the scale factors
    for s, (sx, sy), p in zip(slices, scale, path):
        assert np.allclose(s[:, 2], p[2])
        assert np.allclose(np.ptp(s, axis=0)[:2], [2 * sx, 2 * sy])
        assert np.allclose(s.mean(axis=0), p)

    # an open path without capping should not be a volume
    o = sweep_polygon(square, path, angles=angles, scale=scale, cap=False)
    assert not o.is_volume
    assert np.allclose(o.bounds, m.bounds)


def test_scale_curved():
    # scaling along a curved path: every slice radius should
    # match the requested scale at that path point
    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 50)
    path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * 10
    factors = np.linspace(0.5, 3.0, len(path))
    scale = np.column_stack([factors, factors])

    m = sweep_polygon(circle, path, scale=scale)
    assert m.is_volume
    assert m.is_winding_consistent

    stride = len(m.vertices) // len(path)
    slices = m.vertices.reshape((len(path), stride, 3))
    radii = np.linalg.norm(slices - path[:, None, :], axis=2)
    assert np.allclose(radii, factors[:, None])


def test_scale_hole():
    # a polygon with a hole: inner and outer rings scale together
    outer = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    hole = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    polygon = Polygon(outer, [hole])
    path = [[0, 0, 0], [0, 0, 5]]
    s = 2.0

    m = sweep_polygon(polygon, path, scale=[[s, s], [s, s]])
    assert m.is_volume
    # volume should be the scaled cross section area times height
    assert np.isclose(m.volume, polygon.area * s**2 * 5)
    # outer ring scaled from 4x4 to 8x8
    assert np.allclose(m.extents, [4 * s, 4 * s, 5])

    # taper the tube: hole should shrink with the outer ring
    t = sweep_polygon(polygon, path, scale=[[s, s], [s / 2, s / 2]])
    assert t.is_volume
    # area scales quadratically so integrate the linear taper
    expected = polygon.area * 5 * (s**2 + s * (s / 2) + (s / 2) ** 2) / 3
    assert np.isclose(t.volume, expected, rtol=1e-3)


def test_scale_angles():
    # scaling composes with the per-vertex rotation angles
    circle = Point([0, 0]).buffer(1.0)
    path = [[0, 0, 0], [0, 0, 10]]
    scale = [[2.0, 3.0], [2.0, 3.0]]

    # a quarter-turn roll aligns the slice frame with the world axes
    m0 = sweep_polygon(circle, path, angles=[np.pi / 2] * 2, scale=scale)
    assert m0.is_volume
    assert np.allclose(m0.extents, [4, 6, 10], atol=1e-6)

    # removing the roll rotates the scaled section and swaps extents
    m1 = sweep_polygon(circle, path, scale=scale)
    assert m1.is_volume
    assert np.allclose(m1.extents, [6, 4, 10], atol=1e-6)


def test_scale_closed():
    circle = Point([0, 0]).buffer(1.0)
    path = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [0, 0, 0]]
    scale = np.array(
        [[1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [2.0, 2.0], [1.0, 1.0]]
    )

    # matching first and last scale connects into a single volume
    m = sweep_polygon(circle, path, scale=scale)
    assert m.is_volume
    assert m.body_count == 1

    # mismatched endpoints would leave a broken seam
    bad = scale.copy()
    bad[-1] = [3.0, 3.0]
    with pytest.raises(ValueError):
        sweep_polygon(circle, path, scale=bad)

    # without connecting the seam mismatched endpoints are fine
    o = sweep_polygon(circle, path, scale=bad, connect=False)
    assert o.is_volume


def test_scale_invalid():
    circle = Point([0, 0]).buffer(1.0)
    path = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]

    # wrong number of scale values
    with pytest.raises(ValueError):
        sweep_polygon(circle, path, scale=np.ones((4, 2)))
    # wrong dimensionality
    with pytest.raises(ValueError):
        sweep_polygon(circle, path, scale=np.ones(3))
    # zero scale
    with pytest.raises(ValueError):
        sweep_polygon(circle, path, scale=[[1, 1], [0, 1], [1, 1]])
    # negative scale
    with pytest.raises(ValueError):
        sweep_polygon(circle, path, scale=[[1, 1], [-1, 1], [1, 1]])


def test_scale_default_unchanged():
    # no scale and all-ones scale should be identical
    # and neither should mutate the inputs
    circle = Point([0, 0]).buffer(1.0)
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.float64)
    path_orig = path.copy()
    scale = np.ones((len(path), 2))
    scale_orig = scale.copy()

    a = sweep_polygon(circle, path)
    b = sweep_polygon(circle, path, scale=scale)

    assert np.array_equal(a.faces, b.faces)
    assert np.allclose(a.vertices, b.vertices)
    assert np.array_equal(path, path_orig)
    assert np.array_equal(scale, scale_orig)


if __name__ == "__main__":
    test_simple_closed()
    test_simple_extrude()
    test_simple_open()
    test_screw()
    test_spline_3D()
    test_scale_straight()
    test_scale_curved()
    test_scale_hole()
    test_scale_angles()
    test_scale_closed()
    test_scale_invalid()
    test_scale_default_unchanged()
