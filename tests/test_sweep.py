import numpy as np
import pytest
from pyinstrument import Profiler
from shapely.geometry import Point, Polygon

import trimesh
from trimesh.creation import sweep_polygon

trimesh.tol.strict = True


def _section_rings(mesh, n_path):
    """
    Split the sweep mesh vertices into the ordered per-vertex section
    rings. Sections are laid out contiguously, including the duplicated
    closing ring before it gets folded back onto the first one.
    """
    stride = len(mesh.vertices) // n_path
    assert len(mesh.vertices) == stride * n_path
    return mesh.vertices.reshape(n_path, stride, 3)


def _path_normals(path):
    """Replicate the per-slice plane normals used inside `sweep_polygon`."""
    vector = trimesh.util.unitize(path[1:] - path[:-1])
    vector_mean = trimesh.util.unitize(vector[1:] + vector[:-1])
    normal = np.concatenate([[vector[0]], vector_mean, [vector[-1]]], axis=0)
    closed = np.linalg.norm(path[0] - path[-1]) < trimesh.tol.merge
    if closed:
        normal[0] = trimesh.util.unitize(normal[[0, -1]].mean(axis=0))
    return normal


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


def test_scales_straight():
    # per-point section scaling on a straight path
    circle = Point([0, 0]).buffer(1.0)
    n = 5
    path = np.zeros((n, 3))
    path[:, 2] = np.linspace(0.0, 4.0, n)

    # non-uniform taper, a different factor on each section axis
    scales = np.column_stack(
        [np.linspace(1.0, 3.0, n), np.linspace(2.0, 0.5, n)]
    )
    mesh = sweep_polygon(circle, path, scales=scales)

    assert mesh.is_watertight
    assert mesh.is_volume
    # the section frame maps section `x` to 3D `Y` and section `y` to 3D `X`
    assert np.allclose(mesh.extents, [4.0, 6.0, 4.0])

    rings = _section_rings(mesh, n)
    # rings stay centered on the path and lie in the XY section plane
    assert np.allclose(rings[:, :, 2], path[:, 2][:, None])
    assert np.allclose(rings.mean(axis=1), path, atol=1e-12)

    # each ring must reproduce the requested axis-aligned dimensions,
    # with the section axes swapped into the 3D X/Y frame
    for ring, scale in zip(rings, scales):
        ext = np.ptp(ring[:, :2], axis=0)
        assert np.allclose(ext, 2.0 * scale[::-1], atol=1e-8)

    # section normals point along the straight path direction (+Z)
    normals = _path_normals(path)
    assert np.allclose(normals, np.tile([0.0, 0.0, 1.0], (n, 1)))

    # a constant scale on a straight path is a plain cylinder
    radius = 1.5
    height = 4.0
    cyl_scales = np.tile([radius, radius], (n, 1))
    cylinder = sweep_polygon(circle, path, scales=cyl_scales)
    assert cylinder.is_volume
    assert np.isclose(cylinder.volume, np.pi * radius**2 * height, rtol=0.02)
    # side faces span two slices and stay parallel to the path axis,
    # while cap faces keep normals aligned with the axis
    side = np.ptp(cylinder.vertices[cylinder.faces, 2], axis=1) > 1e-10
    assert np.allclose(cylinder.face_normals[side, 2], 0.0, atol=1e-10)
    assert np.allclose(np.abs(cylinder.face_normals[~side, 2]), 1.0, atol=1e-10)

    # a linear uniform taper is a conical frustum
    r0, r1 = 1.0, 2.0
    frustum = sweep_polygon(
        circle, path, scales=np.linspace(r0, r1, n)[:, None].repeat(2, axis=1)
    )
    expected = np.pi * height / 3.0 * (r0**2 + r0 * r1 + r1**2)
    assert np.isclose(frustum.volume, expected, rtol=0.02)

    # scaling is centered on the section origin, not the polygon centroid
    off_center = Polygon([(1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)])
    straight = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s = np.array([[0.5, 2.0], [0.5, 2.0]])
    off_mesh = sweep_polygon(off_center, straight, scales=s)
    rings = _section_rings(off_mesh, 2)
    # the section frame rotates section `y` onto 3D X and section `x`
    # onto (negated) 3D Y, so check extents and the scaled center instead
    extents_xy = np.ptp(rings[:, :, :2], axis=1)
    assert np.allclose(extents_xy[:, 0], 1.0 * s[:, 1])
    assert np.allclose(extents_xy[:, 1], 2.0 * s[:, 0])
    center = rings[:, :, :2].mean(axis=1)
    assert np.allclose(center[:, 0], 1.5 * s[:, 1])
    assert np.allclose(center[:, 1], -2.0 * s[:, 0])


def test_scales_curved():
    # per-point section scaling on a curved open path
    circle = Point([0, 0]).buffer(1.0)
    theta = np.linspace(0.0, np.pi, 40)
    path = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros(len(theta))]
    ) * 10.0
    n = len(path)
    scales = np.column_stack(
        [np.linspace(0.5, 2.0, n), np.linspace(1.5, 0.75, n)]
    )

    mesh = sweep_polygon(circle, path, scales=scales)
    assert mesh.is_watertight
    assert mesh.is_volume

    normals = _path_normals(path)
    rings = _section_rings(mesh, n)
    for ring, center, normal, scale in zip(rings, path, normals, scales):
        radial = ring - center
        # the section plane is perpendicular to the local path normal
        assert np.allclose(radial @ normal, 0.0, atol=1e-8)
        radii = np.linalg.norm(radial, axis=1)
        assert radii.max() <= scale.max() + 1e-8
        assert radii.min() >= scale.min() - 1e-8

    # unit scales must reproduce the unscaled section dimensions
    plain = sweep_polygon(circle, path)
    plain_rings = _section_rings(plain, n)
    assert np.allclose(
        np.linalg.norm(plain_rings - path[:, None, :], axis=2), 1.0, atol=1e-8
    )

    # a constant factor uniformly enlarges every section
    factor = 1.25
    grown = sweep_polygon(circle, path, scales=np.tile([factor, factor], (n, 1)))
    grown_rings = _section_rings(grown, n)
    assert np.allclose(
        np.linalg.norm(grown_rings - path[:, None, :], axis=2),
        factor,
        atol=1e-8,
    )


def test_scales_polygon_with_hole():
    # inner and outer contours must scale together
    outer = Point([0, 0]).buffer(2.0)
    inner = Point([0, 0]).buffer(1.0)
    annulus = outer.difference(inner)

    n = 6
    path = np.zeros((n, 3))
    path[:, 2] = np.linspace(0.0, 6.0, n)
    scales = np.column_stack(
        [np.linspace(1.0, 2.0, n), np.linspace(2.0, 0.5, n)]
    )

    mesh = sweep_polygon(annulus, path, scales=scales)
    assert mesh.is_watertight
    assert mesh.is_volume

    rings = _section_rings(mesh, n)
    for ring, scale in zip(rings, scales):
        ext = np.ptp(ring, axis=0)
        # outer contour spans the full scaled extents
        assert np.allclose(ext[:2], 4.0 * scale[::-1], atol=1e-8)
        # the scaled hole stays open, so no vertices near the section center
        radii = np.linalg.norm(ring - ring.mean(axis=0), axis=1)
        assert radii.min() > scale.min() - 1e-6

    # volume of a linearly varying annulus, integrated per segment:
    # section area is pi * (4 - 1) * sx * sy
    area0 = np.pi * (2.0**2 - 1.0**2)
    section_area = area0 * scales[:, 0] * scales[:, 1]
    dz = np.diff(path[:, 2])
    expected = np.sum((section_area[:-1] + section_area[1:]) * 0.5 * dz)
    assert np.isclose(mesh.volume, expected, rtol=0.03)


def test_scales_with_angles():
    # scaling happens first, then the roll rotation from `angles`
    circle = Point([0, 0]).buffer(1.0)
    n = 5
    path = np.zeros((n, 3))
    path[:, 2] = np.linspace(0.0, 4.0, n)
    scales = np.tile([2.0, 0.5], (n, 1))

    angle = 0.7
    reference = sweep_polygon(circle, path, scales=scales)
    rolled = sweep_polygon(
        circle, path, angles=np.full(n, angle), scales=scales
    )
    assert rolled.is_volume

    stride = len(reference.vertices) // n
    ref_rings = reference.vertices.reshape(n, stride, 3)
    roll_rings = rolled.vertices.reshape(n, stride, 3)

    # rotation preserves the scaled section dimensions
    ref_radii = np.linalg.norm(ref_rings[:, :, :2], axis=2)
    roll_radii = np.linalg.norm(roll_rings[:, :, :2], axis=2)
    assert np.allclose(ref_radii, roll_radii)

    # for a straight +Z path the section is rotated by Rz(-angle)
    rotation = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    expected_xy = ref_rings[:, :, :2] @ rotation.T
    assert np.allclose(roll_rings[:, :, :2], expected_xy, atol=1e-10)
    assert np.allclose(roll_rings[:, :, 2], ref_rings[:, :, 2])


def test_scales_closed_seam():
    square = Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
    path = np.array(
        [[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    n = len(path)
    scales = np.array(
        [[1.0, 1.0], [2.0, 1.5], [2.0, 1.5], [1.0, 1.0], [1.0, 1.0]]
    )

    # matching first/last scales connect into a single watertight body
    mesh = sweep_polygon(square, path, scales=scales)
    assert mesh.is_volume
    assert mesh.body_count == 1
    # the duplicated closing ring gets folded back, no extra vertices
    assert len(mesh.vertices) == (n - 1) * len(
        sweep_polygon(square, path, connect=False).vertices
    ) // n

    # mismatched first/last scales must be rejected with a clear reason
    mismatch = scales.copy()
    mismatch[-1] = [1.5, 1.0]
    with pytest.raises(ValueError, match="first and last"):
        sweep_polygon(square, path, scales=mismatch)

    # with the connection disabled the mismatch is allowed and capping
    # still follows the `cap` option
    capped = sweep_polygon(square, path, scales=mismatch, connect=False)
    assert capped.is_volume
    uncapped = sweep_polygon(
        square, path, scales=mismatch, connect=False, cap=False
    )
    assert not uncapped.is_watertight
    assert np.allclose(capped.bounds, uncapped.bounds)


def test_scales_validation():
    circle = Point([0, 0]).buffer(1.0)
    n = 4
    path = np.zeros((n, 3))
    path[:, 2] = np.linspace(0.0, 3.0, n)

    # wrong number of rows or columns
    with pytest.raises(ValueError, match="shape"):
        sweep_polygon(circle, path, scales=np.ones((n - 1, 2)))
    with pytest.raises(ValueError, match="shape"):
        sweep_polygon(circle, path, scales=np.ones((n, 3)))
    with pytest.raises(ValueError, match="shape"):
        sweep_polygon(circle, path, scales=np.ones(n))

    # non-positive or non-finite factors
    bad_zero = np.ones((n, 2))
    bad_zero[1, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        sweep_polygon(circle, path, scales=bad_zero)

    bad_negative = np.ones((n, 2))
    bad_negative[2, 1] = -2.0
    with pytest.raises(ValueError, match="positive"):
        sweep_polygon(circle, path, scales=bad_negative)

    bad_nan = np.ones((n, 2))
    bad_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sweep_polygon(circle, path, scales=bad_nan)


def test_scales_default_and_inputs_unchanged():
    circle = Point([0, 0]).buffer(1.0)
    n = 4
    path = np.zeros((n, 3))
    path[:, 2] = np.linspace(0.0, 3.0, n)

    without = sweep_polygon(circle, path)
    with_ones = sweep_polygon(circle, path, scales=np.ones((n, 2)))
    assert np.allclose(
        np.sort(without.vertices, axis=0), np.sort(with_ones.vertices, axis=0)
    )
    assert np.array_equal(without.faces, with_ones.faces)

    # inputs must never be modified
    scales = np.column_stack(
        [np.linspace(1.0, 2.0, n), np.linspace(2.0, 1.0, n)]
    )
    path_backup = path.copy()
    scales_backup = scales.copy()
    sweep_polygon(circle, path, scales=scales)
    assert np.array_equal(path, path_backup)
    assert np.array_equal(scales, scales_backup)


if __name__ == "__main__":
    test_simple_closed()
    test_simple_extrude()
    test_simple_open()
    test_screw()
    test_spline_3D()
    test_scales_straight()
    test_scales_curved()
    test_scales_polygon_with_hole()
    test_scales_with_angles()
    test_scales_closed_seam()
    test_scales_validation()
    test_scales_default_and_inputs_unchanged()
