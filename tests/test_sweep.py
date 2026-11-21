import numpy as np
import pytest
from pyinstrument import Profiler
from shapely import affinity
from shapely.geometry import Point, Polygon

import trimesh
from trimesh.creation import sweep_polygon
from trimesh.creation import triangulate_polygon
from trimesh.geometry import faces_to_edges
from trimesh import grouping

trimesh.tol.strict = True


def _boundary_stride(polygon):
    """Number of boundary vertices the sweep uses per slice for a polygon."""
    vertices_2d, faces_2d = triangulate_polygon(polygon)
    edges = faces_to_edges(faces_2d)
    edges_unique = grouping.group_rows(np.sort(edges, axis=1), require_count=1)
    return len(np.unique(edges[edges_unique].reshape(-1)))


def _slice_normals(path):
    """Recreate the per-slice plane normals used internally by the sweep."""
    vector = trimesh.util.unitize(path[1:] - path[:-1])
    vector_mean = trimesh.util.unitize(vector[1:] + vector[:-1])
    normal = np.concatenate([[vector[0]], vector_mean, [vector[-1]]], axis=0)
    closed = np.linalg.norm(path[0] - path[-1]) < trimesh.tol.merge
    if closed:
        normal[0] = trimesh.util.unitize(normal[[0, -1]].mean(axis=0))
    return normal / np.linalg.norm(normal, axis=1)[:, None]


def _arc_path(count=40, radius=10.0):
    theta = np.linspace(0.0, np.pi, count)
    return (
        np.column_stack([np.cos(theta), np.sin(theta), np.zeros(count)]) * radius
    )


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


def test_scales_default_unchanged():
    # without scales the result must be identical to the original behavior
    polygon = Point([0, 0]).buffer(1.0)
    path = [[0, 0, 0], [0, 0, 10.0]]

    plain = sweep_polygon(polygon, path)
    unit = sweep_polygon(polygon, path, scales=np.ones((2, 2)))
    assert np.allclose(plain.vertices, unit.vertices)
    assert np.allclose(plain.faces, unit.faces)


def test_scales_straight_constant():
    # a straight extrusion with a constant uniform scale keeps the
    # cross section shape and normals while multiplying the area
    polygon = Point([0, 0]).buffer(1.0)
    height = 7.0
    path = [[0, 0, 0], [0, 0, height]]
    factor = 2.0

    plain = sweep_polygon(polygon, path)
    scaled = sweep_polygon(polygon, path, scales=np.full((2, 2), factor))

    # cross section doubles in both in-plane directions, path length fixed
    assert np.allclose(scaled.extents[:2], factor * plain.extents[:2])
    assert np.isclose(scaled.extents[2], height)
    # area scales with the product of the two factors
    assert np.isclose(scaled.volume, polygon.area * factor * factor * height)
    assert scaled.is_volume
    # a straight sweep has the same face normals at any constant scale
    assert np.allclose(plain.face_normals, scaled.face_normals, atol=1e-12)


def test_scales_straight_taper():
    # per-vertex, per-axis scaling produces a tapered sweep on a straight path
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    height = 5.0
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]])
    scales = np.array([[1.0, 1.0], [0.5, 2.0]])

    mesh = sweep_polygon(polygon, path, scales=scales)
    stride = _boundary_stride(polygon)

    # first slice keeps the original 2x2 square, last is 1 wide in x / 4 tall in y
    first = mesh.vertices[:stride]
    last = mesh.vertices[stride : 2 * stride]
    # on a Z path local x maps to world y and local y maps to world x
    assert np.allclose(np.ptp(first[:, 0]), 2.0)
    assert np.allclose(np.ptp(first[:, 1]), 2.0)
    assert np.allclose(np.ptp(last[:, 0]), 4.0)
    assert np.allclose(np.ptp(last[:, 1]), 1.0)

    # slices remain in horizontal planes, normals still point along the path
    assert np.allclose(first[:, 2], 0.0)
    assert np.allclose(last[:, 2], height)
    cap_normals = mesh.face_normals[
        np.isclose(np.abs(mesh.face_normals[:, 2]), 1.0)
    ]
    assert len(cap_normals) > 0

    # linearly interpolated half extents hx = 1 - 0.1 z, hy = 1 + 0.2 z:
    # cross section area A(z) = (2 * hx) * (2 * hy)
    expected_volume = np.trapezoid(
        4
        * (1 - 0.1 * np.linspace(0, height, 1001))
        * (1 + 0.2 * np.linspace(0, height, 1001)),
        dx=height / 1000,
    )
    assert np.isclose(mesh.volume, expected_volume)
    assert mesh.is_volume


def test_scales_curved_slices():
    # on a curved path every scaled slice still lies in the plane normal to
    # the local path tangent and has the requested cross section dimensions
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    count = 40
    path = _arc_path(count=count)
    scales = np.column_stack(
        [np.linspace(1.0, 0.5, count), np.linspace(0.5, 2.0, count)]
    )

    mesh = sweep_polygon(polygon, path, scales=scales)
    stride = _boundary_stride(polygon)
    normals = _slice_normals(path)

    assert len(mesh.vertices) == count * stride
    for index in range(count):
        block = mesh.vertices[index * stride : (index + 1) * stride] - path[index]
        # every slice vertex sits in the slice plane
        assert np.allclose(block @ normals[index], 0.0, atol=1e-8)
        # farthest corner of the scaled square sits at sqrt(sx**2 + sy**2)
        radii = np.linalg.norm(block, axis=1)
        assert np.isclose(
            radii.max(), np.hypot(scales[index, 0], scales[index, 1]), atol=1e-8
        )

    # constant scaling leaves the end-cap normals (slice plane orientation)
    # unchanged and the capped sweep remains a closed volume; side wall
    # normals do tilt because the scaled tube follows the bend
    plain = sweep_polygon(polygon, path)
    constant = sweep_polygon(polygon, path, scales=np.full((count, 2), 1.5))
    cap_selector = np.all(plain.faces < stride, axis=1) | np.all(
        plain.faces >= (count - 1) * stride, axis=1
    )
    assert np.allclose(
        plain.face_normals[cap_selector],
        constant.face_normals[cap_selector],
        atol=1e-12,
    )
    assert constant.is_volume


def test_scales_with_holes():
    # inner and outer rings must scale together about the shared origin
    outer = [(-4, -4), (4, -4), (4, 4), (-4, 4)]
    inner = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    polygon = Polygon(outer, [inner])
    height = 5.0
    path = [[0, 0, 0], [0, 0, height]]
    factor = 2.0

    mesh = sweep_polygon(polygon, path, scales=np.full((2, 2), factor))
    stride = _boundary_stride(polygon)
    block = mesh.vertices[:stride][:, :2]
    radii = np.linalg.norm(block, axis=1)

    # outer corners reach 4 * factor * sqrt(2), inner corners factor * sqrt(2)
    assert np.isclose(radii.max(), factor * 4.0 * np.sqrt(2))
    assert np.any(np.isclose(radii, factor * 1.0 * np.sqrt(2)))
    # volume is the annulus area times height with both rings uniformly scaled
    assert np.isclose(mesh.volume, polygon.area * factor * factor * height)
    assert mesh.is_volume


def test_scales_with_angles():
    # scaling happens about the origin before the existing roll rotation:
    # equivalent to pre-scaling the 2D profile and then rotating it
    polygon = Polygon([(-1.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-1.5, 1.0)])
    height = 8.0
    path = [[0, 0, 0], [0, 0, height]]
    scales = np.array([[2.0, 0.5], [2.0, 0.5]])
    angles = np.array([np.pi / 2, np.pi / 2])

    mesh = sweep_polygon(polygon, path, scales=scales, angles=angles)

    reference_profile = affinity.rotate(
        affinity.scale(polygon, xfact=2.0, yfact=0.5), 90.0, origin=(0.0, 0.0)
    )
    reference = sweep_polygon(reference_profile, path)

    assert np.isclose(mesh.volume, reference.volume)
    assert np.allclose(np.sort(mesh.extents), np.sort(reference.extents))
    # the 90 degree roll swaps the in-plane world axes: without roll
    # local x maps to world y, with the roll scaled local x (width 6)
    # lands on world x and scaled local y (height 1) on world y
    assert np.isclose(mesh.extents[0], 6.0)
    assert np.isclose(mesh.extents[1], 1.0)
    assert mesh.is_volume


def test_scales_closed_seam():
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    path = np.array(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [0, 0, 0]], dtype=np.float64
    )

    # identical first/last scales connect into a single watertight body even
    # when the cross section varies around the loop
    scales = np.array([[2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [2.0, 2.0]])
    mesh = sweep_polygon(polygon, path, scales=scales)
    assert mesh.is_volume
    assert mesh.body_count == 1

    # conflicting end scales cannot share a seam
    bad = np.array([[2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [1.5, 1.5]])
    with pytest.raises(ValueError, match="identical first and last"):
        sweep_polygon(polygon, path, scales=bad)

    # when the loop is not connected the mismatch is allowed and capped
    disconnected = sweep_polygon(
        polygon, path, scales=bad, connect=False, cap=True
    )
    assert disconnected.is_volume


def test_scales_open_capping():
    # open paths keep honoring the cap option while scaling
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    path = _arc_path()
    scales = np.linspace(1.0, 0.5, len(path))[:, None].repeat(2, axis=1)

    capped = sweep_polygon(polygon, path, scales=scales, cap=True, connect=False)
    uncapped = sweep_polygon(polygon, path, scales=scales, cap=False, connect=False)
    assert capped.is_volume
    assert not uncapped.is_volume
    assert np.allclose(capped.bounds, uncapped.bounds)


def test_scales_bad_count():
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    path = [[0, 0, 0], [0, 0, 5.0]]

    for bad in (
        np.ones((3, 2)),
        np.ones((2,)),
        np.ones((2, 3)),
        np.ones((1, 2)),
    ):
        with pytest.raises(ValueError, match="scales must have shape"):
            sweep_polygon(polygon, path, scales=bad)


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 1.0], [0.0, 1.0]],
        [[1.0, 1.0], [-1.0, 1.0]],
        [[1.0, np.nan], [1.0, 1.0]],
        [[1.0, np.inf], [1.0, 1.0]],
    ],
)
def test_scales_must_be_positive(bad):
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    path = [[0, 0, 0], [0, 0, 5.0]]
    with pytest.raises(ValueError, match="finite and strictly positive"):
        sweep_polygon(polygon, path, scales=np.array(bad, dtype=np.float64))


def test_scales_inputs_not_modified():
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    polygon_snapshot = list(polygon.exterior.coords)
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    path_snapshot = path.copy()
    scales = np.array([[1.5, 0.75], [0.5, 2.0]])
    scales_snapshot = scales.copy()

    sweep_polygon(polygon, path, scales=scales)

    assert np.array_equal(path, path_snapshot)
    assert np.array_equal(scales, scales_snapshot)
    assert list(polygon.exterior.coords) == polygon_snapshot


if __name__ == "__main__":
    test_simple_closed()
    test_simple_extrude()
    test_simple_open()
    test_screw()
    test_spline_3D()
    test_scales_default_unchanged()
    test_scales_straight_constant()
    test_scales_straight_taper()
    test_scales_curved_slices()
    test_scales_with_holes()
    test_scales_with_angles()
    test_scales_closed_seam()
    test_scales_open_capping()
    test_scales_bad_count()
    test_scales_must_be_positive()
    test_scales_inputs_not_modified()
