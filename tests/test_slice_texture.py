"""
test_slice_texture.py
---------------------

Tests for textured plane slicing with solid-colored caps.

The new behavior is enabled through the ``cap_color`` keyword of
``trimesh.intersections.slice_mesh_plane`` (and ``Trimesh.slice_plane``):

* the original surface keeps its material and UV texture placement
* the cut (cap) faces receive a solid color
* caps keep cross-section holes, point outward and contain no
  zero-area triangles
* the input mesh is never modified
* old call signatures keep working unchanged
"""

import numpy as np

try:
    from . import generic as g
except BaseException:
    import generic as g

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def _checker_texture(size=32, cell=4):
    """Build a red/blue checkerboard texture image."""
    from PIL import Image

    image = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            image[y, x] = (
                [255, 0, 0] if (x // cell + y // cell) % 2 == 0 else [0, 0, 255]
            )
    return Image.fromarray(image)


def _textured_box(seed=0):
    """A unit box with a checker texture and per-vertex UVs."""
    mesh = g.trimesh.creation.box()
    uv = g.np.random.default_rng(seed).random((len(mesh.vertices), 2))
    mesh.visual = g.trimesh.visual.TextureVisuals(uv=uv, image=_checker_texture())
    return mesh


def _sample_colors(mesh):
    """Sample the texture color at every stored UV coordinate."""
    return g.trimesh.visual.uv_to_color(mesh.visual.uv, mesh.visual.material.image)


class SliceTextureTest(g.unittest.TestCase):
    def test_legacy_behavior_unchanged(self):
        # existing call signatures must behave exactly as before
        mesh = g.trimesh.creation.box()

        # no cap: plain geometry, no visual defined
        plain = mesh.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
        assert len(plain.faces) > 0
        assert not plain.visual.defined

        # cap=True without cap_color keeps the previous (untextured) result
        capped = mesh.slice_plane(
            plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], cap=True
        )
        assert capped.is_watertight
        assert not capped.visual.defined

        # textured mesh with cap=True but no cap_color: the surface
        # texture is now retained (no cap color is assigned)
        textured = _textured_box()
        legacy = textured.slice_plane(
            plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], cap=True
        )
        assert legacy.visual.kind == "texture"
        assert legacy.visual.uv.shape == (len(legacy.vertices), 2)
        assert "color" not in legacy.face_attributes
        assert legacy.is_watertight

        # textured mesh, cap=False still keeps the texture as before
        uncapped = textured.slice_plane(
            plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], cap=False
        )
        assert uncapped.visual.kind == "texture"
        assert uncapped.visual.uv.shape == (len(uncapped.vertices), 2)

        # cap_color is ignored when cap is False
        ignored = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=False,
            cap_color=[1, 2, 3, 255],
        )
        assert not ignored.visual.defined

    def test_input_not_modified(self):
        mesh = _textured_box(seed=1)
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        uv = mesh.visual.uv.copy()
        diffuse = g.np.asarray(mesh.visual.material.diffuse).copy()

        mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[10, 200, 90, 255],
        )

        assert g.np.array_equal(mesh.vertices, vertices)
        assert g.np.array_equal(mesh.faces, faces)
        assert g.np.array_equal(mesh.visual.uv, uv)
        assert g.np.array_equal(mesh.visual.material.diffuse, diffuse)

        # the returned material is a copy, not a shared reference
        sliced = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[10, 200, 90, 255],
        )
        sliced.visual.material.diffuse = g.np.array([9, 9, 9, 255])
        assert g.np.array_equal(mesh.visual.material.diffuse, diffuse)

    def test_texture_preserved_with_real_sampling(self):
        mesh = _textured_box(seed=2)
        image = mesh.visual.material.image
        original_colors = g.trimesh.visual.uv_to_color(mesh.visual.uv, image)

        sliced = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[0, 255, 0, 255],
        )

        # the surface still carries the original texture and material
        assert sliced.visual.kind == "texture"
        assert sliced.visual.uv.shape == (len(sliced.vertices), 2)
        assert g.np.asarray(sliced.visual.material.image).shape == g.np.asarray(
            image
        ).shape

        sampled = _sample_colors(sliced)

        # vertices inherited unchanged from the source keep their exact
        # sampled texture color (texture placement is preserved)
        from scipy.spatial import cKDTree

        distance, index = cKDTree(mesh.vertices).query(sliced.vertices)
        exact = distance < 1e-10
        assert exact.sum() > 0
        assert (sampled[exact] == original_colors[index[exact]]).all()

        # new vertices created on the cut plane have barycentrically
        # interpolated UVs, so they still sample a real texture texel
        cut = g.np.isclose(sliced.vertices[:, 2], 0.0)
        assert cut.sum() > 0
        cut_colors = sampled[cut][:, :3]
        red = g.np.all(cut_colors == [255, 0, 0], axis=1)
        blue = g.np.all(cut_colors == [0, 0, 255], axis=1)
        assert (red | blue).all()

    def test_cap_color_and_cap_faces(self):
        mesh = _textured_box(seed=3)
        cap_rgba = g.np.array([11, 222, 33, 255], dtype=g.np.uint8)
        sliced = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=cap_rgba,
        )

        colors = sliced.face_attributes["color"]
        is_cap = g.np.all(colors == cap_rgba, axis=1)

        # six triangles tile the 1x1 cross-section of the box
        assert is_cap.sum() == 6
        assert g.np.isclose(sliced.area_faces[is_cap].sum(), 1.0)

        # cap normals point away from the retained (+normal) side
        assert g.np.allclose(sliced.face_normals[is_cap], [0, 0, -1])

        # original surface faces carry the default uncolored marker,
        # not the solid cap color
        default = g.trimesh.visual.DEFAULT_COLOR
        assert g.np.all(colors[~is_cap] == default)

        # no zero-area faces anywhere and the mesh is geometrically closed
        assert (sliced.area_faces > 1e-12).all()
        assert sliced.is_watertight
        assert sliced.is_winding_consistent

        # cap vertices all lie on the slicing plane
        cap_vertices = g.np.unique(sliced.faces[is_cap])
        assert g.np.allclose(sliced.vertices[cap_vertices, 2], 0.0)

    def test_cap_color_without_texture(self):
        # a plain (untextured) mesh gets ordinary face colors
        mesh = g.trimesh.creation.box()
        sliced = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color="#ff8800",
        )

        assert sliced.visual.kind == "face"
        colors = sliced.visual.face_colors
        is_cap = g.np.all(colors == [255, 136, 0, 255], axis=1)
        assert is_cap.sum() == 6
        assert g.np.isclose(sliced.area_faces[is_cap].sum(), 1.0)
        assert g.np.allclose(sliced.face_normals[is_cap, 2], -1.0)
        assert sliced.is_watertight

        # 3-channel and 0-1 float colors are accepted too
        rgb = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[10, 20, 30],
        )
        assert g.np.all(rgb.visual.face_colors.reshape(-1, 4)[:, 3] == 255)
        flt = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[1.0, 0.5, 0.0],
        )
        assert g.np.any(
            g.np.all(flt.visual.face_colors == [255, 128, 0, 255], axis=1)
        )

    def test_plane_through_vertices(self):
        mesh = _textured_box(seed=4)
        # the plane passes exactly through the four bottom vertices
        sliced = mesh.slice_plane(
            plane_origin=[0, 0, -0.5],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[7, 77, 177, 255],
        )
        colors = sliced.face_attributes["color"]
        is_cap = g.np.all(colors == [7, 77, 177, 255], axis=1)

        assert sliced.is_watertight
        assert (sliced.area_faces > 1e-12).all()
        assert g.np.isclose(sliced.area_faces[is_cap].sum(), 1.0)
        assert g.np.allclose(sliced.face_normals[is_cap, 2], -1.0)
        # the ten box side faces survive (the bottom face is replaced
        # by the cap that shares its boundary)
        assert (~is_cap).sum() == 10

        # plane through the top vertices retaining the empty side
        empty = mesh.slice_plane(
            plane_origin=[0, 0, 0.5],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[1, 2, 3, 255],
        )
        assert len(empty.vertices) == 0
        assert len(empty.faces) == 0

    def test_hole_preserved_in_cap(self):
        if not g.has_earcut:
            return

        # slicing a torus through its mid-plane produces an annular cap:
        # the cross-section polygon has an interior hole
        torus = g.trimesh.creation.torus(
            major_radius=5.0,
            minor_radius=1.0,
            major_sections=64,
            minor_sections=16,
        )
        sliced = torus.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[255, 128, 0, 255],
        )

        colors = sliced.visual.face_colors
        is_cap = g.np.all(colors == [255, 128, 0, 255], axis=1)
        assert is_cap.sum() > 0

        # closed, outward-facing and non-degenerate geometry
        assert sliced.is_watertight
        assert sliced.is_winding_consistent
        assert (sliced.area_faces > 1e-12).all()
        assert g.np.allclose(sliced.face_normals[is_cap, 2], -1.0)

        # cap area equals the annulus: pi*((R+r)^2 - (R-r)^2)
        expected = g.np.pi * (6.0**2 - 4.0**2)
        assert g.np.isclose(sliced.area_faces[is_cap].sum(), expected, rtol=1e-2)

        # union of cap triangles leaves the center hole uncovered while
        # covering the ring band
        cap_union = unary_union(
            [Polygon(sliced.vertices[face][:, :2]) for face in sliced.faces[is_cap]]
        )
        assert not cap_union.covers(Point(0.0, 0.0))
        assert cap_union.covers(Point(5.0, 0.0))
        assert g.np.isclose(cap_union.area, expected, rtol=1e-2)

    def test_no_intersection(self):
        mesh = _textured_box(seed=5)

        # plane beyond the mesh, retained side contains everything:
        # the full model comes back untouched
        full = mesh.slice_plane(
            plane_origin=[0, 0, -10],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[1, 2, 3, 255],
        )
        assert len(full.vertices) == len(mesh.vertices)
        assert len(full.faces) == len(mesh.faces)
        assert full.visual.kind == "texture"
        assert full.is_watertight

        # plane beyond the mesh on the other side: empty grid
        empty = mesh.slice_plane(
            plane_origin=[0, 0, 10],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[1, 2, 3, 255],
        )
        assert len(empty.vertices) == 0
        assert len(empty.faces) == 0

        # uncapped miss still returns an empty textured grid
        uncapped = mesh.slice_plane(
            plane_origin=[0, 0, 10], plane_normal=[0, 0, 1], cap=False
        )
        assert len(uncapped.vertices) == 0
        assert uncapped.visual.kind == "texture"
        assert uncapped.visual.uv.shape == (0, 2)

    def test_uv_seam_texture_preserved(self):
        # an unwrapped mesh duplicates 3D vertices along UV seams: the
        # same spatial position carries several different UV coordinates.
        # Slicing must keep every seam vertex's texture placement.
        box = g.trimesh.creation.box()
        unmerged_vertices = box.vertices[box.faces].reshape(-1, 3)
        unmerged_faces = g.np.arange(len(box.faces) * 3).reshape(-1, 3)
        mesh = g.trimesh.Trimesh(
            vertices=unmerged_vertices, faces=unmerged_faces, process=False
        )
        uv = g.np.random.default_rng(7).random((len(mesh.vertices), 2))
        image = _checker_texture()
        mesh.visual = g.trimesh.visual.TextureVisuals(uv=uv, image=image)

        original_colors = g.trimesh.visual.uv_to_color(mesh.visual.uv, image)

        sliced = mesh.slice_plane(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
            cap=True,
            cap_color=[1, 2, 3, 255],
        )

        # capping merges the duplicated seam boundary into a closed shell
        assert sliced.is_watertight
        assert (sliced.area_faces > 1e-12).all()
        assert sliced.visual.uv.shape[0] == len(sliced.vertices)

        # surviving seam vertices keep their exact per-vertex texture
        # color. Spatial positions are duplicated at seams, so match on
        # the (position, uv) pair rather than position alone.
        sampled = _sample_colors(sliced)
        source_keys = g.np.hstack([mesh.vertices, mesh.visual.uv])
        result_keys = g.np.hstack([sliced.vertices, sliced.visual.uv])
        from scipy.spatial import cKDTree

        distance, index = cKDTree(source_keys).query(result_keys)
        exact = distance < 1e-10
        assert exact.sum() > 0
        assert (sampled[exact] == original_colors[index[exact]]).all()

        # the cap is still a solid color with the expected 1x1 area
        colors = sliced.face_attributes["color"]
        is_cap = g.np.all(colors == [1, 2, 3, 255], axis=1)
        assert g.np.isclose(sliced.area_faces[is_cap].sum(), 1.0)
        assert g.np.allclose(sliced.face_normals[is_cap, 2], -1.0)

    def test_multiple_planes_textured(self):
        # two planes at once: both caps colored, surface stays textured
        mesh = _textured_box(seed=6)
        origins = [mesh.bounds[1] - 0.05, mesh.bounds[1] - 0.05]
        normals = [[0, 0, 1], [0, 1, 0]]
        sliced = mesh.slice_plane(
            plane_origin=origins,
            plane_normal=normals,
            cap=True,
            cap_color=[12, 34, 56, 255],
        )

        assert sliced.visual.kind == "texture"
        assert sliced.visual.uv.shape[0] == len(sliced.vertices)
        assert sliced.is_watertight
        assert (sliced.area_faces > 1e-12).all()

        colors = sliced.face_attributes["color"]
        is_cap = g.np.all(colors == [12, 34, 56, 255], axis=1)
        assert is_cap.sum() > 0
        # every cap face is parallel to one of the slicing planes
        cap_normals = sliced.face_normals[is_cap]
        aligned = g.np.isclose(g.np.abs(cap_normals[:, 2]), 1.0) | g.np.isclose(
            g.np.abs(cap_normals[:, 1]), 1.0
        )
        assert aligned.all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
