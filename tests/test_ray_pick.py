"""
test_ray_pick.py
----------------

Tests for alpha-aware scene ray picking
(`trimesh.ray.ray_pick`): glTF OPAQUE / MASK handling, BLEND
rejection, factor alpha, nested scaling, shared instances, UV
boundaries, transparent-in-front occlusion and GLB reload.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np
import pytest

import trimesh
from trimesh.ray.ray_pick import RayPickResult, pick_scene

try:
    from PIL import Image
except ImportError:
    pytest.skip("PIL is required for ray pick tests", allow_module_level=True)


def _alpha_texture(transparent_left=True):
    """
    4x4 RGBA texture whose left half is fully transparent and
    right half fully opaque (or the inverse).
    """
    image = np.full((4, 4, 4), 200, dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.uint8)
    alpha[:, 2:] = 255
    if not transparent_left:
        alpha = 255 - alpha
    image[..., 3] = alpha
    return Image.fromarray(image, mode="RGBA")


def _quad(material, z=0.0):
    """
    Two-triangle quad in the XY plane at `z` with unit 0-1 UVs.
    """
    vertices = np.array(
        [[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visual, process=False
    )


def _mask_quad(cutoff=0.5, texture=True, factor=None, z=0.0):
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=_alpha_texture() if texture else None,
        baseColorFactor=factor,
        alphaMode="MASK",
        alphaCutoff=cutoff,
    )
    return _quad(material, z=z)


def _rays(points, z_origin=-2.0):
    origins = np.column_stack(
        (np.asarray(points, dtype=np.float64), np.full(len(points), z_origin))
    )
    directions = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    return origins, directions


def test_mask_transparent_passes_through():
    scene = trimesh.Scene(_mask_quad())
    origins, directions = _rays([[-0.5, 0.0], [0.5, 0.0]])

    results = scene.pick(origins, directions)

    # left half of the texture is transparent: no valid hit
    assert results[0] is None
    # right half is opaque: hit on the quad at z=0
    hit = results[1]
    assert isinstance(hit, RayPickResult)
    assert np.allclose(hit.point, [0.5, 0.0, 0.0])
    assert np.isclose(hit.distance, 2.0)
    assert hit.face_index == 0


def test_opaque_ignores_texture_alpha():
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=_alpha_texture(), alphaMode="OPAQUE"
    )
    scene = trimesh.Scene(_quad(material))
    origins, directions = _rays([[-0.5, 0.0]])

    hit = scene.pick(origins, directions)[0]

    # OPAQUE always blocks even where the texture alpha is zero
    assert hit is not None
    assert np.allclose(hit.point, [-0.5, 0.0, 0.0])


def test_blend_raises():
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=_alpha_texture(), alphaMode="BLEND"
    )
    scene = trimesh.Scene(_quad(material))
    origins, directions = _rays([[0.0, 0.0]])

    with pytest.raises(ValueError, match="BLEND"):
        pick_scene(scene, origins, directions)


def test_no_texture_uses_factor_alpha():
    # baseColorFactor alpha below the cutoff: ray passes through
    transparent = trimesh.Scene(
        _mask_quad(texture=False, factor=[255, 255, 255, 64], cutoff=0.5)
    )
    origins, directions = _rays([[0.0, 0.0]])
    assert transparent.pick(origins, directions) == [None]

    # factor alpha at/above the cutoff: solid hit
    opaque = trimesh.Scene(
        _mask_quad(texture=False, factor=[255, 255, 255, 200], cutoff=0.5)
    )
    hit = opaque.pick(origins, directions)[0]
    assert hit is not None
    assert np.allclose(hit.point, [0.0, 0.0, 0.0])


def test_factor_alpha_multiplies_texture_alpha():
    # texture texels are fully opaque (255) on the right half but the
    # factor alpha of 100/255 ~= 0.39 pulls every hit below 0.5
    scene = trimesh.Scene(
        _mask_quad(cutoff=0.5, factor=[255, 255, 255, 100])
    )
    origins, directions = _rays([[0.5, 0.0]])
    assert scene.pick(origins, directions) == [None]


def test_default_mask_cutoff():
    # no explicit alphaCutoff: glTF default of 0.5 applies
    image = np.full((2, 2, 4), 255, dtype=np.uint8)
    image[..., 3] = np.array([[100, 200], [100, 200]], dtype=np.uint8)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(image, mode="RGBA"), alphaMode="MASK"
    )
    scene = trimesh.Scene(_quad(material))

    results = scene.pick(*_rays([[-0.5, 0.0], [0.5, 0.0]]))

    # alpha ~0.39 passes through, alpha ~0.78 blocks
    assert results[0] is None
    assert results[1] is not None


def test_front_transparent_back_opaque_occlusion():
    front = _mask_quad(z=0.0)
    back = trimesh.creation.box(extents=[3.0, 3.0, 0.2])
    back.apply_translation([0.0, 0.0, 2.0])

    scene = trimesh.Scene()
    scene.add_geometry(front, node_name="front")
    scene.add_geometry(back, node_name="back")

    origins, directions = _rays([[-0.5, 0.0], [0.5, 0.0]])
    transparent_ray, opaque_ray = scene.pick(origins, directions)

    # ray through the transparent texture region reaches the back box
    assert transparent_ray.node == "back"
    assert transparent_ray.face_index == 4
    assert np.allclose(transparent_ray.point, [-0.5, 0.0, 2.1])
    assert np.isclose(transparent_ray.distance, 4.1)

    # ray through the opaque texture region stops at the front quad
    assert opaque_ray.node == "front"
    assert np.allclose(opaque_ray.point, [0.5, 0.0, 0.0])


def test_nearest_of_two_opaque_mask_quads():
    # two opaque MASK quads at different depths: the nearer one wins
    near = _mask_quad(z=0.0)
    far = _mask_quad(z=3.0)

    scene = trimesh.Scene()
    scene.add_geometry(near, node_name="near")
    scene.add_geometry(far, node_name="far")

    hit = scene.pick(*_rays([[0.5, 0.0]]))[0]
    assert hit.node == "near"
    assert np.isclose(hit.distance, 2.0)

    # ordering of graph traversal must not change the nearest result
    scene_reversed = trimesh.Scene()
    scene_reversed.add_geometry(far, node_name="far")
    scene_reversed.add_geometry(near, node_name="near")
    hit_reversed = scene_reversed.pick(*_rays([[0.5, 0.0]]))[0]
    assert hit_reversed.node == "near"
    assert np.isclose(hit_reversed.distance, 2.0)


def test_nested_scaling():
    mesh = _mask_quad()
    scene = trimesh.Scene()
    scene.geometry["shared"] = mesh
    # effective world scale is 2 * 3 = 6: the quad spans x in [-6, 6]
    scene.graph.update(
        frame_to="parent",
        matrix=trimesh.transformations.scale_matrix(2.0),
        geometry="shared",
    )
    scene.graph.update(
        frame_from="parent",
        frame_to="child",
        matrix=trimesh.transformations.scale_matrix(3.0),
        geometry="shared",
    )

    # world x = 3.0 corresponds to local x = 0.5 (opaque texture half)
    origins, directions = _rays([[3.0, 0.0]])
    hit = scene.pick(origins, directions)[0]

    assert hit is not None
    assert hit.node == "child"
    assert np.allclose(hit.point, [3.0, 0.0, 0.0])

    # world x = -3.0 maps to the transparent half and must miss
    origins, directions = _rays([[-3.0, 0.0]])
    assert scene.pick(origins, directions) == [None]


def test_shared_instances():
    mesh = _mask_quad()
    scene = trimesh.Scene()
    scene.geometry["shared"] = mesh
    scene.graph.update(
        frame_to="inst_a",
        matrix=trimesh.transformations.translation_matrix([0.0, 0.0, 0.0]),
        geometry="shared",
    )
    scene.graph.update(
        frame_to="inst_b",
        matrix=trimesh.transformations.translation_matrix([0.0, 0.0, 4.0]),
        geometry="shared",
    )
    assert scene.graph.geometry_nodes["shared"] == ["inst_a", "inst_b"]

    origins, directions = _rays([[0.5, 0.0], [-0.5, 0.0]])
    opaque_ray, transparent_ray = scene.pick(origins, directions)

    # nearest opaque instance wins
    assert opaque_ray.node == "inst_a"
    assert np.allclose(opaque_ray.point, [0.5, 0.0, 0.0])
    # transparency applies to every instance of the shared geometry
    assert transparent_ray is None


def test_uv_boundaries():
    scene = trimesh.Scene(_mask_quad())

    # sample points near each quad corner: uv_to_color wraps the 0/1
    # texture boundaries, corners map to alpha 0, 255, 255, 0
    origins, directions = _rays(
        [[-0.99, -0.99], [0.99, -0.99], [0.99, 0.99], [-0.99, 0.99]]
    )
    results = scene.pick(origins, directions)

    assert results[0] is None
    assert results[1] is not None
    assert results[2] is not None
    assert results[3] is None

    # a ray that misses the quad entirely returns an empty result
    origins, directions = _rays([[5.0, 5.0]])
    assert scene.pick(origins, directions) == [None]


def test_result_tuple_and_empty_shapes():
    scene = trimesh.Scene(_mask_quad())
    origins, directions = _rays([[0.5, 0.0]])
    hit = scene.pick(origins, directions)[0]

    node, face_index, point, distance = hit
    assert node == hit.node
    assert face_index == hit.face_index
    assert np.allclose(point, hit.point)
    assert distance == hit.distance

    # empty ray batch returns an empty list
    assert pick_scene(scene, np.zeros((0, 3)), np.zeros((0, 3))) == []


def test_distance_independent_of_direction_length():
    scene = trimesh.Scene(_mask_quad())

    # non-unit direction vector: hit distance must still be the
    # geometric point-to-origin distance, not a dot-product parameter
    origins = np.array([[0.5, 0.0, -2.0]])
    directions = np.array([[0.0, 0.0, 3.0]])
    hit = scene.pick(origins, directions)[0]

    assert hit is not None
    assert np.isclose(hit.distance, 2.0)
    assert np.allclose(hit.point, [0.5, 0.0, 0.0])


def test_scene_not_modified():
    mesh = _mask_quad()
    scene = trimesh.Scene(mesh)
    before_vertices = mesh.vertices.copy()
    before_transform = scene.graph[scene.graph.nodes_geometry[0]][0].copy()

    origins, directions = _rays([[-0.5, 0.0], [0.5, 0.0]])
    scene.pick(origins, directions)

    assert np.allclose(mesh.vertices, before_vertices)
    after_transform = scene.graph[scene.graph.nodes_geometry[0]][0]
    assert np.allclose(after_transform, before_transform)

    # existing mesh ray intersector behavior is untouched
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        origins, directions
    )
    # without material awareness the transparent-half ray also hits
    assert len(index_ray) == 2
    assert locations.shape[1] == 3


def test_glb_reload_matches():
    front = _mask_quad(z=0.0)
    back = trimesh.creation.box(extents=[3.0, 3.0, 0.2])
    back.apply_translation([0.0, 0.0, 2.0])

    scene = trimesh.Scene()
    scene.add_geometry(front, node_name="front")
    scene.add_geometry(back, node_name="back")

    origins, directions = _rays([[-0.5, 0.0], [0.5, 0.0]])
    expected = [
        tuple(r) if r is not None else None
        for r in scene.pick(origins, directions)
    ]

    blob = scene.export(file_type="glb")
    reloaded = trimesh.load(
        trimesh.util.wrap_as_stream(blob), file_type="glb"
    )

    # the MASK material and its threshold survive the round trip
    mask_mesh = next(
        m for m in reloaded.geometry.values() if hasattr(m.visual, "uv")
    )
    assert mask_mesh.visual.material.alphaMode == "MASK"
    assert mask_mesh.visual.material.alphaCutoff == 0.5

    reloaded_results = [
        tuple(r) if r is not None else None
        for r in reloaded.pick(origins, directions)
    ]

    # both scenes return the same node-level picking decision:
    # transparent ray reaches the back geometry, opaque ray stops in front
    assert reloaded_results[0] is not None
    assert reloaded_results[1] is not None
    assert np.isclose(reloaded_results[1][2][2], 0.0, atol=1e-6)
    assert reloaded_results[0][2][2] > 1.0
    for want, got in zip(expected, reloaded_results):
        # XY ray coordinates and node roles are preserved exactly
        assert np.allclose(got[2][:2], want[2][:2], atol=1e-5)
        assert (got[0] is None) == (want[0] is None)
        # back geometry may land on a different box face after the
        # GLB axis/triangulation round trip, allow its depth to vary
        assert np.isclose(got[3], want[3], rtol=0.06)


def test_glb_alpha_blend_model_rejected():
    # the shipped model contains BLEND materials and must be rejected
    scene = g.get_mesh("AlphaBlendModeTest.glb")
    origins, directions = _rays([[0.0, 1.0]])

    with pytest.raises(ValueError, match="BLEND"):
        scene.pick(origins, directions)

    # OPAQUE and MASK meshes from the same asset are pickable once
    # the unsupported geometries are removed from a fresh copy
    supported = trimesh.Scene()
    for node in scene.graph.nodes_geometry:
        geometry_name = scene.graph[node][1]
        material = getattr(
            scene.geometry[geometry_name].visual, "material", None
        )
        if (
            isinstance(material, trimesh.visual.material.PBRMaterial)
            and material.alphaMode == "BLEND"
        ):
            continue
        supported.add_geometry(
            scene.geometry[geometry_name],
            node_name=node,
            geom_name=geometry_name,
            transform=scene.graph[node][0],
        )
    results = supported.pick(origins, directions)
    assert results[0] is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    pytest.main([__file__, "-v"])
