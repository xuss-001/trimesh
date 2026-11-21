import numpy as np
import pytest

import trimesh

PIL = pytest.importorskip("PIL.Image")


def _quad(alpha, mode="OPAQUE", cutoff=None, factor=None, uvs=None):
    alpha = np.asanyarray(alpha, dtype=np.uint8)
    image = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    image[:] = [255, 0, 0, 255]
    image[..., 3] = alpha
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=PIL.fromarray(image),
        baseColorFactor=factor,
        alphaMode=mode,
        alphaCutoff=cutoff,
    )
    if uvs is None:
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return trimesh.Trimesh(
        vertices=np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        visual=trimesh.visual.texture.TextureVisuals(
            uv=np.array(uvs), material=material
        ),
        process=False,
    )


def _translation(x=0.0, y=0.0, z=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _scale(value):
    matrix = np.eye(4)
    matrix[:3, :3] *= value
    return matrix


def test_ray_pick_nested_scaling():
    mesh = _quad(np.full((2, 2), 255))
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh, node_name="child", geom_name="quad", parent_node_name="parent"
    )
    scene.graph.update(frame_from="world", frame_to="parent", matrix=_scale(2.0))
    scene.graph.update(
        frame_from="parent", frame_to="child", matrix=_translation(z=2.0)
    )

    hit = scene.ray_pick([0.7, 0.4, 10.0], [0.0, 0.0, -1.0])

    assert hit is not None
    assert hit.node_name == "child"
    assert hit.face_index == 0
    assert np.allclose(hit.point, [0.7, 0.4, 4.0])
    assert np.isclose(hit.distance, 6.0)


def test_ray_pick_shared_geometry_instances():
    mesh = _quad(np.full((2, 2), 255))
    scene = trimesh.Scene({"shared": mesh})
    scene.graph.update(
        frame_to="left", matrix=_translation(x=-2.0), geometry="shared"
    )
    scene.graph.update(
        frame_to="right", matrix=_translation(x=2.0), geometry="shared"
    )

    assert len(scene.geometry) == 1
    assert scene.graph["left"][1] == scene.graph["right"][1] == "shared"

    right = scene.ray_pick([2.1, 0.1, 5.0], [0.0, 0.0, -1.0])
    left = scene.ray_pick([-1.9, 0.1, 5.0], [0.0, 0.0, -1.0])

    assert right is not None and right.node_name == "right"
    assert left is not None and left.node_name == "left"
    assert np.allclose(right.point, [2.1, 0.1, 0.0])
    assert np.allclose(left.point, [-1.9, 0.1, 0.0])


def test_ray_pick_uv_boundaries():
    # the sampled pixels are: (u=0, v=1) opaque, (u=1, v=0) transparent
    alpha = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    mesh = _quad(alpha, mode="MASK", cutoff=0.5)
    scene = trimesh.Scene({"quad": mesh})

    # just outside the 0.0/1.0 edges still sample the nearest edge pixel
    hits = {
        "00": scene.ray_pick([-0.48, -0.48, 5.0], [0.0, 0.0, -1.0]),
        "01": scene.ray_pick([-0.48, 0.48, 5.0], [0.0, 0.0, -1.0]),
        "10": scene.ray_pick([0.48, -0.48, 5.0], [0.0, 0.0, -1.0]),
        "11": scene.ray_pick([0.48, 0.48, 5.0], [0.0, 0.0, -1.0]),
    }
    assert hits["00"] is None
    assert hits["01"] is not None
    assert hits["10"] is None
    assert hits["11"] is not None

    # exact 0.0 and 1.0 UV values in the interior of the triangle
    uv_zero = _quad(alpha, mode="MASK", cutoff=0.5, uvs=np.zeros((4, 2)))
    uv_one = _quad(alpha, mode="MASK", cutoff=0.5, uvs=np.ones((4, 2)))
    assert (
        trimesh.Scene({"quad": uv_zero}).ray_pick(
            [0.1, 0.1, 5.0], [0.0, 0.0, -1.0]
        )
        is None
    )
    assert (
        trimesh.Scene({"quad": uv_one}).ray_pick(
            [0.1, 0.1, 5.0], [0.0, 0.0, -1.0]
        )
        is not None
    )


def test_ray_pick_alpha_factor_and_occlusion():
    front_mask = _quad(
        np.zeros((2, 2)), mode="MASK", cutoff=0.5, factor=[1, 1, 1, 0.4]
    )
    back_opaque = _quad(np.zeros((2, 2)), mode="OPAQUE")
    scene = trimesh.Scene(
        {
            "front": front_mask,
            "back": back_opaque,
        }
    )
    scene.graph.update(frame_to="front", matrix=_translation(z=1.0), geometry="front")

    before = {name: mesh.vertices.copy() for name, mesh in scene.geometry.items()}
    hit = scene.ray_pick([0.1, 0.1, 5.0], [0.0, 0.0, -1.0])

    assert hit is not None and hit.node_name == "back"
    assert np.allclose(hit.point, [0.1, 0.1, 0.0])
    assert np.isclose(hit.distance, 5.0)

    # picking must not modify scene geometry
    for name, mesh in scene.geometry.items():
        assert np.array_equal(mesh.vertices, before[name])

    # OPAQUE ignores texture alpha and is hit even where alpha is zero
    opaque = _quad(np.zeros((2, 2)), mode="OPAQUE")
    assert trimesh.Scene({"opaque": opaque}).ray_pick(
        [0.1, 0.1, 5.0], [0.0, 0.0, -1.0]
    ) is not None


def test_ray_pick_blend_raises():
    mesh = _quad(np.full((2, 2), 255), mode="BLEND")
    scene = trimesh.Scene({"blend": mesh})

    with pytest.raises(ValueError, match="BLEND"):
        scene.ray_pick([0.1, 0.1, 5.0], [0.0, 0.0, -1.0])


def test_ray_pick_glb_roundtrip():
    alpha = np.zeros((2, 2), dtype=np.uint8)
    alpha[0, 1] = 255
    mesh = _quad(alpha, mode="MASK", cutoff=0.5)
    scene = trimesh.Scene({"quad": mesh})

    original = scene.ray_pick([0.25, 0.25, 5.0], [0.0, 0.0, -1.0])
    assert original is not None

    reloaded = trimesh.load(
        trimesh.util.wrap_as_stream(scene.export(file_type="glb")), file_type="glb"
    )
    material = reloaded.geometry["quad"].visual.material
    assert material.alphaMode == "MASK"
    assert material.alphaCutoff == 0.5

    hit = reloaded.ray_pick([0.25, 0.25, 5.0], [0.0, 0.0, -1.0])
    assert hit is not None
    assert hit.node_name == original.node_name
    assert hit.face_index == original.face_index
    assert np.allclose(hit.point, original.point)
    assert np.isclose(hit.distance, original.distance)

    # the texture alpha is transparent at this UV in both scenes
    assert reloaded.ray_pick([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]) is None


def test_old_ray_interface_unchanged():
    mesh = _quad(np.zeros((2, 2)), mode="MASK", cutoff=0.5)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        [[0.1, 0.1, 5.0]], [[0.0, 0.0, -1.0]]
    )

    # the mesh-level ray query still reports the geometric hit
    assert len(locations) == 1
    assert len(index_ray) == 1
    assert len(index_tri) == 1
