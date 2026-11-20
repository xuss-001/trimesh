"""
ray_pick.py
-----------

Scene-level ray picking which takes the alpha information of
glTF-style materials into account.

Rays are intersected against every geometry instance of a
`trimesh.Scene`. Hits on materials using `alphaMode="MASK"` are
discarded when the sampled base color alpha, multiplied with the
base color factor alpha, falls below the material cutoff, so the
ray passes through transparent texture regions. `OPAQUE` materials
are always solid. `BLEND` materials are not supported and raise a
`ValueError`, since alpha blending has no binary hit rule.
"""

import numpy as np

from .. import transformations, triangles as triangles_mod, util
from ..typed import ArrayLike
from ..visual import color as color_mod
from ..visual import material as material_mod
from .ray_triangle import RayMeshIntersector

# default glTF alpha cutoff when `alphaMode="MASK"` is set
# but no explicit `alphaCutoff` is stored
_DEFAULT_ALPHA_CUTOFF = 0.5


class RayPickResult:
    """
    A single successful alpha-aware ray pick.

    Attributes
    ------------
    node : hashable
      Instance node of the scene graph that was hit.
    geometry : str
      Name of the geometry referenced by the node.
    face_index : int
      Index of the hit triangle inside the source mesh.
    point : (3,) float
      Intersection point in world coordinates.
    distance : float
      Geometric (Euclidean) distance from the ray origin
      to `point`, independent of direction length.
    """

    __slots__ = ("node", "geometry", "face_index", "point", "distance")

    def __init__(self, node, geometry, face_index, point, distance):
        self.node = node
        self.geometry = geometry
        self.face_index = int(face_index)
        self.point = point
        self.distance = float(distance)

    def __iter__(self):
        # behave like the documented tuple
        # (node, face_index, point, distance)
        yield self.node
        yield self.face_index
        yield self.point
        yield self.distance

    def __eq__(self, other):
        if isinstance(other, RayPickResult):
            return (
                self.node == other.node
                and self.geometry == other.geometry
                and self.face_index == other.face_index
                and np.allclose(self.point, other.point)
                and np.isclose(self.distance, other.distance)
            )
        if isinstance(other, tuple):
            return (
                (self.node, self.face_index) == other[:2]
                and np.allclose(self.point, other[2])
                and np.isclose(self.distance, other[3])
            )
        return NotImplemented

    def __hash__(self):
        return hash((self.node, self.geometry, self.face_index))

    def __repr__(self):
        return (
            f"RayPickResult(node={self.node!r}, geometry={self.geometry!r}, "
            f"face_index={self.face_index}, point={self.point.tolist()}, "
            f"distance={self.distance})"
        )


def _material_alpha_rule(material):
    """
    Resolve a material into the alpha handling used for picking.

    Returns
    ---------
    rule : tuple
      `(opaque, cutoff, factor_alpha)` where `opaque` marks
      materials that always block rays and `cutoff` is the
      alpha threshold below which MASK hits pass through.
    """
    if not isinstance(material, material_mod.PBRMaterial):
        # non-PBR visuals (vertex colors, simple OBJ materials)
        # do not carry glTF alpha semantics, treat as opaque
        return True, 1.0, 1.0

    alpha_mode = material.alphaMode
    if alpha_mode == "BLEND":
        raise ValueError(
            "alphaMode `BLEND` is not supported for alpha-aware ray "
            "picking: only `OPAQUE` and `MASK` materials define a "
            "binary hit rule."
        )

    factor = material.baseColorFactor
    if factor is None:
        factor_alpha = 1.0
    else:
        factor_alpha = float(factor[3]) / 255.0

    if alpha_mode != "MASK":
        # OPAQUE (including the glTF default where alphaMode is None)
        return True, 1.0, factor_alpha

    cutoff = material.alphaCutoff
    if cutoff is None:
        cutoff = _DEFAULT_ALPHA_CUTOFF
    return False, float(cutoff), factor_alpha


def _hit_uv(mesh, face_index, points):
    """
    Interpolate the texture UV coordinates of mesh-local hit points.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Mesh carrying per-vertex UV coordinates.
    face_index : (h,) int
      Hit triangle indices.
    points : (h, 3) float
      Hit points in mesh-local coordinates.

    Returns
    ---------
    uv : (h, 2) float
      Interpolated UV coordinates.
    """
    faces = mesh.faces[face_index]
    triangles = mesh.triangles[face_index]
    barycentric = triangles_mod.points_to_barycentric(triangles, points)
    return np.einsum(
        "ij,ijk->ik",
        barycentric,
        mesh.visual.uv[faces],
    )


def pick_scene(
    scene,
    ray_origins: ArrayLike,
    ray_directions: ArrayLike,
):
    """
    Cast rays at a scene and return the nearest material-aware hit.

    Transparent regions of `alphaMode="MASK"` textures do not block
    selection: the sampled texture alpha is multiplied with the
    base color factor alpha and the hit passes through when the
    result is below the material `alphaCutoff`. `OPAQUE` materials
    always block. `BLEND` materials raise a `ValueError`.

    The scene is never modified.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene containing the geometry instances to pick.
    ray_origins : (n, 3) float
      Ray origin points in world coordinates.
    ray_directions : (n, 3) float
      Ray direction vectors in world coordinates.

    Returns
    ----------
    results : list
      One `RayPickResult` per ray, or `None` for rays that did not
      hit any opaque region.
    """
    ray_origins = np.asanyarray(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)
    if not util.is_shape(ray_origins, (-1, 3)) or not util.is_shape(
        ray_directions, (-1, 3)
    ):
        raise ValueError("ray origins and directions must be (n, 3)")
    if len(ray_origins) != len(ray_directions):
        raise ValueError("ray origins and directions must have the same length")

    # fail up-front if any instance uses an unsupported material
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if (
            isinstance(material, material_mod.PBRMaterial)
            and material.alphaMode == "BLEND"
        ):
            raise ValueError(
                "alphaMode `BLEND` is not supported for alpha-aware ray "
                "picking: only `OPAQUE` and `MASK` materials define a "
                "binary hit rule."
            )

    # per-ray running nearest valid hit
    nearest_node = [None] * len(ray_origins)
    nearest_geometry = [None] * len(ray_origins)
    nearest_face = np.full(len(ray_origins), -1, dtype=np.int64)
    nearest_point = np.zeros((len(ray_origins), 3), dtype=np.float64)
    nearest_distance = np.full(len(ray_origins), np.inf, dtype=np.float64)

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not hasattr(mesh, "triangles"):
            continue

        # intersect in the mesh-local frame so face indices and UVs
        # refer directly to the source mesh; the scene stays untouched
        inverse = transformations.inverse_matrix(transform)
        local_origins = transformations.transform_points(ray_origins, inverse)
        local_directions = transformations.transform_points(
            ray_directions, inverse, translate=False
        )

        # reuse the mesh's existing intersector when available so the
        # cached r-tree is shared with the legacy ray API; shared
        # geometry instances therefore pay the build cost only once
        intersector = getattr(mesh, "ray", None)
        if intersector is None or not hasattr(intersector, "intersects_id"):
            intersector = RayMeshIntersector(mesh)
        index_tri, index_ray, local_points = intersector.intersects_id(
            local_origins,
            local_directions,
            return_locations=True,
            multiple_hits=True,
        )
        if len(index_tri) == 0:
            continue

        opaque, cutoff, factor_alpha = _material_alpha_rule(
            getattr(mesh.visual, "material", None)
        )
        if not opaque:
            valid = _mask_opaque_faces(
                mesh, factor_alpha, cutoff, index_tri, local_points
            )
            index_tri = index_tri[valid]
            index_ray = index_ray[valid]
            local_points = local_points[valid]
            if len(index_tri) == 0:
                continue

        world_points = transformations.transform_points(local_points, transform)
        vectors = world_points - ray_origins[index_ray]
        # the intersectors only return forward hits; take the geometric
        # length so distance is independent of ray direction length and
        # agrees across the native and embree ray backends (embree
        # unitizes directions internally)
        distance = np.sqrt(util.diagonal_dot(vectors, vectors))

        # keep the nearest valid hit of each ray
        closer = distance < nearest_distance[index_ray]
        if not closer.any():
            continue
        index_tri = index_tri[closer]
        index_ray = index_ray[closer]
        world_points = world_points[closer]
        distance = distance[closer]

        nearest_face[index_ray] = index_tri
        nearest_point[index_ray] = world_points
        nearest_distance[index_ray] = distance
        for ray_index, face_index, point, ray_distance in zip(
            index_ray, index_tri, world_points, distance
        ):
            nearest_node[ray_index] = node_name
            nearest_geometry[ray_index] = geometry_name

    results = []
    for ray_index in range(len(ray_origins)):
        if nearest_node[ray_index] is None:
            results.append(None)
        else:
            results.append(
                RayPickResult(
                    node=nearest_node[ray_index],
                    geometry=nearest_geometry[ray_index],
                    face_index=nearest_face[ray_index],
                    point=nearest_point[ray_index],
                    distance=nearest_distance[ray_index],
                )
            )
    return results


def _mask_opaque_faces(mesh, factor_alpha, cutoff, index_tri, local_points):
    """
    Boolean mask of MASK-material hits whose sampled alpha is opaque.
    """
    visual = mesh.visual
    image = getattr(visual.material, "baseColorTexture", None)
    if not hasattr(visual, "uv") or visual.uv is None or image is None:
        # no texture: the base color factor alpha alone decides
        alpha = np.full(len(index_tri), factor_alpha, dtype=np.float64)
    else:
        uv = _hit_uv(mesh, index_tri, local_points)
        # reuse the same texture sampling as the rest of trimesh
        texture_alpha = (
            color_mod.uv_to_color(uv, image)[:, 3].astype(np.float64) / 255.0
        )
        alpha = texture_alpha * factor_alpha
    return alpha >= cutoff


__all__ = ["RayPickResult", "pick_scene"]
