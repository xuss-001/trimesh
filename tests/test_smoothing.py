try:
    from . import generic as g
except BaseException:
    import generic as g


def explicit_laplacian_calculation(mesh, equal_weight=True, pinned_vertices=None):
    """
    Exact copy of non-sparse calculation of laplacian, for correctness testing.

    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neighbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors

    # if a node is pinned, it will average his coordinates by himself
    # in practice it will not move
    if pinned_vertices is not None:
        for i in pinned_vertices:
            neighbors[i] = [i]

    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(g.np.ndarray)

    # stack neighbors to 1D arrays
    col = g.np.concatenate(neighbors)
    row = g.np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = g.np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = g.np.ones(3)
        # the distance from verticesex to neighbors
        norms = [
            1.0
            / g.np.maximum(
                1e-6, g.np.sqrt(g.np.dot((vertices[i] - vertices[n]) ** 2, ones))
            )
            for i, n in enumerate(neighbors)
        ]
        # normalize group and stack into single array
        data = g.np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = g.trimesh.graph.coo_matrix((data, (row, col)), shape=[len(vertices)] * 2)

    return matrix


class SmoothTest(g.unittest.TestCase):
    def planar_grid(self, side=5, noise=None, seed=0):
        """
        Build an open planar triangle grid with ``side * side`` vertices.
        """
        xs, ys = g.np.meshgrid(g.np.arange(side), g.np.arange(side))
        vertices = g.np.column_stack(
            [xs.ravel(), ys.ravel(), g.np.zeros(side * side)]
        ).astype(float)
        faces = []
        for i in range(side - 1):
            for j in range(side - 1):
                a = i * side + j
                b = i * side + j + 1
                c = (i + 1) * side + j
                d = (i + 1) * side + j + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        mesh = g.trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
        if noise is not None:
            rng = g.np.random.default_rng(seed)
            mesh.vertices += rng.normal(scale=noise, size=mesh.vertices.shape)
        return mesh

    def test_feature_preserving_pins_open_boundary(self):
        mesh = self.planar_grid(side=5)
        pinned = g.trimesh.smoothing.get_pinned_vertices(
            mesh, feature_angle=g.np.pi / 2
        )

        # every coplanar interior edge is smooth, only boundary vertices
        # are pinned
        boundary = {
            index
            for index in range(len(mesh.vertices))
            if index < 5 or index >= 20 or index % 5 in (0, 4)
        }
        assert set(pinned.tolist()) == boundary

        before = mesh.vertices.copy()
        g.trimesh.smoothing.filter_feature_preserving(
            mesh, feature_angle=g.np.pi / 2, iterations=5
        )
        assert g.np.allclose(mesh.vertices[list(boundary)], before[list(boundary)])

    def test_feature_preserving_reduces_surface_noise(self):
        mesh = self.planar_grid(side=7, noise=0.1)
        original = mesh.vertices.copy()

        interior = g.np.ones(len(mesh.vertices), dtype=bool)
        boundary = g.trimesh.smoothing.get_pinned_vertices(
            mesh, feature_angle=g.np.pi / 2
        )
        interior[boundary] = False

        g.trimesh.smoothing.filter_feature_preserving(
            mesh, feature_angle=g.np.pi / 2, lamb=0.5, iterations=20
        )

        # noise on the free interior vertices is reduced towards z == 0
        before = g.np.abs(original[interior, 2]).mean()
        after = g.np.abs(mesh.vertices[interior, 2]).mean()
        assert after < before * 0.5
        # the protected boundary never drifts
        assert g.np.allclose(mesh.vertices[boundary], original[boundary])

    def test_feature_preserving_keeps_hard_crease(self):
        # start from a planar grid and fold one half about the middle row,
        # producing an interior crease whose endpoints are not on the
        # open boundary
        mesh = self.planar_grid(side=5)
        fold = g.np.pi / 3
        vertices = mesh.vertices.view(g.np.ndarray)
        # fold the right-hand wing about the y == 2 grid line so that
        # the crease edges align with edges of the triangulation
        right = vertices[:, 0] >= 2.0
        distance = vertices[right, 0] - 2.0
        vertices[right, 0] = 2.0 + distance * g.np.cos(fold)
        vertices[right, 2] = distance * g.np.sin(fold)
        mesh = g.trimesh.Trimesh(
            vertices=vertices,
            faces=mesh.faces.view(g.np.ndarray),
            process=False,
        )
        crease_angle = mesh.face_adjacency_angles.max()
        crease_edges = mesh.face_adjacency_edges[
            g.np.abs(mesh.face_adjacency_angles - crease_angle) < 1e-9
        ]
        assert len(crease_edges) > 0
        crease_vertices = g.np.unique(crease_edges)

        # at the crease angle itself, the crease endpoints must be pinned
        pinned = g.trimesh.smoothing.get_pinned_vertices(
            mesh, feature_angle=crease_angle
        )
        assert set(crease_vertices.tolist()).issubset(set(pinned.tolist()))

        # a looser threshold does not treat the fold as a feature
        pinned_loose = g.trimesh.smoothing.get_pinned_vertices(
            mesh, feature_angle=crease_angle + 0.1
        )
        boundary = set(
            g.trimesh.smoothing.get_pinned_vertices(
                mesh, feature_angle=g.np.pi
            ).tolist()
        )
        interior_crease = [
            vertex
            for vertex in crease_vertices.tolist()
            if vertex not in boundary
        ]
        assert len(interior_crease) > 0
        assert not set(interior_crease).issubset(set(pinned_loose.tolist()))

        before = mesh.vertices.copy()
        g.trimesh.smoothing.filter_feature_preserving(
            mesh, feature_angle=crease_angle, iterations=5
        )
        assert g.np.allclose(
            mesh.vertices[crease_vertices], before[crease_vertices]
        )

    def test_feature_preserving_non_manifold_edge(self):
        # three triangles fanning out of one shared edge: the shared edge
        # is used by more than two faces and is therefore non-manifold
        vertices = g.np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            dtype=float,
        )
        faces = g.np.array([[0, 1, 2], [0, 1, 3], [1, 0, 4]])
        mesh = g.trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )

        pinned = g.trimesh.smoothing.get_pinned_vertices(mesh)
        assert {0, 1}.issubset(set(pinned.tolist()))

        before = mesh.vertices.copy()
        g.trimesh.smoothing.filter_feature_preserving(mesh, iterations=5)
        assert g.np.allclose(mesh.vertices[[0, 1]], before[[0, 1]])

    def test_feature_preserving_isolated_vertex(self):
        mesh = self.planar_grid(side=4)
        mesh.vertices = g.np.vstack(
            (mesh.vertices, [[7.0, 7.0, 7.0]])
        )
        isolated_index = len(mesh.vertices) - 1

        pinned = g.trimesh.smoothing.get_pinned_vertices(mesh)
        assert isolated_index in set(pinned.tolist())

        # the operator must cover every vertex, including the isolated one
        operator = g.trimesh.smoothing.laplacian_calculation(mesh)
        assert operator.shape == (len(mesh.vertices), len(mesh.vertices))
        assert not g.np.isnan(operator.toarray()).any()
        weighted = g.trimesh.smoothing.laplacian_calculation(
            mesh, equal_weight=False
        )
        assert weighted.shape == operator.shape
        assert not g.np.isnan(weighted.toarray()).any()

        before = mesh.vertices.copy()
        g.trimesh.smoothing.filter_feature_preserving(mesh, iterations=5)
        assert g.np.allclose(
            mesh.vertices[isolated_index], before[isolated_index]
        )

    def test_feature_preserving_translation_invariant(self):
        offset = g.np.array([10.0, -20.0, 30.0])
        centered = self.planar_grid(side=5, noise=0.1)
        translated = centered.copy()
        translated.apply_translation(offset)

        g.trimesh.smoothing.filter_feature_preserving(
            centered, iterations=10
        )
        g.trimesh.smoothing.filter_feature_preserving(
            translated, iterations=10
        )

        assert g.np.allclose(
            translated.vertices - offset, centered.vertices, atol=1e-12
        )

    def test_feature_preserving_keeps_topology(self):
        mesh = self.planar_grid(side=5, noise=0.1)
        vertex_count = len(mesh.vertices)
        faces_before = mesh.faces.copy()

        g.trimesh.smoothing.filter_feature_preserving(mesh, iterations=5)

        assert len(mesh.vertices) == vertex_count
        assert g.np.array_equal(mesh.faces, faces_before)

    def test_feature_preserving_invalid_angle(self):
        mesh = self.planar_grid(side=3)
        with self.assertRaises(ValueError):
            g.trimesh.smoothing.get_pinned_vertices(mesh, feature_angle=-1.0)

    def test_laplacian_volume_constraint_is_translation_invariant(self):
        centered = g.trimesh.creation.icosahedron()
        translated = centered.copy()
        offset = g.np.array([10.0, 20.0, 30.0])
        translated.apply_translation(offset)

        g.trimesh.smoothing.filter_laplacian(centered, iterations=1)
        g.trimesh.smoothing.filter_laplacian(translated, iterations=1)

        assert g.np.allclose(translated.vertices - offset, centered.vertices)

    def test_laplacian_volume_constraint_pinned_drift_is_position_independent(self):
        # a uniform volume rescale cannot hold pinned vertices exactly:
        # each vertex drifts by (prod(scales) - 1) * distance-from-center.
        # that residual must depend only on the geometry, never on where
        # the mesh sits in space.
        drifts = []
        for offset in ([10.0, 20.0, 30.0], [100.0, 200.0, 300.0]):
            m = g.trimesh.creation.icosphere(subdivisions=2)
            m.apply_translation(offset)
            pinned = g.np.argsort(m.vertices[:, 0])[-12:].tolist()
            start = m.vertices[pinned].copy()
            operator = g.trimesh.smoothing.laplacian_calculation(
                m, pinned_vertices=pinned
            )
            g.trimesh.smoothing.filter_laplacian(
                m, iterations=10, laplacian_operator=operator
            )
            drifts.append(g.np.linalg.norm(m.vertices[pinned] - start, axis=1).max())

        assert g.np.allclose(drifts[0], drifts[1])

    def test_laplacian_calculation(self):
        m = g.trimesh.creation.icosahedron()
        m.vertices, m.faces = g.trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1)

        explicit_laplacian = explicit_laplacian_calculation(m)
        laplacian = g.trimesh.smoothing.laplacian_calculation(m)

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(m, equal_weight=False)
        laplacian = g.trimesh.smoothing.laplacian_calculation(m, equal_weight=False)

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(m, pinned_vertices=[0, 1, 4])
        laplacian = g.trimesh.smoothing.laplacian_calculation(
            m, pinned_vertices=[0, 1, 4]
        )

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(
            m, equal_weight=False, pinned_vertices=[0, 1, 4]
        )
        laplacian = g.trimesh.smoothing.laplacian_calculation(
            m, equal_weight=False, pinned_vertices=[0, 1, 4]
        )

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

    def test_smooth(self):
        """
        Load a collada scene with pycollada.
        """
        m = g.trimesh.creation.icosahedron()
        m.vertices, m.faces = g.trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        assert m.is_volume

        # Equal Weights
        lap = g.trimesh.smoothing.laplacian_calculation(mesh=m, equal_weight=True)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        # umbrella Weights
        lap = g.trimesh.smoothing.laplacian_calculation(m, equal_weight=False)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
