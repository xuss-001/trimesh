try:
    from . import generic as g
except BaseException:
    import generic as g


class RegistrationTest(g.unittest.TestCase):
    def test_procrustes(self):
        # every combination of possible boolean options
        # a_flip and a_scale are apply-to-test-data
        opt = list(g.itertools.combinations([True, False] * 6, 6))
        # generate deterministic but random-ish transforms
        matrices = iter(g.random_transforms(count=len(opt)))

        for reflection, translation, scale, a_flip, a_scale, weight in opt:
            # create random points in space
            points_a = (g.random((1000, 3)) - 0.5) * 1000
            # get a random transform from the iterator
            matrix = next(matrices)
            # apply a flip (reflection) to test data
            if a_flip:
                matrix = g.np.dot(
                    matrix,
                    g.trimesh.transformations.reflection_matrix(
                        point=[0, 0, 0], normal=[0, 1, 0]
                    ),
                )
            # apply scale to test data
            if a_scale:
                matrix = g.np.dot(
                    matrix, g.trimesh.transformations.scale_matrix(0.1235234)
                )
            # apply transform to points A
            points_b = g.trimesh.transform_points(points_a, matrix)

            # weight points or not
            if weight:
                weights = g.np.zeros(len(points_a))
                weights[::3] = 1.0
            else:
                weights = None

            # run the solver
            (matrixN, transformed, cost) = g.trimesh.registration.procrustes(
                points_a,
                points_b,
                reflection=reflection,
                translation=translation,
                scale=scale,
                weights=weights,
            )
            # if we're not weighting the results
            # should be identical with None vs all-ones
            (matrixN_C, transformed_C, cost_C) = g.trimesh.registration.procrustes(
                points_a,
                points_b,
                reflection=reflection,
                translation=translation,
                scale=scale,
                weights=g.np.ones(len(points_a)),
            )
            if not weight:
                # no weights so everything should be identical
                assert g.np.allclose(matrixN, matrixN_C)
                assert g.np.allclose(transformed_C, transformed)
                assert g.np.isclose(cost, cost_C)

            # the points should be identical if the function
            # was allowed to translate in space
            # and there were no weights, scaling, or reflection
            identical = (
                translation
                and (not weight)
                and (not a_flip or reflection)
                and (not a_scale or scale)
            )

            if identical:
                assert cost < 0.001
                # it should have found the matrix we used
                assert g.np.allclose(matrixN, matrix)

            # if reflection is not allowed, the determinant
            # should always be close to 1.0 for the rotation
            det = g.np.linalg.det(matrixN[:3, :3])
            # reflection not allowed
            if not reflection:
                # determinant should always be positive
                assert det > 1e-8
                if not scale:
                    # no reflection or scale means determinant
                    # of 1.0
                    assert g.np.isclose(det, 1.0)

            # scaling not allowed
            if not scale:
                # allowed to be -1.0 or 1.0
                assert g.np.isclose(g.np.abs(det), 1.0)

            # points have a flip applied
            # procrustes is allowed to use reflection
            # and there is no scaling in the matrix
            if a_flip and reflection and not scale:
                assert g.np.isclose(det, -1.0), det

    def test_procrustes_float_weights(self):
        from trimesh.registration import procrustes

        # create two meshes that are a box and some arbitrary other stuff
        a = g.trimesh.creation.box() + g.get_mesh("featuretype.STL")
        b = g.trimesh.creation.box() + g.get_mesh("rabbit.obj")

        # mangle the larger mesh to have the same number of vertices
        a.vertices = a.vertices[: len(b.vertices)]
        a.faces = a.faces[(a.faces < len(b.vertices)).all(axis=1)]
        assert a.vertices.shape == b.vertices.shape
        # the box should match exactly
        assert g.np.allclose(a.vertices[:8], b.vertices[:8])

        # move `b` to an arbitrary transform
        transform = g.trimesh.transformations.rotation_matrix(
            0.456456, [0.14586, 2.0, 0.8946513], [100.1456, 51456.123, 447.2]
        )
        b.apply_transform(transform)

        # create weights that just consider the box
        weights = g.np.zeros(len(a.vertices))
        weights[:8] = 1.0

        # the easy case with boolean weights that just consider the box
        register, _, _ = procrustes(a.vertices, b.vertices, weights=weights)
        assert g.np.allclose(register, transform)

        # now try it with floating point weights that should still match exactly
        weights[:8] = g.np.arange(8) / 7.0
        register, _, _ = procrustes(a.vertices, b.vertices, weights=weights)
        assert g.np.allclose(register, transform)

        # no weights shouldn't match at all
        register, _, _ = procrustes(a.vertices, b.vertices)
        assert not g.np.allclose(register, transform)

    def test_icp_mesh(self):
        # see if ICP alignment works with meshes
        m = g.trimesh.creation.box()
        X = m.sample(10)
        X = X + [0.1, 0.1, 0.1]
        _matrix, _transformed, cost = g.trimesh.registration.icp(X, m, scale=False)
        assert cost < 0.01

    def test_mesh_other_no_reflection(self):
        # regression test for #2482: `mesh_other` seeds ICP with sign-flip
        # initial transforms, half of which are reflections. With
        # `reflection=False` those reflected seeds could still win and return
        # a transform with a negative determinant (i.e. a reflection).
        a = g.get_mesh("featuretype.STL")

        # build a mirrored, inside-out copy of `a`
        b = a.copy()
        b.apply_transform(g.trimesh.transformations.rotation_matrix(0.9, [0.3, 0.6, 0.2]))
        b.apply_transform(g.np.diag([-1.0, -1.0, 1.0, 1.0]))
        b.vertices[:, 1] *= -1.0

        # with reflection disabled the result must be a proper rotation
        for seed in range(3):
            matrix, _cost = g.trimesh.registration.mesh_other(
                a, b, samples=500, reflection=False, scale=False, seed=seed
            )
            assert g.np.linalg.det(matrix[:3, :3]) > 0

        # with reflection enabled it is still allowed to use a reflection
        # to fit the mirrored mesh
        matrix, _cost = g.trimesh.registration.mesh_other(
            a, b, samples=500, reflection=True, scale=False, seed=0
        )
        assert g.np.linalg.det(matrix[:3, :3]) < 0

    def test_icp_points(self):
        # see if ICP alignment works with point clouds
        # create random points in space
        points_a = (g.random((1000, 3)) - 0.5) * 1000
        # create a random transform
        # matrix = g.trimesh.transformations.random_rotation_matrix()
        # create a small transform
        # ICP will not work at all with large transforms
        matrix = g.trimesh.transformations.rotation_matrix(g.np.radians(1.0), [0, 0, 1])

        # take a few randomly chosen points and make
        # sure the order is permutated
        index = g.np.random.default_rng(seed=0).choice(g.np.arange(len(points_a)), 20)
        # transform and apply index
        points_b = g.trimesh.transform_points(points_a[index], matrix)
        # tun the solver
        matrixN, transformed, cost = g.trimesh.registration.icp(points_b, points_a)
        assert cost < 1e-3
        assert g.np.allclose(matrix, g.np.linalg.inv(matrixN))
        assert g.np.allclose(transformed, points_a[index])

    def test_icp_filter_partial_overlap_outliers(self):
        # ICP with correspondence filtering should be robust to
        # partially overlapping scans containing outliers
        rng = g.np.random.default_rng(seed=0)

        # source cloud in a unit cube
        points_a = rng.random((400, 3))
        # a small transform recoverable from an identity initial guess
        matrix = g.trimesh.transformations.rotation_matrix(
            g.np.radians(2.0), [0, 0, 1]
        )
        matrix[:3, 3] = [0.02, -0.01, 0.01]

        # target: only 70% of the source points have a true match
        inliers = g.trimesh.transform_points(points_a[:280], matrix)
        # add outliers far away from the overlap region
        outliers = rng.random((40, 3)) * 10 + 20
        points_b = g.np.vstack((inliers, outliers))

        # the registration must not modify its inputs
        a_original = points_a.copy()
        b_original = points_b.copy()

        matrixN, transformed, cost = g.trimesh.registration.icp(
            points_a,
            points_b,
            scale=False,
            max_correspondence_distance=0.5,
            keep_ratio=0.7,
        )

        assert g.np.allclose(points_a, a_original)
        assert g.np.allclose(points_b, b_original)

        # the recovered transform should be close to the true one
        assert g.np.allclose(matrixN, matrix, atol=1e-6)
        # the returned transform is applied to all source points
        assert transformed.shape == points_a.shape
        assert g.np.allclose(
            transformed, g.trimesh.transform_points(points_a, matrixN)
        )
        # cost is the mean squared distance of the kept correspondences
        assert cost < 1e-10

        # without filtering the outliers should degrade the result
        matrixU, _transformedU, costU = g.trimesh.registration.icp(
            points_a, points_b, scale=False
        )
        error_filtered = g.np.abs(matrixN - matrix).max()
        error_unfiltered = g.np.abs(matrixU - matrix).max()
        assert error_filtered < error_unfiltered
        assert costU > cost

    def test_icp_filter_order(self):
        # `max_correspondence_distance` is applied first and
        # `keep_ratio` then keeps a rounded-up fraction of the
        # closest *remaining* correspondences
        a = g.np.array(
            [[float(i), 0.3 * (i % 2), 0.0] for i in range(10)]
        )
        # offset targets so nearest-neighbor distances are known
        offsets = 0.05 * g.np.arange(1, 11)
        b = a + g.np.column_stack(
            (g.np.zeros(10), offsets, g.np.zeros(10))
        )

        # distances are [0.05 .. 0.50]: the distance limit keeps the
        # first 6 and the ratio then keeps ceil(0.5 * 6) == 3 of them
        matrixN, transformed, cost = g.trimesh.registration.icp(
            a,
            b,
            scale=False,
            max_iterations=1,
            max_correspondence_distance=0.32,
            keep_ratio=0.5,
        )

        # the expected result is a procrustes solve on exactly the
        # three closest in-limit correspondences
        matrixE, transformedE, costE = g.trimesh.registration.procrustes(
            a=a[:3], b=b[:3], scale=False
        )
        assert g.np.allclose(matrixN, matrixE)
        assert g.np.isclose(cost, costE)
        # all points are transformed even though three were used
        assert g.np.allclose(transformed, g.trimesh.transform_points(a, matrixN))
        assert g.np.allclose(transformed[:3], transformedE)

        # keeping everything should match the unfiltered behavior
        m1, t1, c1 = g.trimesh.registration.icp(a, b, scale=False)
        m2, t2, c2 = g.trimesh.registration.icp(a, b, scale=False, keep_ratio=1.0)
        assert g.np.allclose(m1, m2)
        assert g.np.allclose(t1, t2)
        assert g.np.isclose(c1, c2)

    def test_icp_filter_invalid(self):
        # invalid filtering parameters should raise
        rng = g.np.random.default_rng(seed=0)
        a = rng.random((10, 3))
        b = rng.random((10, 3))
        icp = g.trimesh.registration.icp

        with g.pytest.raises(ValueError):
            icp(a, b, max_correspondence_distance=-1.0)
        with g.pytest.raises(ValueError):
            icp(a, b, max_correspondence_distance=0.0)
        with g.pytest.raises(ValueError):
            icp(a, b, max_correspondence_distance=g.np.inf)
        with g.pytest.raises(ValueError):
            icp(a, b, keep_ratio=0.0)
        with g.pytest.raises(ValueError):
            icp(a, b, keep_ratio=1.5)
        with g.pytest.raises(ValueError):
            icp(a, b, keep_ratio=g.np.nan)

        # too few correspondences survive the filtering
        with g.pytest.raises(ValueError):
            icp(a, b, max_correspondence_distance=1e-12)

    def test_mesh(self):
        noise = 0.05
        extents = [6, 12, 3]

        # create the mesh as a simple box
        mesh = g.trimesh.creation.box(extents=extents)

        # subdivide until we have more faces than we want
        for _i in range(3):
            mesh = mesh.subdivide()
        # apply tessellation and random noise
        mesh = mesh.permutate.noise(noise)
        # randomly rotation with translation
        transform = next(g.random_transforms(1, translate=1000))

        mesh.apply_transform(transform)

        scan = mesh
        # create a "true" mesh
        truth = g.trimesh.creation.box(extents=extents)

        for a, b in [[truth, scan], [scan, truth]]:
            a_to_b, _cost = a.register(b)

            a_check = a.copy()
            a_check.apply_transform(a_to_b)

            assert g.np.linalg.norm(a_check.centroid - b.centroid) < (noise * 2)

            # find the distance from the truth mesh to each scan vertex
            distance = a_check.nearest.on_surface(b.vertices)[1]
            assert distance.max() < (noise * 2)

        # try our registration with points
        points = g.trimesh.transform_points(
            scan.sample(100, seed=0),
            matrix=next(g.random_transforms(1, translate=0.0)),
        )
        truth_to_points, _cost = truth.register(points)
        truth.apply_transform(truth_to_points)
        distance = truth.nearest.on_surface(points)[1]

        assert distance.mean() < noise

    def test_nricp(self):
        # Get two meshes that have a comparable shape
        source = g.get_mesh("reference.obj", process=False)
        target = g.get_mesh("target.obj", process=False)

        # Vertex indices of landmarks source / target
        landmarks_vertex_indices = g.np.array(
            [
                [177, 1633],
                [181, 1561],
                [614, 1556],
                [610, 1629],
                [114, 315],
                [398, 413],
                [812, 412],
                [227, 99],
                [241, 87],
                [674, 86],
                [660, 98],
                [362, 574],
                [779, 573],
            ]
        )

        source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
        target_markers_vertices = target.vertices[landmarks_vertex_indices[:, 1]]

        T = g.trimesh.registration.procrustes(
            source_markers_vertices, target_markers_vertices
        )[0]
        source.vertices = g.trimesh.transformations.transform_points(source.vertices, T)

        # Just for the sake of using barycentric coordinates...
        use_barycentric_coordinates = True
        if use_barycentric_coordinates:
            source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
            source_markers_tids = g.trimesh.proximity.closest_point(
                source, source_markers_vertices
            )[2]
            source_markers_barys = g.trimesh.triangles.points_to_barycentric(
                source.triangles[source_markers_tids], source_markers_vertices
            )
            source_landmarks = (source_markers_tids, source_markers_barys)
        else:
            source_landmarks = landmarks_vertex_indices[:, 0]
        # Make copies to check nothing is modified
        source_copy = source.copy()
        target_copy = target.copy()

        # Parameters for nricp_amberg
        wl = 3
        max_iter = 10
        wn = 0.5
        steps_amberg = [
            # ws, wl, wn, max_iter
            [0.02, wl, wn, max_iter],
            [0.007, wl, wn, max_iter],
            [0.002, wl, wn, max_iter],
        ]

        # Parameters for nricp_sumner
        wl = 1000
        wi = 0.0001
        ws = 1
        wn = 1
        steps_sumner = [
            # wc, wi, ws, wl, wn
            [0.1, wi, ws, wl, wn],
            [0.1, wi, ws, wl, wn],
            [1, wi, ws, wl, wn],
            [5, wi, ws, wl, wn],
            [10, wi, ws, wl, wn],
            [50, wi, ws, wl, wn],
            [100, wi, ws, wl, wn],
            [1000, wi, ws, wl, wn],
            [10000, wi, ws, wl, wn],
        ]
        # Amberg et. al 2007
        records_amberg_no_ldm = g.trimesh.registration.nricp_amberg(
            source,
            target,
            distance_threshold=0.05,
            steps=steps_amberg,
            return_records=True,
        )
        records_amberg_ldm = g.trimesh.registration.nricp_amberg(
            source,
            target,
            source_landmarks=source_landmarks,
            target_positions=target_markers_vertices,
            steps=steps_amberg,
            return_records=True,
            distance_threshold=0.05,
        )
        try:
            g.trimesh.registration.nricp_amberg(source, target)
        except KeyError:
            raise AssertionError()  # related to #1724

        # Sumner and Popovic 2004
        records_sumner_no_ldm = g.trimesh.registration.nricp_sumner(
            source,
            target,
            distance_threshold=0.05,
            steps=steps_sumner,
            return_records=True,
        )
        records_sumner_ldm = g.trimesh.registration.nricp_sumner(
            source,
            target,
            source_landmarks=source_landmarks,
            target_positions=target_markers_vertices,
            steps=steps_sumner,
            return_records=True,
            distance_threshold=0.05,
        )
        try:
            g.trimesh.registration.nricp_sumner(source, target)
        except KeyError:
            raise AssertionError()  # related to #1724

        d_amberg_no_ldm = g.trimesh.proximity.closest_point(
            target, records_amberg_no_ldm[-1]
        )[1]
        d_amberg_ldm = g.trimesh.proximity.closest_point(target, records_amberg_ldm[-1])[
            1
        ]
        d_sumner_no_ldm = g.trimesh.proximity.closest_point(
            target, records_sumner_no_ldm[-1]
        )[1]
        d_sumner_ldm = g.trimesh.proximity.closest_point(target, records_sumner_ldm[-1])[
            1
        ]

        # Meshes should remain untouched
        assert g.np.allclose(source.vertices, source_copy.vertices)
        assert g.np.allclose(target.vertices, target_copy.vertices)

        # Basically a lot of check to check if the return values change somehow
        assert d_amberg_no_ldm.min() < 1e-7
        assert d_amberg_no_ldm.max() > 0.05
        assert d_amberg_no_ldm.mean() < 1e-3

        assert d_amberg_ldm.min() > 1e-7
        assert d_amberg_ldm.max() < 0.06
        assert d_amberg_ldm.mean() < 1e-3

        assert d_sumner_no_ldm.min() < 1e-8
        assert d_sumner_no_ldm.max() < 0.03
        assert d_sumner_no_ldm.mean() < 1e-3

        assert d_sumner_ldm.min() < 1e-8
        assert d_sumner_ldm.max() > 0.05
        assert d_sumner_ldm.mean() < 1e-3

        dl_amberg_no_ldm = g.np.linalg.norm(
            records_amberg_no_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_amberg_ldm = g.np.linalg.norm(
            records_amberg_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_sumner_no_ldm = g.np.linalg.norm(
            records_sumner_no_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )
        dl_sumner_ldm = g.np.linalg.norm(
            records_sumner_ldm[-1][landmarks_vertex_indices[:, 0]]
            - target_markers_vertices,
            axis=-1,
        )

        assert dl_amberg_no_ldm.min() > 0.01
        assert dl_amberg_no_ldm.max() > 0.1
        assert dl_amberg_no_ldm.mean() > 0.05

        assert dl_amberg_ldm.min() < 0.0002
        assert dl_amberg_ldm.max() < 0.001
        assert dl_amberg_ldm.mean() < 0.0004

        assert dl_sumner_no_ldm.min() > 0.01
        assert dl_sumner_no_ldm.max() > 0.1
        assert dl_sumner_no_ldm.mean() > 0.05

        assert dl_sumner_ldm.min() < 0.0007
        assert dl_sumner_ldm.max() < 0.006
        assert dl_sumner_ldm.mean() < 0.004

    def test_query_from_points(self):
        points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        query_point = [[0, 0.5, 0]]
        qres = g.trimesh.registration._from_points(
            points, query_point, return_normals=False
        )
        assert qres["vertex_indices"][0] == 1
        assert g.np.all(qres["nearest"][0] == [0, 1, 0])
        assert "normals" not in qres
        qres = g.trimesh.registration._from_points(
            points, query_point, return_normals=True
        )
        normal = g.np.ones(3)
        normal = normal / g.np.linalg.norm(normal)
        assert g.np.allclose(qres["normals"][0], normal) or g.np.allclose(
            qres["normals"][0], -normal
        )

    def test_query_from_mesh(self):
        points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 1, 2]]
        mesh = g.trimesh.Trimesh(vertices=points, faces=faces)
        query_point = [[0, 0.5, 0]]
        qres = g.trimesh.registration._from_mesh(
            mesh,
            query_point,
            return_barycentric_coordinates=False,
            return_normals=False,
            return_interpolated_normals=False,
        )

        assert "normals" not in qres
        assert "barycentric_coordinates" not in qres
        assert "interpolated_normals" not in qres

        qres = g.trimesh.registration._from_mesh(
            mesh,
            query_point,
            return_barycentric_coordinates=False,
            return_normals=False,
            return_interpolated_normals=True,
        )

        assert "normals" not in qres
        assert "barycentric_coordinates" in qres
        assert "interpolated_normals" in qres


@g.pytest.mark.parametrize("reflection", [False, True])
@g.pytest.mark.parametrize("translation", [False, True])
@g.pytest.mark.parametrize("scale", [False, True])
@g.pytest.mark.parametrize("mirrored", [False, True])
def test_procrustes_weighted_correspondence_repetition(
    reflection, translation, scale, mirrored
):
    # Integer weights must give the same fit as repeating each correspondence.
    # Noise is needed: exact rigid pairs fit even with an unweighted covariance.
    a = g.np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3], [2, 1, 1], [7, 5, 2]],
        dtype=g.np.float64,
    )
    b = a[:, [1, 2, 0]] + [0.5, -0.2, 0.1]
    if mirrored:
        b[:, 0] *= -1.0
    b[4] += [1.0, -2.0, 0.5]
    weights = g.np.array([5, 4, 3, 2, 1, 0])
    options = {"reflection": reflection, "translation": translation, "scale": scale}
    expected, _, expected_cost = g.trimesh.registration.procrustes(
        g.np.repeat(a, weights, axis=0), g.np.repeat(b, weights, axis=0), **options
    )
    matrix, transformed, cost = g.trimesh.registration.procrustes(
        a, b, weights=weights, **options
    )
    g.np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
    g.np.testing.assert_allclose(
        transformed, g.trimesh.transform_points(a, expected), rtol=1e-12, atol=1e-12
    )
    g.np.testing.assert_allclose(cost, expected_cost, rtol=1e-12, atol=1e-12)
    assert g.np.isclose(
        cost, (((b - transformed) ** 2).sum(axis=1) * weights).sum() / weights.sum()
    )
    # Weights need not be integers or normalized; the matrix-only path agrees.
    matrix_only = g.trimesh.registration.procrustes(
        a, b, weights=weights * 0.37, return_cost=False, **options
    )
    g.np.testing.assert_allclose(matrix_only, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
