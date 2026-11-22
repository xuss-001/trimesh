import copy
import itertools

import numpy as np

from .. import constants, grouping, util
from ..typed import ArrayLike, Integer, NDArray, Number
from .entities import Arc, Line
from .util import is_ccw

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from ..exceptions import ExceptionWrapper

    nx = ExceptionWrapper(E)


def vertex_graph(entities):
    """
    Given a set of entity objects generate a networkx.Graph
    that represents their vertex nodes.

    Parameters
    --------------
    entities : list
       Objects with 'closed' and 'nodes' attributes

    Returns
    -------------
    graph : networkx.Graph
        Graph where node indexes represent vertices
    closed : (n,) int
        Indexes of entities which are 'closed'
    """
    graph = nx.Graph()
    closed = []
    for index, entity in enumerate(entities):
        if entity.closed:
            closed.append(index)
        else:
            # or `entity.end_points`
            graph.add_edges_from(entity.nodes, entity_index=index)
    return graph, np.array(closed)


def vertex_to_entity_path(vertex_path, graph, entities, vertices=None):
    """
    Convert a path of vertex indices to a path of entity indices.

    Parameters
    ----------
    vertex_path : (n,) int
        Ordered list of vertex indices representing a path
    graph : nx.Graph
        Vertex connectivity
    entities : (m,) list
        Entity objects
    vertices :  (p, dimension) float
        Vertex points in space

    Returns
    ----------
    entity_path : (q,) int
        Entity indices which make up vertex_path
    """

    def edge_direction(a, b):
        """
        Given two edges, figure out if the first needs to be
         reversed to keep the progression forward.

         [1,0] [1,2] -1  1
         [1,0] [2,1] -1 -1
         [0,1] [1,2]  1  1
         [0,1] [2,1]  1 -1

        Parameters
        ------------
        a : (2,) int
        b : (2,) int

        Returns
        ------------
        a_direction : int
        b_direction : int
        """
        if a[0] == b[0]:
            return -1, 1
        elif a[0] == b[1]:
            return -1, -1
        elif a[1] == b[0]:
            return 1, 1
        elif a[1] == b[1]:
            return 1, -1
        else:
            constants.log.debug(
                "\n".join(
                    [
                        "edges not connected!",
                        "vertex path %s",
                        "entity path: %s",
                        "entity[a]: %s,",
                        "entity[b]: %s",
                    ]
                ),
                vertex_path,
                entity_path,
                entities[ea].points,
                entities[eb].points,
            )

            return None, None

    if vertices is None or vertices.shape[1] != 2:
        ccw_direction = 1
    else:
        ccw_check = is_ccw(vertices[np.append(vertex_path, vertex_path[0])])
        ccw_direction = (ccw_check * 2) - 1

    # make sure vertex path is correct type
    vertex_path = np.asanyarray(vertex_path, dtype=np.int64)
    # we will be saving entity indexes
    entity_path = []
    # loop through pairs of vertices
    for i in np.arange(len(vertex_path) + 1):
        # get two wrapped vertex positions
        vertex_path_pos = np.mod(np.arange(2) + i, len(vertex_path))
        vertex_index = vertex_path[vertex_path_pos]
        entity_index = graph.get_edge_data(*vertex_index)["entity_index"]
        entity_path.append(entity_index)
    # remove duplicate entities and order CCW
    entity_path = grouping.unique_ordered(entity_path)[::ccw_direction]
    # check to make sure there is more than one entity
    if len(entity_path) == 1:
        # apply CCW reverse in place if necessary
        if ccw_direction < 0:
            index = entity_path[0]
            entities[index].reverse()

        return entity_path
    # traverse the entity path and reverse entities in place to
    # align with this path ordering
    round_trip = np.append(entity_path, entity_path[0])
    round_trip = itertools.pairwise(round_trip)
    for ea, eb in round_trip:
        da, db = edge_direction(entities[ea].end_points, entities[eb].end_points)
        if da is not None:
            entities[ea].reverse(direction=da)
            entities[eb].reverse(direction=db)

    entity_path = np.array(entity_path)

    return entity_path


def closed_paths(entities, vertices):
    """
    Paths are lists of entity indices.
    We first generate vertex paths using graph cycle algorithms,
    and then convert them to entity paths.

    This will also change the ordering of entity.points in place
    so a path may be traversed without having to reverse the entity.

    Parameters
    -------------
    entities : (n,) entity objects
        Entity objects
    vertices : (m, dimension) float
        Vertex points in space

    Returns
    -------------
    entity_paths : sequence of (n,) int
        Ordered traversals of entities
    """
    # get a networkx graph of entities
    graph, closed = vertex_graph(entities)
    # add entities that are closed as single- entity paths
    entity_paths = np.reshape(closed, (-1, 1)).tolist()
    # look for cycles in the graph, or closed loops
    vertex_paths = nx.cycles.cycle_basis(graph)

    # loop through every vertex cycle
    for vertex_path in vertex_paths:
        # a path has no length if it has fewer than 2 vertices
        if len(vertex_path) < 2:
            continue
        # convert vertex indices to entity indices
        entity_paths.append(vertex_to_entity_path(vertex_path, graph, entities, vertices))

    return entity_paths


def discretize_path(entities, vertices, path, scale=1.0):
    """
    Turn a list of entity indices into a path of connected points.

    Parameters
    -----------
    entities : (j,) entity objects
       Objects like 'Line', 'Arc', etc.
    vertices: (n, dimension) float
        Vertex points in space.
    path : (m,) int
        Indexes of entities
    scale : float
        Overall scale of drawing used for
        Number tolerances in certain cases

    Returns
    -----------
    discrete : (p, dimension) float
       Connected points in space that lie on the
       path and can be connected with line segments.
    """
    # make sure vertices are numpy array
    vertices = np.asanyarray(vertices)
    path_len = len(path)
    if path_len == 0:
        raise ValueError("Cannot discretize empty path!")
    if path_len == 1:
        # case where we only have one entity
        discrete = np.asanyarray(entities[path[0]].discrete(vertices, scale=scale))
    else:
        # run through path appending each entity
        discrete = []
        for i, entity_id in enumerate(path):
            # the current (n, dimension) discrete curve of an entity
            current = entities[entity_id].discrete(vertices, scale=scale)
            # check if we are on the final entity
            if i >= (path_len - 1):
                # if we are on the last entity include the last point
                discrete.append(current)
            else:
                # slice off the last point so we don't get duplicate
                # points from the end of one entity and the start of another
                discrete.append(current[:-1])
        # stack all curves to one nice (n, dimension) curve
        discrete = np.vstack(discrete)
    # make sure 2D curves are are counterclockwise
    if vertices.shape[1] == 2 and not is_ccw(discrete):
        # reversing will make array non c- contiguous
        discrete = np.ascontiguousarray(discrete[::-1])

    return discrete


class PathSample:
    def __init__(self, points: ArrayLike):
        # make sure input array is numpy
        self._points = np.array(points)
        # find the direction of each segment
        self._vectors = np.diff(self._points, axis=0)
        # find the length of each segment
        self._norms = util.row_norm(self._vectors)
        # unit vectors for each segment
        nonzero = self._norms > constants.tol_path.zero
        self._unit_vec = self._vectors.copy()
        self._unit_vec[nonzero] /= self._norms[nonzero].reshape((-1, 1))
        # total distance in the path
        self.length = self._norms.sum()
        # cumulative sum of section length
        # note that this is sorted
        self._cum_norm = np.cumsum(self._norms)

    def sample(
        self, distances: ArrayLike, include_original: bool = False
    ) -> NDArray[np.float64]:
        """
        Return points at the distances along the path requested.

        Parameters
        ----------
        distances
          Distances along the path to sample at.
        include_original
          Include the original vertices even if they are not
          specified in `distance`. Useful as this will return
          a result with identical area and length, however
          indexes of `distance` will not correspond with result.

        Returns
        --------
        samples : (n, dimension)
          Samples requested.
          `n==len(distances)` if not `include_original`
        """
        # return the indices in cum_norm that each sample would
        # need to be inserted at to maintain the sorted property
        positions = np.searchsorted(self._cum_norm, distances)
        positions = np.clip(positions, 0, len(self._unit_vec) - 1)
        offsets = np.append(0, self._cum_norm)[positions]
        # the distance past the reference vertex we need to travel
        projection = distances - offsets
        # find out which direction we need to project
        direction = self._unit_vec[positions]
        # find out which vertex we're offset from
        origin = self._points[positions]

        # just the parametric equation for a line
        resampled = origin + (direction * projection.reshape((-1, 1)))

        if include_original:
            # find the original positions that were not inserted
            # note that this checks *exact float equal*
            uninserted = ~np.isin(np.append(0.0, self._cum_norm), distances)

            if uninserted.any():
                # find the index of the uninserted original points in the new sampling
                index = np.searchsorted(positions, np.nonzero(uninserted)[0])
                # insert the original points at the index
                resampled = np.insert(resampled, index, self._points[uninserted], axis=0)

        return resampled

    def truncate(self, distance: Number) -> NDArray[np.float64]:
        """
        Return a truncated version of the path.
        Only one vertex (at the endpoint) will be added.

        Parameters
        ----------
        distance
          Distance along the path to truncate at.

        Returns
        ----------
        path
          Path clipped to `distance` requested.
        """
        position = np.searchsorted(self._cum_norm, distance)
        offset = distance - self._cum_norm[position - 1]

        if offset < constants.tol_path.merge:
            truncated = self._points[: position + 1]
        else:
            vector = util.unitize(
                np.diff(self._points[np.arange(2) + position], axis=0).reshape(-1)
            )
            vector *= offset
            endpoint = self._points[position] + vector
            truncated = np.vstack((self._points[: position + 1], endpoint))
        assert (
            util.row_norm(np.diff(truncated, axis=0)).sum() - distance
        ) < constants.tol_path.merge

        return truncated


class CurveSample:
    """
    Sample a connected path of analytic line and arc entities at
    exact arc-length intervals.

    Distances are accumulated continuously along the traversal so
    sampling never restarts at an entity junction and never relies
    on a coarse polyline approximation: lines are evaluated exactly
    and arcs are evaluated on their circle using ``s = R * theta``.

    Input entities and vertices are never modified; any entity whose
    control points run against the traversal direction is handled
    internally without calling :meth:`Entity.reverse`.
    """

    def __init__(self, entities, vertices, path, closed=None):
        """
        Parameters
        ------------
        entities : (m,) trimesh.path.entities.Entity
          Line and Arc objects referenced by `path`.
        vertices : (n, 2) or (n, 3) float
          Vertex points referenced by the entities.
        path : (k,) int
          Entity indices giving the traversal order.
        closed : bool or None
          If ``None`` closure is detected from endpoint connectivity:
          every endpoint shared by two entities means closed, while
          dangling endpoints mean open.
        """
        vertices = np.asanyarray(vertices, dtype=np.float64)
        path = np.asanyarray(path, dtype=np.int64).reshape(-1)
        if len(path) == 0:
            raise ValueError("cannot sample an empty path!")

        # the ordered entities of the traversal
        ordered = [entities[index] for index in path]
        for entity in ordered:
            if not isinstance(entity, (Line, Arc)):
                raise TypeError(
                    "arc-length sampling only supports Line and Arc entities"
                )

        # determine the traversal direction of each entity without
        # mutating the entities themselves
        if closed is None:
            # a closed contour has every endpoint shared by exactly
            # two entities, while an open contour has two dangling ends
            degree = {}
            for entity in ordered:
                for node in entity.end_points:
                    degree[int(node)] = degree.get(int(node), 0) + 1
            closed = len(ordered) > 1 and all(
                count == 2 for count in degree.values()
            )
        directions = self._orient(ordered, closed=closed)

        # per-entity exact length and point-at-arc-length data
        pieces = []
        lengths = np.zeros(len(ordered), dtype=np.float64)
        for index, (entity, direction) in enumerate(zip(ordered, directions)):
            piece, lengths[index] = self._piece(
                entity=entity,
                vertices=vertices,
                reverse=direction < 0,
            )
            pieces.append(piece)

        # cumulative length, i.e. the global distance of each junction
        self._bounds = np.append(0.0, np.cumsum(lengths))
        self._total = float(self._bounds[-1])
        self._pieces = pieces
        self._dimension = vertices.shape[1]
        # original entity index in `entities` for every ordered piece
        self._entity_ids = np.asanyarray(path, dtype=np.int64)
        self._closed = bool(closed)
        self.length = self._total

    @staticmethod
    def _orient(ordered, closed):
        """
        Find a traversal direction (+1/-1) for every ordered entity
        so that the end of one entity matches the start of the next.
        """
        directions = np.ones(len(ordered), dtype=np.int64)

        if len(ordered) == 1:
            # a single open entity is traversed as given; a closed
            # circle is identical in either direction
            return directions

        for index in range(len(ordered) - 1):
            current = ordered[index]
            following = ordered[index + 1]
            start, end = current.end_points[:: directions[index]]
            other = following.end_points
            shared = set(other) & {start, end}
            if not shared:
                raise ValueError("entities in path are not connected!")
            shared = next(iter(shared))
            if shared != end:
                directions[index] *= -1
            # the junction is the start of the following entity
            directions[index + 1] = -1 if other[0] != shared else 1

        if closed:
            # verify the loop closes back on the first start point
            start = ordered[0].end_points[:: directions[0]][0]
            end = ordered[-1].end_points[:: directions[-1]][-1]
            if start != end:
                raise ValueError("closed path does not return to its start!")

        return directions

    @staticmethod
    def _piece(entity, vertices, reverse):
        """
        Build exact point-at-arc-length data for one oriented entity.

        Returns
        ---------
        piece : dict
          Geometry evaluated along the traversal direction.
        length : float
          Exact length of the entity.
        """
        if not isinstance(entity, Arc):
            points = vertices[entity.points]
            if reverse:
                points = points[::-1]
            vectors = np.diff(points, axis=0)
            segment = util.row_norm(vectors)
            if np.any(segment <= constants.tol_path.zero):
                raise ValueError("line entity contains zero-length segments!")
            piece = {
                "kind": "line",
                "points": points,
                "unit": vectors / segment.reshape((-1, 1)),
                "cum": np.append(0.0, np.cumsum(segment)),
            }
            return piece, float(segment.sum())

        fit = entity.center(vertices, return_normal=True, return_angle=True)
        radius = fit.radius
        points = vertices[entity.points]
        if reverse:
            # traverse the arc from its last control point backwards
            points = points[::-1]
            normal = -fit.normal
        else:
            normal = fit.normal
        length = (
            np.pi * 2.0 * radius if entity.closed else fit.span * radius
        )

        # v1 points from the center to the oriented start point; v2 is
        # the unit tangent 90 degrees along the oriented arc direction.
        # Its sign is taken from the actual control-point sweep rather
        # than the fitted normal, which flips with arc handedness
        v1 = util.unitize(points[0] - fit.center)
        if points.shape[1] == 2:
            radial = util.unitize(points[1] - fit.center)
            side = v1[0] * radial[1] - v1[1] * radial[0]
            v2 = np.array([-v1[1], v1[0]]) * np.sign(side)
        else:
            radial = util.unitize(points[1] - fit.center)
            side = np.dot(np.cross(v1, radial), normal)
            v2 = util.unitize(np.cross(normal, v1)) * np.sign(side)

        piece = {
            "kind": "arc",
            "center": fit.center,
            "radius": radius,
            "v1": v1,
            "v2": v2,
        }
        return piece, float(length)

    def sample(self, step):
        """
        Return points sampled at constant arc-length spacing.

        Parameters
        ------------
        step : float
          Arc-length distance between consecutive samples.

        Returns
        ---------
        points : (n, dimension) float
          Sample coordinates in traversal order.
        entity_ids : (n,) int
          Original entity index each sample belongs to. A sample on
          a junction is assigned to the later entity in the traversal;
          the open-contour endpoint belongs to the last entity.
        local : (n,) float
          Arc length from the entity's traversal start to the sample.
        """
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("`step` must be a positive finite number!")

        # floor-division based indices avoid np.arange float overshoot
        if self._closed:
            count = self._total / step
            # snap near-integer divisions upward against float error
            if np.isclose(count, round(count), rtol=1e-12):
                count = int(round(count))
            else:
                count = int(np.floor(count))
            if count < 1:
                count = 1
            distances = np.arange(count, dtype=np.float64) * step
        else:
            count = int(np.floor(self._total / step)) + 1
            distances = np.arange(count, dtype=np.float64) * step
            # the open contour always keeps its final endpoint, even
            # when the remaining segment is shorter than `step`
            if not np.isclose(distances[-1], self._total):
                distances = np.append(distances, self._total)

        # assign each distance to an entity; junctions go to the
        # later entity, so use the right insertion side of the
        # exclusive per-entity upper bounds
        indices = np.searchsorted(self._bounds[1:], distances, side="right")
        indices = np.clip(indices, 0, len(self._pieces) - 1)
        local = distances - self._bounds[indices]

        # evaluate every sample while preserving traversal order
        points = self._evaluate(indices=indices, local=local)

        return (
            points,
            self._entity_ids[indices],
            local,
        )

    def _evaluate(self, indices, local):
        """
        Evaluate piece geometry at local arc-length values,
        keeping the result in the input order.
        """
        points = np.zeros((len(indices), self._dimension))
        for index in np.unique(indices):
            chosen = indices == index
            piece = self._pieces[index]
            local_chosen = local[chosen]
            if piece["kind"] == "line":
                segment = np.searchsorted(
                    piece["cum"][1:], local_chosen, side="right"
                )
                segment = np.clip(segment, 0, len(piece["unit"]) - 1)
                offset = local_chosen - piece["cum"][segment]
                points[chosen] = (
                    piece["points"][segment]
                    + piece["unit"][segment] * offset.reshape((-1, 1))
                )
            else:
                angle = local_chosen / piece["radius"]
                points[chosen] = (
                    piece["center"]
                    + piece["radius"]
                    * (
                        np.cos(angle).reshape((-1, 1)) * piece["v1"]
                        + np.sin(angle).reshape((-1, 1)) * piece["v2"]
                    )
                )
        return points


def sample_arc_length(entities, vertices, path, step, closed=None):
    """
    Sample a connected line/arc contour at exact arc-length intervals.

    Parameters
    ------------
    entities : (m,) trimesh.path.entities.Entity
      Line and Arc objects referenced by `path`.
    vertices : (n, 2) or (n, 3) float
      Vertex points referenced by the entities.
    path : (k,) int
      Entity indices giving the traversal order.
    step : float
      Arc-length distance between consecutive samples.
    closed : bool or None
      If ``None`` closure is detected from endpoint connectivity.

    Returns
    ---------
    points : (n, dimension) float
      Sample coordinates in traversal order.
    entity_ids : (n,) int
      Original entity index each sample belongs to.
    local : (n,) float
      Arc length from the entity's traversal start to each sample.
    """
    return CurveSample(entities, vertices, path, closed=closed).sample(step)


def resample_path(
    points: ArrayLike,
    count: Integer | None = None,
    step: Number | None = None,
    step_round: bool = True,
    include_original: bool = False,
) -> NDArray[np.float64]:
    """
    Given a path along (n,d) points, resample them such that the
    distance traversed along the path is constant in between each
    of the resampled points. Note that this can produce clipping at
    corners, as the original vertices are NOT guaranteed to be in the
    new, resampled path.

    ONLY ONE of count or step can be specified
    Result can be uniformly distributed (np.linspace) by specifying count
    Result can have a specific distance (np.arange) by specifying step


    Parameters
    ----------
    points:   (n, d) float
      Points in space
    count : int,
      Number of points to sample evenly (aka np.linspace)
    step : float
      Distance each step should take along the path (aka np.arange)
    step_round
      Alter `step` to the nearest integer division of overall length.
    include_original
      Include the exact original points in the output.

    Returns
    ----------
    resampled : (j,d) float
        Points on the path
    """
    points = np.array(points, dtype=np.float64)
    # generate samples along the perimeter from kwarg count or step
    if (count is not None) and (step is not None):
        raise ValueError("Only step OR count can be specified")
    if (count is None) and (step is None):
        raise ValueError("Either step or count must be specified")

    sampler = PathSample(points)
    if step is not None and step_round:
        if step >= sampler.length:
            return points[[0, -1]]

        count = int(np.ceil(sampler.length / step))

    if count is not None:
        samples = np.linspace(0, sampler.length, count)
    elif step is not None:
        samples = np.arange(0, sampler.length, step)

    resampled = sampler.sample(samples, include_original=include_original)

    if constants.tol.strict:
        check = util.row_norm(points[[0, -1]] - resampled[[0, -1]])
        assert check[0] < constants.tol_path.merge
        if count is not None:
            assert check[1] < constants.tol_path.merge

    return resampled


def split(path):
    """
    Split a Path2D into multiple Path2D objects where each
    one has exactly one root curve.

    Parameters
    --------------
    path : trimesh.path.Path2D
      Input geometry

    Returns
    -------------
    split : list of trimesh.path.Path2D
      Original geometry as separate paths
    """
    # avoid a circular import by referencing class of path
    Path2D = type(path)

    # save the results of the split to an array
    split = []

    # get objects from cache to avoid a bajillion
    # cache checks inside the tight loop
    paths = path.paths
    discrete = path.discrete
    polygons_closed = path.polygons_closed
    enclosure_directed = path.enclosure_directed

    for root_index, root in enumerate(path.root):
        # get a list of the root curve's children
        connected = list(enclosure_directed[root].keys())
        # add the root node to the list
        connected.append(root)

        # store new paths and entities
        new_paths = []
        new_entities = []

        for index in connected:
            nodes = paths[index]
            # add a path which is just sequential indexes
            new_paths.append(np.arange(len(nodes)) + len(new_entities))
            # save the entity indexes
            new_entities.extend(nodes)

        # store the root index from the original drawing
        metadata = copy.deepcopy(path.metadata)
        metadata["split_2D"] = root_index
        # we made the root path the last index of connected
        new_root = np.array([len(new_paths) - 1])

        # prevents the copying from nuking our cache
        with path._cache:
            # create the Path2D
            split.append(
                Path2D(
                    entities=copy.deepcopy(path.entities[new_entities]),
                    vertices=copy.deepcopy(path.vertices),
                    metadata=metadata,
                )
            )

            # add back expensive things to the cache
            split[-1]._cache.update(
                {
                    "paths": new_paths,
                    "polygons_closed": polygons_closed[connected],
                    "discrete": [discrete[c] for c in connected],
                    "root": new_root,
                }
            )
            # set the cache ID
            split[-1]._cache.id_set()

    return np.array(split)
