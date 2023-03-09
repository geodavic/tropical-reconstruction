import numpy as np
import math
from scipy.spatial import ConvexHull
from utils import all_subsets, binary_to_subset

TOLERANCE = 1e-5
TOLERANCE_DIGITS = -int(math.log10(TOLERANCE))


def random_zonotope(n, d, scale=None):
    """Create a random Zonotope of rank n in R^d"""
    if scale is None:
        scale = np.sqrt(1 / n)

    Az = scale * np.random.rand(n, d)
    Z = Zonotope(generators=Az)
    return Z


def random_polytope(k, d, scale=1):
    """Create a random polytope as a convex hull of k
    random points.
    """
    pts = scale * np.random.rand(k, d)
    P = Polytope(pts=pts)
    return P


def zonotope_generators_from_vertices(vert):
    """Compute the Zonotope generators of a convex hull vertices.
    This is somewhat nontrivial for higher dimensions.
    """
    if not is_centrally_symmetric(vert):
        print("Polytope not centrally symmetric, can't compute zonotope generators")
        return None

    if len(vert[0]) == 2:
        # For 2d, vertices of a ConvexHull object are in counterclockwise
        # order, so just take the successive differences to get generators.
        generators = []
        for i in range(len(vert) - 1):
            g = vert[i + 1] - vert[i]
            g *= np.sign(g[0])
            generators.append(g)
        generators = np.array(generators)
        return np.unique(generators.round(decimals=TOLERANCE_DIGITS), axis=0)

    else:
        raise NotImplementedError


def is_centrally_symmetric(vert, tolerance=TOLERANCE):
    """Check if a hull is centrally symmetric."""
    barycenter = np.sum(vert, axis=0) / len(vert)
    for pt1 in vert:
        ref = 2 * barycenter - pt1
        found_reflection = False
        for pt2 in vert:
            err = np.abs(pt2 - ref)
            if max(err) < tolerance:
                found_reflection = True
        if not found_reflection:
            return False
    return True


class Halfspace:
    """A halfspace of the form a \dot x + c <= 0"""

    def __init__(self, a, c):
        self.tolerance = TOLERANCE
        self.a = a
        self.c = c

    def contains(self, x):
        if self.a @ x + self.c <= 0:
            return True
        return False

    def boundary_contains(self, x):
        if abs(self.a @ x + self.c) < self.tolerance:
            return True
        return False

    @property
    def is_vertical(self):
        """Check if halfspace is vertically oriented (w.r.t. last coordinate)

        Should this depend on TOLERANCE?
        """
        return np.round(self.a[-1], decimals=8) > 0


class Polytope:
    """A polytope in V representation.

    TODO: add sample() method.
    """

    def __init__(self, hull: ConvexHull = None, pts: np.array = None):
        self.tolerance = TOLERANCE
        assert (
            hull is not None or pts is not None
        ), "Must pass either a ConvexHull object or a list of points"
        if hull is None:
            self.hull = ConvexHull(pts)
        else:
            self.hull = hull

    def __add__(self, Q):
        """Minkowski sum"""
        new_pts = []
        assert isinstance(Q, Polytope), "Can only Minkowski add Polytope objects"
        for p1 in self.vertices:
            for p2 in Q.vertices:
                new_pts += [p1 + p2]
        return self.__class__(pts=new_pts)

    def __mul__(self, a):
        """Multiply by a scalar"""
        new_pts = []
        for pt in self.pts:
            new_pts.append(a * pt)
        return self.__class__(pts=new_pts)

    def __rmul__(self, a):
        """Same as __mul__"""
        return self.__mul__(a)

    def update(self, new_pts):
        """Add `new_pts` to self.pts"""
        new_pts = np.append(self.pts, new_pts, axis=0)
        return self.Polytope(pts=new_pts)

    @property
    def radius(self):
        return max([np.linalg.norm(self.barycenter - v) for v in self.vertices])

    @property
    def vertices(self):
        return self.pts[self.hull.vertices]

    @property
    def hyperplanes(self):
        inequalities = [e for e in np.unique(self.hull.equations, axis=0)]
        return [Halfspace(eq[:-1], eq[-1]) for eq in inequalities]

    @property
    def upper_hull(self):
        """Get the upper hull hyperplanes of P"""
        return [h for h in self.hyperplanes if h.is_vertical]

    @property
    def dim(self):
        return len(self.pts[0])

    @property
    def pts(self):
        return self.hull.points

    @property
    def barycenter(self):
        """Compute the barycenter of the zonotope."""
        return np.sum(self.vertices, axis=0) / len(self.vertices)

    @property
    def regular_subdivision(self):
        """Return the simplices of the regular subdivision of the projection of P
        away from its last coordinate.
        """
        simplices = []
        for h in self.upper_hull:
            simplex = []
            for v in self.vertices:
                if h.boundary_contains(v):
                    simplex.append(v[:-1])
            simplices.append(simplex)
        return np.array(simplices)

    @property
    def upper_hull_vertices(self):
        """Return the vertices of the upper hull of P."""
        vert = []
        for v in self.vertices:
            for h in self.upper_hull:
                if h.boundary_contains(v):
                    vert.append(v)
                    break
        return np.array(vert)

    @property
    def is_centrally_symmetric(self):
        return is_centrally_symmetric(self.vertices, self.tolerance)

    def incident_hyperplanes(self, x):
        """Return the supporting hyperplanes to x. There might be more efficient
        ways to compute this (e.g. Simplex method).

        Hyperplanes will be in the form (a,c) where a is the normal and c is the offset.
        The associated halfspace is defined by a \dot x + c <= 0.
        """
        hyperplanes = []
        for eq in self.hull.equations:
            plane = Halfspace(eq[:-1], eq[-1])
            if plane.boundary_contains(x):
                hyperplanes.append(plane)

        return hyperplanes

    def has_vertex(self, x):
        """Check if x is a vetex of P"""
        # exactly a vertex
        if x in self.vertices:
            return True

        # approximately a vertex:
        # if len(self.incident_hyperplanes(x)) >= self.dim:
        #    return True
        for v in self.vertices:
            if np.linalg.norm(v - x) <= self.tolerance:
                return True

        return False

    def get_pt_idx(self, x, force=False):
        """Get index of x in self.pts"""
        dists = [np.linalg.norm(p - x) for p in self.pts]
        if force:
            return np.argmin(dists)

        if np.min(dists) < self.tolerance:
            return np.argmin(dists)
        else:
            raise ValueError("Passed point is not a point of polytope")

    def get_facet_generators(self, facet_eq):
        gens = []
        for idx, g in enumerate(self.generators):
            if abs(facet_eq.a @ g) <= facet_eq.tolerance:
                gens += [idx]
        return gens

    def bounds(self):
        """Return the max and min values of all coordinates in P."""
        maxs = [-np.infty for _ in range(self.dim)]
        mins = [np.infty for _ in range(self.dim)]

        for pt in self.vertices:
            for i in range(self.dim):
                if pt[i] > maxs[i]:
                    maxs[i] = pt[i]
                if pt[i] < mins[i]:
                    mins[i] = pt[i]
        return maxs, mins

    def translate(self, v):
        """Translate whole polytope by v"""
        for p in self.pts:
            p += v
        self.hull = ConvexHull(self.pts)

    def sym(self, O=None):
        """
        Compute smallest zonotope containing P about the point O.
        """
        if self.dim > 2:
            raise NotImplementedError

        if O is None:
            O = self.barycenter

        reflect = lambda p: 2 * O - p
        new_pts = []
        for pt in self.pts:
            new_pts += [pt, reflect(pt)]

        return Zonotope(pts=new_pts)

    def project(self):
        """Project P away from its last component."""
        new_pts = self.pts[:, :-1]
        return self.__class__(pts=new_pts)


class Zonotope(Polytope):
    """A zonotope in V representation or affine representation.

    Either pass (generators,mu), which is the affine representation
    of Z (i.e. the affine map for which the image is Z) or pts,
    a collection of points whose convex hull is a zonotope.
    """

    def __init__(self, generators=None, mu=None, pts=None):
        self._generators = None

        if generators is not None:
            if mu is None:
                mu = np.zeros(len(generators[0]))
            self._generators = generators
        elif pts is not None:
            mu = None
            vert = Polytope(pts=pts).vertices
            self._generators = zonotope_generators_from_vertices(vert)
        else:
            raise Exception("Must pass either `generators` or `pts`")

        self._pts_subsets = all_subsets(self.rank)
        if mu is None:
            # Find offset mu (this is a bit of a hack?)
            _cubical_vert = np.array(
                [(self.generators).T @ e for e in self._pts_subsets]
            )
            mu = -sum(_cubical_vert) / len(_cubical_vert) + sum(vert) / len(vert)

        self.mu = mu
        cubical_vert = np.array(
            [(self.generators).T @ e + mu for e in self._pts_subsets]
        )
        hull = ConvexHull(cubical_vert)

        super().__init__(hull=hull)

    def update(self, new_pts):
        """Must override parent class, since instances of this class should be immutable."""
        raise NotImplementedError

    def translate(self, v):
        """Must override parent class, since instances of this class should be immutable."""
        generators = np.copy(self.generators)
        mu = np.copy(self.mu)
        return Zonotope(generators=generators, mu=mu + v)

    @property
    def generators(self):
        if self._generators is None:
            self._generators = self._get_generators()
        return self._generators

    def pts_subsets(self, i, binary=True):
        if self._pts_subsets is None:
            self._pts_subsets = self._get_pts_subsets()
        rval = self._pts_subsets[i]
        if not binary:
            rval = binary_to_subset(rval)
        return rval

    @property
    def rank(self):
        return len(self.generators)

    def _get_pts_subsets(self):
        """Take cubical vertices and return the vertices
        of the n-cube they correspond to. Can be optimized probably.
        """
        error = Exception(
            f"Error: passed cubical vertices not in bijection with vertices of a {self.rank}-cube."
        )
        pts_subsets = []
        subsets = all_subsets(self.rank)

        for pt in self.pts:
            found_subset = False
            for i, s in enumerate(subsets):
                cand = sum([self.generators[j] for j in range(len(s)) if s[j]])
                if np.linalg.norm(cand - pt) < TOLERANCE:
                    found_subset = True
                    pts_subsets += [s]
                    break

            if not found_subset:
                raise error

        return pts_subsets

    def _get_generators(self):
        return zonotope_generators_from_vertices(self.vertices)

    def reflected_pt(self, idx):
        """Return the index of the vertex across from the barycenter"""
        ref = 2 * self.barycenter - self.pts[idx]
        for i, pt in enumerate(self.pts):
            err = np.abs(pt - ref)
            if max(err) < self.tolerance:
                return i
        raise Exception(
            "Could not compute reflected point, polytope is not centrally symmetric."
        )

    def move_pt(self, idx, v, normalize=False):
        """Send self.pts[idx] to self.pts[idx] + v and symmetrize

        WARNING: the resulting object is not guaranteed to be a zonotope
        when the dimension exceeds 2. This should only be used when the
        dimension is 2.
        """

        if self.dim != 2:
            Warning(
                "Calling method `move_pt()` in dimensions greater than 2 will likely cause the Zonotope not to be a zonotope anymore"
            )

        if np.linalg.norm(v) == 0:
            return self

        self._generators = None  # generators have to be recomputed
        reflected_idx = self.reflected_pt(idx)

        # move point
        barycenter = self.barycenter
        self.pts[idx] += v
        # symmetrize
        self.pts[reflected_idx] = 2 * barycenter - self.pts[idx]

        # return Zonotope(pts=self.vertices)
        return Zonotope(pts=self.pts)


class BalancedZonotope(Zonotope):
    """A balanced zonotope (image of cube in R^n whose vertices are
    \pm 1 valued not (0,1) valued.)"""

    def __init__(self, generators, mu=None):
        if mu is None:
            mu = np.zeros(len(generators[0]))
        self.mu = mu
        self._generators = generators

        self._pts_subsets = all_subsets(self.rank)

        cubical_vert = np.array(
            [(self.generators).T @ e + mu for e in self._pts_subsets]
        )
        hull = ConvexHull(cubical_vert)

        super().__init__(hull=hull)
