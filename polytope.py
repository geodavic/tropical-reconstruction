import numpy as np
from scipy.spatial import ConvexHull

TOLERANCE = 1e-12

def cube(n):
    """ Return vertices of unit cube in R^n.
    """
    subsets = []
    for i in range(2 ** n):
        L = list(bin(i)[2:])
        L = [0] * (n - len(L)) + [int(l) for l in L]
        subsets += [L]
    return np.array(subsets)

def random_zonotope(n,d,scale=None):
    """ Create a random Zonotope of rank n in R^d
    """
    if scale is None:
        scale = np.sqrt(1/n)

    Az = scale*np.random.rand(n,d)
    Z = Zonotope(Az)
    return Z

def random_polytope(k,d,scale=1):
    """ Create a random polytope as a convex hull of k
    random points.
    """
    pts = scale*np.random.rand(k,d)
    P = Polytope(pts=pts)
    return P


class Halfspace:
    """ A halfspace of the form a \dot x + c <= 0
    """
    def __init__(self,a,c):
        self.tolerance = TOLERANCE
        self.a = a
        self.c = c

    def contains(self,x):
        if self.a@x + self.c <= 0:
            return True
        return False

    def boundary_contains(self,x):
        if abs(self.a@x + self.c) < self.tolerance:
            return True
        return False

    @property
    def is_vertical(self):
        """ Check if halfspace is vertically oriented (w.r.t. last coordinate)

        Should this depend on TOLERANCE?
        """
        return np.round(self.a[-1],decimals=8) > 0


class Polytope:
    """ A polytope in V representation.
    """
    def __init__(self, hull: ConvexHull=None, pts: np.array=None):
        self.tolerance = TOLERANCE
        assert hull is not None or pts is not None, "Must pass either a ConvexHull object or a list of points"
        if hull is None:
            self.hull = ConvexHull(pts)
        else:
            self.hull = hull
        
    @property
    def vertices(self):
        return self.pts[self.hull.vertices]

    @property
    def hyperplanes(self):
        inequalities = [e for e in np.unique(self.hull.equations, axis=0)]
        return [Halfspace(eq[:-1],eq[-1]) for eq in inequalities]

    @property
    def upper_hull(self):
        """ Get the upper hull hyperplanes of P """
        return [h for h in self.hyperplanes if h.is_vertical]

    @property
    def dim(self):
        return len(self.pts[0])

    @property
    def pts(self):
        return self.hull.points

    @property
    def barycenter(self):
        """ Compute the barycenter of the zonotope.
        """
        return np.sum(self.vertices,axis=0)/len(self.vertices)


    def incident_hyperplanes(self, x):
        """ Return the supporting hyperplanes to x. There might be more efficient
        ways to compute this (e.g. Simplex method).

        Hyperplanes will be in the form (a,c) where a is the normal and c is the offset.
        The associated halfspace is defined by a \dot x + c <= 0.
        """
        hyperplanes = []
        for eq in self.hull.equations:
            plane = Halfspace(eq[:-1],eq[-1])
            if plane.boundary_contains(x):
                hyperplanes.append(plane)

        return hyperplanes

    def has_vertex(self, x):
        """ Check if x is a vetex of P """
        # exactly a vertex
        if x in self.vertices:
            return True

        # approximately a vertex:
        if len(self.incident_hyperplanes(x)) >= self.dim:
            return True

        return False
    
    def get_pt_idx(self, x):
        """ Get index of vertex x in P."""
        dists = [np.linalg.norm(p-x) for p in self.pts]
        if np.min(dists) < self.tolerance:
            return np.argmin(dists)
        else:
            raise ValueError("Passed point is not a point of P")

    def bounds(self):
        """ Return the max and min values of all coordinates in P.
        """
        maxs = [-np.infty for _ in range(self.dim)]
        mins = [np.infty for _ in range(self.dim)]

        for pt in self.vertices:
            for i in range(self.dim):
                if pt[i] > maxs[i]:
                    maxs[i] = pt[i]
                if pt[i] < mins[i]:
                    mins[i] = pt[i]
        return maxs,mins

    def translate(self,v):
        """Translate whole polytope by v"""
        for p in self.pts:
            p+= v
        self.hull = ConvexHull(self.pts)

    def sym(self,O=None):
        """ Symmetrize P by taking the reflection of all vertices
        about O and including those in the convex hull. If dim=2,
        this will return a zonotope. (In general centrally symmetric
        bodies in dim>2 aren't necessarily zonotopes).
        """
        if O is None:
            O = self.barycenter

        reflect = lambda p : 2*O-p
        new_pts = []
        for pt in self.pts:
            new_pts += [pt,reflect(pt)]

        if self.dim == 2:
            return Zonotope(pts=new_pts)
        else:
            return Polytope(pts=new_pts)


class Zonotope(Polytope):
    """ A zonotope in V representation.
    """
    def __init__(self,generators=None, pts=None):
        
        self._generators = generators
        if not pts:
            assert generators is not None, "If not passing points, must pass generators to construct Zonotope."
            pts = np.array([(self.generators).T@e for e in cube(self.rank)])
        Z = ConvexHull(pts)

        super().__init__(hull=Z)

    @property
    def generators(self):
        if self._generators is None:
            self._generators = self._get_generators()
        return self._generators

    @property
    def rank(self):
        return(len(self.generators))

    def _get_generators(self):
        """ Compute the generators from self.polytope.
        This is somewhat nontrivial.
        """
        raise NotImplementedError

    def reflected_pt(self, idx):
        """ Return the index of the vertex across from the barycenter
        """
        ref = 2*self.barycenter - self.pts[idx]
        for i,pt in enumerate(self.pts):
            err = np.abs(pt-ref)
            if max(err) < self.tolerance:
                return i
        raise Exception("Could not compute reflected point, polytope is not centrally symmetric.")

    def move_pt(self, idx, v, normalize=False):
        """ Send self.pts[idx] to self.pts[idx] + v and symmetrize
        """
        if np.linalg.norm(v) == 0:
            return 

        self._generators = None #generators have to be recomputed
        reflected_idx = self.reflected_pt(idx)

        # move point
        barycenter = self.barycenter
        self.pts[idx] += v
        #symmetrize
        self.pts[reflected_idx] = 2*barycenter - self.pts[idx]
        self.hull = ConvexHull(self.vertices) # to not increase rank
        #self.hull = ConvexHull(self.pts) # might increase or decrease rank
