import numpy as np
import symengine as se
from symengine import Integer, Number, Expr
from tropical_reconstruction.polytope import Zonotope, TOLERANCE
from tropical_reconstruction.utils import all_subsets
from tropical_reconstruction.metrics import distance_to_polytope_l2


class ZonotopeGradient:
    """
    Base class for gradients in Zonotope space.
    """

    def __init__(self, zonotope: Zonotope):
        self.zonotope = zonotope
        self.d = len(zonotope.generators[0])
        self.n = len(zonotope.generators)

    @property
    def generators(self):
        return self.zonotope.generators

    @property
    def mu(self):
        return self.zonotope.mu


class ZonotopeVertexGradient(ZonotopeGradient):
    """
    Vertex gradient class for a Zonotope.

    In this case, the control point is a vertex of the zonotope.
    """

    def __init__(self, zonotope, pt_subset, normal_vec):
        """
        Parameters
        ----------

        zonotope: Zonotope
            The zonotope being optimized (current best guess)
        pt_subset: list
            List of generator indices such that the sum of the
            corresponding generators is the control point.
        normal_vec: np.ndarray
            Direction that control point is to be moved. This is
            the normal vector of the target polytope face P on
            which the target point lies.

        Note: normal_vec should be the outward normal, not inward.
        """
        self.pt_subset = pt_subset
        self.normal_vec = normal_vec

        super().__init__(zonotope)

    def __call__(self):
        """
        Return the value of the gradient of the distance between the control
        point and the target face.
        """
        grad = np.zeros((self.n, self.d))
        for i in range(self.n):
            for j in range(self.d):
                grad[i][j] = sum(
                    [self.normal_vec[k] * self._partial(k, i, j) for k in range(self.d)]
                )
        grad = np.append(grad, [self.normal_vec], axis=0)
        return grad

    def _partial(self, k, i, j):
        return 1 if j == k and i in self.pt_subset else 0


class ZonotopeBoundaryPointGradient(ZonotopeGradient):
    """
    Boundary point gradient class for a Zonotope.

    In this case, the control point is a point on the boundary of the zonotope.
    This calculates the gradient that moves the control point to the target
    point explicitly.
    """

    def __init__(self, zonotope, control_pt):
        """
        Parameters
        ----------

        zonotope: Zonotope
            The zonotope being optimized
        control_pt:
            The control point on the zonotope.
        """
        super().__init__(zonotope)
        _,_,self.sum_coeffs = distance_to_polytope_l2(control_pt, zonotope)
        assert len(self.sum_coeffs) == len(self.zonotope.vertices), "Convex sum coefficients not in bijection with zonotope vertices"

    def __call__(self, v):
        """
        Calculate the gradient of the squared distance between the control
        point and v.
        """
        q = sum([lam*vert for lam,vert in zip(self.sum_coeffs,self.zonotope.vertices)])
        I = lambda k: self.zonotope.pts_subsets(self.zonotope.get_pt_idx(self.zonotope.vertices[k]), binary=False)

        grad = np.zeros((self.n+1, self.d))
        for i in range(self.n):
            for j in range(self.d):
                grad[i][j] = (v[j] - q[j])*sum([self.sum_coeffs[k] for k in range(len(self.sum_coeffs)) if i in I(k)])

        for j in range(self.d):
            grad[self.n][j] = v[j] - q[j]

        return grad
        

class ZonotopeFacetGradient(ZonotopeGradient):
    """
    Facet gradient class for a Zonotope.

    In this case, the control point is on a facet of the zonotope. This 
    calculates a gradient that moves the entire facet on which the control
    point lies.

    Question: is this the same gradient as ZonotopeBoundaryPointGradient?
    It would be interesting if it wasn't...
    """

    def __init__(self, zonotope, hyperplane):
        """
        Parameters
        ----------

        hyperplane: Hyperplane
            Supporting hyperplane of control facet.
        """
        super().__init__(zonotope)
        if self.d > self.n:
            raise ValueError("This gradient method is not valid when the zonotope rank is less than the ambient dimension.")
        self.hyperplane = hyperplane
        self.face_subset = zonotope.get_facet_generators(hyperplane)
        self.pt_subset = self._get_sample_pt_subset()
        self.symbols = [
            [se.Symbol(f"g{i}{j}", real=True) for j in range(self.d)]
            for i in range(self.n)
        ]
        self.mu_symbols = [se.Symbol(f"mu{j}", real=True) for j in range(self.d)]
        self.all_symbols_flat = [s for g in self.symbols for s in g] + [s for s in self.mu_symbols]

        self._facet_normal = None
        self._facet_normal_evaluated = None
        self._offset = None
        self._offset_evaluated = None

    def _get_sample_pt_subset(self):
        """
        Get any vertex on the control facet.
        """
        for v in self.zonotope.vertices:
            if self.hyperplane.boundary_contains(v):
                idx = self.zonotope.get_pt_idx(v, force=True)
                return self.zonotope.pts_subsets(idx, binary=False)

        raise Exception("Given hyperplane does not support any faces of the zonotope")

    @property
    def facet_normal(self):
        if self._facet_normal is None:
            self._facet_normal = self._calculate_facet_normal(evaluate=False)
        return self._facet_normal

    @property
    def facet_normal_evaluated(self):
        if self._facet_normal_evaluated is None:
            self._facet_normal_evaluated = self._evaluate(self.facet_normal)
        return self._facet_normal_evaluated

    def _calculate_facet_normal(self, evaluate=False, normalize=True):
        """
        Calculate normal vector to facet of the Zonotope
        correseonding to self.face_subset.
        """

        slice_generator = lambda i, g: g[:i] + g[(i + 1) :]

        nu = []
        for i in range(self.d):
            mat = se.Matrix(
                [slice_generator(i, self.symbols[j]) for j in self.face_subset]
            )
            nu.append((-1) ** (i + 1) * mat.det())

        nu = se.Matrix(nu)

        # nu should point outwards. The Determinant formula is ambiguous
        # on the sign of nu. It should be the same sign as the normal
        # vector to self.hyperplane
        nu_eval = self._evaluate(nu)
        nz_idx = 0
        mult = nu_eval[nz_idx]
        while abs(mult) < TOLERANCE:
            nz_idx += 1
            mult = nu_eval[nz_idx]

        sign = np.sign(self.hyperplane.a[0] / mult)

        if normalize:
            nu /= se.sqrt(sum([a**2 for a in nu.ravel()]))
        nu *= sign

        if evaluate:
            return self._evaluate(nu)
        else:
            return nu

    @property
    def offset(self):
        if self._offset is None:
            self._offset = self._calculate_offset(evaluate=False)
        return self._offset

    @property
    def offset_evaluated(self):
        if self._offset_evaluated is None:
            self._offset_evaluated = self._calculate_offset(evaluate=True)
        return self._offset_evaluated

    def _calculate_offset(self, evaluate=False):
        """
        Calculate the offset value to the supporting hyperplane of the
        face of the zonotope. In other words, if eta is the normal vector
        to the hyperplane, then calculate c in the following equation

        hyperplane = \{ x | <eta, x> = c \}
        """
        p = []
        for j in range(self.d):
            pj = self.mu_symbols[j]
            for i in self.pt_subset:
                pj += self.symbols[i][j]
            p.append(pj)
        p = se.Matrix(p)
        eta = self.facet_normal

        c = eta.T @ p

        if evaluate:
            return self._evaluate(c[0])
        else:
            return c[0]

    def _distance_to_pt(self, v, evaluate=False):
        """
        Calculate distance from v to the supporting hyperplane of the control face.
        """
        v = se.Matrix(list(v))
        eta = self.facet_normal
        d = (eta.T @ v)[0] - self.offset

        if evaluate:
            return self._evaluate(d)
        else:
            return d

    def _grad_facet_normal(self, evaluate=False):
        """
        Calculate the gradient of the facet normal.

        Something something Jacobi's formula
        """
        nu = self.facet_normal
        grad = [
            [
                [se.diff(t, self.symbols[i][j]) for j in range(self.d)]
                for i in range(self.n)
            ]
            for t in nu
        ]
        for j in range(len(grad)):
            grad[j] += [[Integer(0)] * self.d]

        if evaluate:
            return self._evaluate(grad)
        else:
            return se.Matrix(grad) #<- this is bugged, can't convert to se.Matrix

    def _distance_gradient_explicit(self, v):
        """
        Return the gradient of d(Z,v) (original method, accelerated a bit
        by some precomputation).
        """
        grad_eta = np.array(self._grad_facet_normal(evaluate=True))
        eta = self.facet_normal_evaluated

        etaI = []
        for i in range(self.n):
            if i in self.pt_subset:
                etaI += [eta]
            else:
                etaI += [[0] * self.d]
        etaI += [eta]
        etaI = np.array(etaI)

        first_term = np.zeros((self.n + 1, self.d))
        for j in range(self.d):
            mult = (
                v[j]
                - self.zonotope.mu[j]
                - sum([self.generators[i][j] for i in self.pt_subset])
            )
            first_term += mult * grad_eta[j]

        return first_term - etaI

    def _distance_gradient_naiive(self, v):
        """
        Return the gradient of d(Z,v) (new method, by naievely differentiating
        the distance directly). This is a bit slower than the explicit method.
        """
        d = self._distance_to_pt(v, evaluate=False)
        grad = [
            [se.diff(d, self.symbols[i][j]) for j in range(self.d)]
            for i in range(self.n)
        ]
        grad += [[se.diff(d, self.mu_symbols[j]) for j in range(self.d)]]
        return np.array(self._evaluate(grad))

    def __call__(self, v, explicit_method=True):
        if explicit_method:
            return self._distance_gradient_explicit(v)
        return self._distance_gradient_naiive(v)

    def _evaluate(self, expr):
        """Kinda messy"""
        try:
            return float(expr)
        except:
            pass
        if not isinstance(expr, Expr):
            return [self._evaluate(t) for t in expr]
        else:
            vals = np.append(self.generators.flatten() , self.mu)
            var = self.all_symbols_flat
            expr = expr.subs(dict(zip(var,vals)))
            """
            for symb, g in zip(self.symbols, self.generators):
                for sg, gi in zip(symb, g):
                    expr = expr.subs(sg, gi)
            for mu_symb, m in zip(self.mu_symbols, self.mu):
                expr = expr.subs(mu_symb, m)
            """

            return self._evaluate(expr)


def ZonotopeFaceGradient(ZonotopeGradient):
    """
    Generalized version of the gonotope facet gradient for when the control point
    lies on a possibly lower dimensional face, not necessarily a facet.

    This is the "correct" gradient, as opposed to using the normal cone hack.

    This is still TODO
    """

    def __init__(self, facet_gradients: list[ZonotopeFacetGradient]):
        """
        Parameters
        ----------
        facet_gradients: list[ZonotopeFacetGradient]
            The individual gradient objects for each facet that contains the face
            containing the control point.
        """
        assert (
            len(facet_gradients) > 0
        ), "Number of facets defining the face must be nonzero"
        super().__init__(facet_gradients[0].zonotope)
        assert (
            len(facet_gradients) < self.d
        ), "Number of facets defining the face must be less than the ambient dimension (the face cannot be a point)"

        self.facet_gradients = facet_gradients
