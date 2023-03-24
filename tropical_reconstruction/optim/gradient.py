import numpy as np
import sympy as sp
from sympy.core.numbers import Number
from sympy import Integer
from sympy.core import Expr
from tropical_reconstruction.polytope import Zonotope, TOLERANCE
from tropical_reconstruction.utils import all_subsets


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


class ZonotopePointGradient(ZonotopeGradient):
    """
    Point gradient class for a Zonotope.

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
        Return the value of the gradient
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


class ZonotopeFacetGradient(ZonotopeGradient):
    """
    Facet gradient class for a Zonotope, used in the normal cone hack approximation
    to the gradient.

    This one's a bit more complicated...

    """

    def __init__(self, zonotope, hyperplane):
        """
        Parameters
        ----------

        hyperplane: Hyperplane
            Supporting hyperplane of control facet.
        """
        super().__init__(zonotope)
        self.hyperplane = hyperplane
        self.face_subset = zonotope.get_facet_generators(hyperplane)
        self.pt_subset = self._get_sample_pt_subset()
        self.symbols = [
            [sp.Symbol(f"g{i}{j}", real=True) for j in range(self.d)]
            for i in range(self.n)
        ]
        self.mu_symbols = [sp.Symbol(f"mu{j}", real=True) for j in range(self.d)]
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
        corresponding to self.face_subset.
        """

        slice_generator = lambda i, g: g[:i] + g[(i + 1) :]

        nu = []
        for i in range(self.d):
            mat = sp.Matrix(
                [slice_generator(i, self.symbols[j]) for j in self.face_subset]
            )
            nu.append((-1) ** (i + 1) * sp.det(mat))

        nu = sp.Matrix(nu)

        # nu should point outwards. The Determinant formula is ambiguous
        # on the sign of nu. It should be the same sign as the normal
        # vector to self.hyperplane
        sign = np.sign(self.hyperplane.a[0] / self._evaluate(nu)[0])

        if normalize:
            nu /= nu.norm(2)
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
        p = sp.Matrix(p)
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
        v = sp.Matrix(v)
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
                [sp.diff(t, self.symbols[i][j]) for j in range(self.d)]
                for i in range(self.n)
            ]
            for t in nu
        ]
        for j in range(len(grad)):
            grad[j] += [[Integer(0)] * self.d]

        if evaluate:
            return self._evaluate(grad)
        else:
            return sp.Matrix(grad)

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
            [sp.diff(d, self.symbols[i][j]) for j in range(self.d)]
            for i in range(self.n)
        ]
        grad += [[sp.diff(d, self.mu_symbols[j]) for j in range(self.d)]]
        return np.array(self._evaluate(grad))

    def __call__(self, v, explicit_method=True):
        if explicit_method:
            return self._distance_gradient_explicit(v)
        return self._distance_gradient_naiive(v)

    def _evaluate(self, expr):
        """Kinda messy"""
        if not isinstance(expr, Expr) and not isinstance(expr, Number):
            return [self._evaluate(t) for t in expr]

        if isinstance(expr, Number):
            return float(expr)
        else:
            vals = np.append(self.generators.flatten() , self.mu)
            var = self.all_symbols_flat
            expr = expr.evalf(subs=dict(zip(var,vals)))
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
