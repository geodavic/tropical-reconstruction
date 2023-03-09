import numpy as np
import sympy as sp
from sympy.core.numbers import Number
from sympy import Integer
from sympy.core import Expr
from polytope import Zonotope
from utils import all_subsets


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
        grad = np.append(grad,[self.normal_vec],axis=0)
        return grad

    def _partial(self, k, i, j):
        return 1 if j == k and i in self.pt_subset else 0


class ZonotopeFacetGradient(ZonotopeGradient):
    """
    Facet gradient class for a Zonotope.
    This one's a bit more complicated...
    """

    def __init__(self, zonotope, pt_subset, face_subset, v):
        """
        Parameters
        ----------

        pt_subset: list
            Subset of indices of generators corresponding to a vertex
            on the target face (the smallest point on which the target 
            point lies). 
        face_subset: list
            Subset of indices of generators corresponding to the target
            face.
        """
        super().__init__(zonotope)
        self.pt_subset = pt_subset
        self.face_subset = face_subset
        self.v = v
        self.symbols = [
            [sp.Symbol(f"g{i}{j}") for j in range(self.d)] for i in range(self.n)
        ]

    def _facet_normal(self, subset, evaluate=False):
        """
        Calculate normal vector to facet of a Zonotope in R^d
        corresponding to subset of generators
        """
        assert len(subset) == self.d - 1

        slice_generator = lambda i, g: g[:i] + g[(i + 1) :]

        nu = []
        for i in range(self.d):
            mat = sp.Matrix([slice_generator(i, self.symbols[j]) for j in subset])
            nu.append((-1) ** (i + 1) * sp.det(mat))

        if evaluate:
            return self._evaluate(nu)
        else:
            return sp.Array(nu)

    def _grad_facet_normal(self, subset, evaluate=False):
        """
        Calculate the gradient of the facet normal.

        Something something Jacobi's formula
        """
        nu = self._facet_normal(subset)
        grad = [
            [
                [sp.diff(t, self.symbols[i][j]) for j in range(self.d)]
                for i in range(self.n)
            ]
            for t in nu
        ]
        for j in range(len(grad)):
            grad[j] += [[Integer(0)]*self.d]

        if evaluate:
            return self._evaluate(grad)
        else:
            return sp.Array(grad)

    def __call__(self):
        """
        Return the value of the gradient
        """
        eta = self._facet_normal(self.face_subset, evaluate=True)
        etaI = []
        for i in range(self.n):
            if i in self.pt_subset:
                etaI += [eta]
            else:
                etaI += [[0] * self.d]
        etaI += [eta]
        etaI = np.array(etaI)

        first_term = np.zeros((self.n+1, self.d))
        grad_eta = np.array(self._grad_facet_normal(self.face_subset, evaluate=True))
        for j in range(self.d):
            mult = self.v[j] - self.zonotope.mu[j] - sum([self.generators[i][j] for i in self.pt_subset])
            first_term += mult * grad_eta[j]

        return first_term - etaI

    def _evaluate(self, expr):
        """Kinda messy"""
        if not isinstance(expr, Expr) and not isinstance(expr, Number):
            return [self._evaluate(t) for t in expr]

        if isinstance(expr, Number):
            return float(expr)
        else:
            for symb, g in zip(self.symbols, self.generators):
                for sg, gi in zip(symb, g):
                    expr = expr.subs(sg, gi)
            return self._evaluate(expr)
