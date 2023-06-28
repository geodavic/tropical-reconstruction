from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import numpy as np
from copy import copy
from tropical_reconstruction.tzlp import solve_tzlp, solve_algebraic_reconstruction
from tropical_reconstruction.polytope import Polytope, Zonotope, TOLERANCE

from tropical_reconstruction.utils import all_subsets
from tropical_reconstruction.utils.draw import draw_polytope
from typing import List


class TropicalPolynomial:
    """A max-plus tropical polynomial."""

    def __init__(self, monomials, coeffs):
        """
        Parameters
        ---------
        monomials : List[tuple]
            The monomials of the tropical polynomial.
        coeffs : List[float]
            The coefficients of the monomials.

        This uses monomials and coeffs to build a dictionary (`poly`)
        that represents the tropical polynomial. This class is atomic to `poly`.

        TODO: change `legendre` to something more correct
        """
        self.input_dim = len(monomials[0])
        self._set_poly(monomials, coeffs)
        self._zonotope = None  # if Newt(f) is a zonotope

    def _set_poly(self, mons, coeffs):
        """Set the dictionary defining the tropical polynomial, asserting
        the appropriate constraints.
        """
        assert len(mons) == len(coeffs), "Monomials and coefficients must be bijection."
        for m, c in zip(mons, coeffs):
            mon_np = np.array(m)
            assert (mon_np >= 0).all(), "Monomials must all have non-negative powers."
            assert (
                len(m) == self.input_dim
            ), "Monomials must all be in the same dimension."
            assert c > -np.infty, "Coefficients must be finite."
            assert c < np.infty, "Coefficients must be finite."

        self.poly = self._zip(mons, coeffs)

    def _zip(self, mons, coeffs):
        """Create the poly dictionary, combining like terms."""
        poly = {}
        for m, c in zip(mons, coeffs):
            if m in poly:
                poly[m] = max(c, poly[m])
            else:
                poly[m] = c
        return poly

    def __add__(self, g):
        """Add another polynomial (tropically).

        TODO: handle floats and ints
        """
        if not isinstance(g, TropicalPolynomial):
            g = TropicalPolynomial([(0,) * self.input_dim], [g])

        new_poly = {}
        new_mons = list(g.poly) + list(self.poly)
        new_coefs = list(g.poly.values()) + list(self.poly.values())
        return TropicalPolynomial(new_mons, new_coefs)

    def __mul__(self, g):
        """Multiply by another polynomial (tropically).

        TODO: handle floats and ints
        """
        new_poly = {}
        if isinstance(g, float) or isinstance(g, int):
            g = TropicalPolynomial([ (0,0) ], [g])

        # Handle monomial case as base case
        if len(g.poly) == 1:
            g_mon = list(g.poly)[0]
            for m, c in self.poly.items():
                new_mon = tuple([a + b for a, b in zip(m, g_mon)])
                new_poly[new_mon] = c + g.poly[g_mon]
            return TropicalPolynomial(list(new_poly.keys()), list(new_poly.values()))

        # Multiply self by each monomial of g
        summands = []
        for mon in g.poly:
            monf = self * (TropicalPolynomial([mon], [g.poly[mon]]))
            summands += [monf]

        # Sum them up to get final product
        product = summands[0]
        for s in summands[1:]:
            product += s
        return product

    def __rmul__(self, g):
        """Same as __mul__ (tropical semiring is commutative)"""
        return self.__mul__(g)

    def __eq__(self, g):
        """Check if f is equal to g algebraically (i.e. they have identical
        algebraic expressions)
        """
        if isinstance(g, self.__class__):
            return self.poly == g.poly
        else:
            return False

    def __call__(self, x):
        """
        Evaluate polynomial on x.
        """
        if type(x) != tuple:
            try:
                x = tuple(x)
            except:
                x = (x,)
        assert len(x) == self.input_dim, "Input has improper dimension."
        evals = [c + np.array(m) @ x for m, c in self.poly.items()]
        return max(evals)

    def __len__(self):
        return len(self.poly)

    def copy(self):
        """Return a copy of self."""
        return TropicalPolynomial(list(self.poly.keys()), list(self.poly.values()))

    def simplify(self):
        """Return a polynomial consisting of only active monomials. This is another
        tropical polynomial that is pointwise equal to f.

        Note: this implementation is imperfect when the number of vertices is small
        (i.e. when the lifted newton polytop is not full dimensional) and in this case
        this will simply return a copy of self.
        """
        try:
            active_vertices = self.legendre_vertices
            monomials = [tuple(m) for m in active_vertices[:, :-1]]
            coeffs = active_vertices[:, -1]
            return TropicalPolynomial(monomials, coeffs)
        except QhullError:
            return self.copy()

    def coef(self, mon):
        """Retrieve the coefficient of the specified monomial."""
        return self.poly.get(tuple(mon), -np.infty)

    @property
    def constant_term(self):
        """Retrieve the constant term of the polynomial."""
        mon = (0,) * self.input_dim
        return self.coef(mon)

    def power(self, alpha, lazy=False):
        """Tropically raise f to a power.

        lazy: bool
            If true, returns a polynomial that is pointwise equal to f^alpha,
            but isn't necessarily equal to f^alpha algebraically.

        Note if alpha is not an integer, lazy must be true (otherwise
        the power isn't well-defined)
        """
        new_poly = {}

        isint = isinstance(alpha, int) or isinstance(alpha, np.int64)
        if not isint and not lazy:
            raise ValueError(
                "Cannot lazily raise a tropical polynomial to a non-integer power (try passing lazy=True)"
            )

        if lazy:
            for m, c in self.poly.items():
                new_mon = tuple([alpha * i for i in m])
                new_poly[new_mon] = c * alpha
            return TropicalPolynomial(list(new_poly.keys()), list(new_poly.values()))
        else:
            g = self.copy()
            for _ in range(alpha - 1):
                g = self * g
            return g

    @property
    def newton_polytope(self):
        """Return the Newton polytope."""
        pts = [np.array(m) for m, _ in self.poly.items()]
        P = Polytope(pts=pts)
        return P

    @property
    def lifted_newton_polytope(self):
        """Return the lifted Newton polytope."""
        pts = [np.array(m + (c,)) for m, c in self.poly.items()]
        P = Polytope(pts=pts)
        return P

    @property
    def legendre(self):
        """Return the legendre transform of the polynomial represented
        as a collection of hyperplanes.
        """
        return self.lifted_newton_polytope.upper_hull

    @property
    def legendre_vertices(self):
        """Return the vertices of the legendre transform (the vertices
        of the upper hull of the lifted newton polytopoe).
        """
        return self.lifted_newton_polytope.upper_hull_vertices

    @property
    def dual_diagram(self):
        """Return the dual diagram of V(f), represented as the regular
        subdivision of the lifted newton polytope of f.
        """
        return self.lifted_newton_polytope.regular_subdivision

    @property
    def zonotope(self):
        """Checks if the newton polytope of f is a zonotope. If yes,
        then return the generators of that zonotope. Otherwise return None.

        TODO: assert that the zonotope is positive and contains origin
        """
        if self._zonotope is None:
            Z = Zonotope(pts=self.newton_polytope.pts)
            try:
                Z.generators.T
                self._zonotope = Z
            except:
                self._zonotope = None

        return self._zonotope

    def _get_tzlp_data(self, verbose=True):
        """Get the data (Qz,U,Epsilon,z0) necessary to set up the TZLP associated to
        this polynomial.
        """
        if self.zonotope is not None:
            upper_hull_vertices = self.legendre_vertices
            U = [list(v) for v in upper_hull_vertices if sum(v[:-1])]
            Qz = self.zonotope.generators.T.astype(np.float64)
            d, n = Qz.shape
            z0 = [0] * d + [self.constant_term]
            Epsilon = []

            subsets = all_subsets(n)
            for u in U:
                m = u[:-1]
                for eps in subsets:
                    if np.sum(np.abs(Qz @ eps - m)) < TOLERANCE:
                        Epsilon.append(eps)
                        break
            if len(Epsilon) != len(U):
                if verbose:
                    print(
                        "Warning: some vertices of the upper hull do not project to cubical vertices of the newton polytope. The TZLP is not well-posed."
                    )
                return None
            return Qz.tolist(), U, z0, Epsilon
        else:
            if verbose:
                print("Unable to express the Newton polytope of f as a zonotope")
            return None

    def neural_network(self, b=None, verbose=False):
        """Return a (d,n,1) NeuralNetwork representation of f, is possible. This solves the
        TZLP associated to f, and then the algebraic reconstruction problem.

        TODO: allow for d=1 architectures.
        """
        if self.input_dim == 1:
            raise NotImplementedError("Only implemented for (d,n,1) for d>1")

        data = self._get_tzlp_data(verbose=verbose)
        if data is None:
            raise Exception()

        sol = solve_tzlp(data, verbose=verbose)
        if sol is None:
            raise Exception("Could not set up TZLP.")

        QZ, c = sol
        z0 = np.array([0] * (QZ.shape[0] - 1) + [self.constant_term])
        weights, thresholds = solve_algebraic_reconstruction(QZ, c, z0, b=b)

        return PolynomialNeuralNetwork(weights, thresholds)

class TropicalRationalFunction:
    def __init__(self, f: TropicalPolynomial, g: TropicalPolynomial):
        self.numerator = f
        self.denominator = g

    def __call__(self, x):
        return self.numerator(x) - self.denominator(x)

class NeuralNetwork:
    """ A fully connected homogenous Neural Network with ReLU activation """

    def __init__(self, weights, thresholds):
        """
        Parameters
        ----------
        weights : List[np.array]
            The list of weight matrices.
        thresholds : List[np.array]
            The list of thresholds for ReLU.
        """
        self.weights = weights
        self.thresholds = thresholds
        self.depth = len(weights)
        self.output_dim = len(weights[-1])
        self.input_dim = len(weights[0][0])

        assert len(weights) == len(
            thresholds
        ), "Weights and thresholds must be in bijection."
        for A, t in zip(self.weights, self.thresholds):
            assert len(A) == len(
                t
            ), "At least one pair (A,t) of weights and thresholds are dimensionally incompatible"

    @property
    def architecture(self):
        """Return the architecture tuple"""
        arch = (self.input_dim,)
        for w in self.weights:
            arch = arch + (len(w),)
        return arch

    @property
    def complexity(self):
        """Return number of parameters."""
        weights_complexity = sum([w.size for w in self.weights])
        thresh_complexity = sum([t.size for t in self.thresholds])
        return weights_complexity + thresh_complexity

    def component(self, i):
        """Return a neural network function which is the ith component
        of the current network.
        """
        if self.output_dim == 1:
            return self

        new_weights = []
        new_thresh = []
        c = 0
        for A, t in zip(self.weights, self.thresholds):
            if c == len(self.weights):
                new_weights += [np.copy(A[i])]
                new_thresh += [np.copy(t[i])]
            else:
                new_weights += np.copy(A)
                new_thresh += np.copy(t)
            c += 1

        return type(self)(new_weights, new_thresh)

    def __call__(self, x):
        """Evaluate the Neural Network at x."""
        assert len(x) == self.input_dim, "Input has incorrect dimension."
        if type(x) == np.ndarray:
            ret = np.copy(x)
        else:
            ret = np.array(x)

        for A, t in zip(self.weights, self.thresholds):
            ret = np.maximum(A @ ret, t)
        return ret

    def _multiply_row(self, row: List[int],polys: List[TropicalPolynomial]) -> TropicalPolynomial:
        P = [p.power(a, lazy=True) for a, p in zip(row, polys)]
        prod = P[0]
        for p in P[1:]:
            prod = prod * p
            prod = prod.simplify()
        return prod

    def tropical(self):
        """
        Return a pair of tropical polynomials f,g such that this network
        is equal to f - g pointwise.
        """
        var_xi = lambda i: tuple([int(i == j) for j in range(self.input_dim)])
        # F starts off as coordinate functions
        polysF = [TropicalPolynomial([var_xi(i)], [0]) for i in range(self.input_dim)]
        # G starts off at zero
        polysG = [TropicalPolynomial([(0,)*self.input_dim], [0]) for i in range(self.input_dim)]

        # Recursively calculate polynomials at each layer
        for L in range(self.depth):
            new_polysF = []
            new_polysG = []
            
            for row, thresh in zip(self.weights[L], self.thresholds[L]):
                assert len(row) == len(polysF), f"Dimension mismatch with weights at layer {L+2}"
                assert len(row) == len(polysG), f"Dimension mismatch with weights at layer {L+2}"
                row_plus = 0.5*(row+abs(row))
                row_neg = -0.5*(row-abs(row))

                Apf = self._multiply_row(row_plus, polysF)
                Apg = self._multiply_row(row_plus, polysG)
                Anf = self._multiply_row(row_neg, polysF)
                Ang = self._multiply_row(row_neg, polysG)
                
                new_polysF += [ (Apf * Ang) + (thresh * Anf * Apg) ]
                new_polysG += [ Anf * Apg ]

            polysF = new_polysF
            polysG = new_polysG

        return [TropicalRationalFunction(f,g) for f,g in zip(new_polysF, new_polysG)]


class PolynomialNeuralNetwork(NeuralNetwork):
    """
    A fully connected homogenous Neural Network with non-negative weights
    and ReLU activation.
    """

    def __init__(self, weights, thresholds):
        super().__init__(weights, thresholds)
        for A, t in zip(self.weights, self.thresholds):
            assert (A >= 0).all(), "All weights must be nonnegative."


    def tropical(self, verbose=False):
        """Return the associated tropical polynomial(s) to the network."""

        # View the coordinate functions (xi) as tropical polynomials
        var_xi = lambda i: tuple([int(i == j) for j in range(self.input_dim)])
        polys = [TropicalPolynomial([var_xi(i)], [0]) for i in range(self.input_dim)]

        # Recursively calculate tropical polynomials at each layer
        for L in range(self.depth):
            new_polys = []
            for row, thresh in zip(self.weights[L], self.thresholds[L]):
                # Compute row @ new_polys
                assert len(row) == len(
                    polys
                ), f"Dimension mismatch with weights at layer {L+2}"
                prod = self._multiply_row(row, polys)

                if verbose:
                    if prod.constant_term == (prod + thresh).constant_term:
                        print(f" - (threshold at layer {L} did nothing)")
                    else:
                        print(f" x (threshold at layer {L} did something)")
                prod = prod + thresh
                new_polys += [prod]

            polys = new_polys

        # In depth two case, manually set the zonotope,
        # since we know it a-priori.
        if self.depth == 2:
            for j, f in enumerate(polys):
                A = self.weights[0]
                B = self.weights[1][j]
                generators = np.array([B[i] * A[i] for i in range(len(A))])
                f._zonotope = Zonotope(generators=generators)

        return polys


def test_equal(f1, f2, input_dim, n_samples=10000, size=500):
    """Test if two functions are equal pointwise by checking a bunch of
    random points inside [-500,500]^d.
    """
    for _ in range(n_samples):
        x = 2 * size * np.random.rand(input_dim) - size
        diff = abs(f1(x) - f2(x))
        if diff > 1e-8:
            print(f"Failed at x = {x} (difference of {diff})")
            return False
    return True
