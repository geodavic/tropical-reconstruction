from scipy.spatial import ConvexHull
import numpy as np
from copy import copy
from tzlp import TZLP_Solver
from polytope import Polytope, Zonotope, TOLERANCE

from utils import all_subsets


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
        """
        self.input_dim = len(monomials[0])
        self._set_poly(monomials, coeffs)
        self.Qz = None  # if Newt(f) is a zonotope, this is the generator matrix (see self.zonotope())

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

        poly = {}
        for m, c in zip(mons, coeffs):
            if m in poly:
                poly[m] = max(c, poly[m])
            else:
                poly[m] = c
        self.poly = poly

    def coef(self, mon):
        """Retrieve the coefficient of the specified monomial."""
        return self.poly.get(tuple(mon), -np.infty)

    def copy(self):
        """Return a copy of self."""
        return TropicalPolynomial(list(self.poly.keys()), list(self.poly.values()))

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

    def __add__(self, g):
        """Add another polynomial (tropically)."""
        if not isinstance(g, TropicalPolynomial):
            g = TropicalPolynomial([(0,) * self.input_dim], [g])

        new_poly = {}
        new_mons = list(g.poly) + list(self.poly)
        new_coefs = list(g.poly.values()) + list(self.poly.values())
        return TropicalPolynomial(new_mons, new_coefs)

    def __mul__(self, g):
        """Multiply by another polynomial (tropically)."""
        new_poly = {}

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

    def newton_polytope(self):
        """Return the Newton polytope."""
        pts = [np.array(m) for m, _ in self.poly.items()]
        P = Polytope(pts=pts)
        return P

    def lifted_newton_polytope(self):
        """Return the lifted Newton polytope."""
        pts = [np.array(m + (c,)) for m, c in self.poly.items()]
        P = Polytope(pts=pts)
        return P

    def legendre(self):
        """Return the legendre transform of the polynomial represented
        as a collection of hyperplanes.
        """
        planes = self.lifted_newton_polytope().upper_hull
        return planes

    def legendre_vertices(self):
        """Return the vertices of the legendre transform (the vertices
        of the upper hull of the lifted newton polytopoe).
        """
        upper_hull_vertices = []
        planes = self.legendre()
        newt = self.lifted_newton_polytope()
        vertices = newt.vertices
        for v in vertices:
            for h in planes:
                found_h = False
                if h.boundary_contains(v):
                    upper_hull_vertices.append(v)
                    found_h = True
                if found_h:
                    break
        return np.array(upper_hull_vertices)

    def zonotope(self):
        """Checks if the newton polytope of f is a zonotope. If yes,
        then return the generators of that zonotope. Otherwise return None.
        """
        if self.Qz is not None:
            return self.Qz
        else:
            return self._get_zonotope()

    def _get_zonotope(self):
        """Get the zonotope representation of Newt(f) (if it exists).
        """
        Z = Zonotope(self.newton_polytope.pts)
        try:
            return Z.generators
        except:
            return None

    def _get_tzlp_data(self):
        """Get the datum (Qz,U,Epsilon,z0) necessary to set up the TZLP associated to
        this polynomial.
        """
        if self.zonotope() is not None:
            upper_hull_vertices = self.legendre_vertices()
            U = [list(v) for v in upper_hull_vertices if sum(v[:-1])]
            Qz = self.zonotope().astype(np.float64)
            d, n = Qz.shape
            z0 = [0] * d + [self.constant_term()]
            Epsilon = []

            subsets = all_subsets(n)
            for u in U:
                m = u[:-1]
                for eps in subsets:
                    if np.sum(np.abs(Qz @ eps - m)) < TOLERANCE:
                        Epsilon.append(eps)
                        break
            return Qz.tolist(), U, z0, Epsilon
        else:
            return None

    def _solve_tzlp(self, verbose=True):
        data = self._get_tzlp_data()
        Qz = np.array(data[0])
        d, n = Qz.shape
        if n <= d + 1:
            print("Warning: n <= d+1, so assumptions of TZLP are not satisfied")
        if data is not None:
            tzlp = TZLP_Solver(*data)
            sol = tzlp.solve(verbose=verbose, numpy=True)

            if sol is not None:
                x, c = sol
                return (np.append(Qz, [x], axis=0), c)
            else:
                return None
        else:
            if verbose:
                print("Newton polytope of this polynomial is not a zonotope.")
            return None

    def _solve_algebraic_reconstruction(self, QZ, c, z0, b=None):
        """Solve the algebraic reconstruction problem given a solution (Q_Z,c,z_0)
        to the geometric reconstruction problem.
        """
        if b is None:
            b = np.ones(QZ.shape[1])

        A1 = np.array([r / b for r in QZ[:-1]]).T
        A2 = np.array([b])
        t1 = -QZ[-1] / b
        t2 = np.array([z0[-1]])
        return [A1, A2], [t1, t2]

    def neural_network(self, b=None, verbose=True):
        """Return a (d,n,1) NeuralNetwork representation of f, is possible. This solves the
        TZLP associated to f, and then the algebraic reconstruction problem.

        TODO: allow for different architectures.

        TODO: the t^2 term is wrong when going NN -> f -> NN', but NN = NN' pointwise still ..?
        """
        if self.input_dim > 1:
            QZ, c = self._solve_tzlp(verbose=verbose)
            z0 = np.array([0] * (QZ.shape[0] - 1) + [self.constant_term()])
            weights, thresholds = self._solve_algebraic_reconstruction(QZ, c, z0, b=b)
        else:
            raise NotImplementedError("Only implemented for (d,n,1) for d>1")

        return PolynomialNeuralNetwork(weights, thresholds)


class PolynomialNeuralNetwork:
    """
    A fully connected homogenous Neural Network with non-negative weights
    and ReLU activation.
    """

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
            assert (A >= 0).all(), "All weights must be nonnegative."
            assert len(A) == len(
                t
            ), "At least one pair (A,t) of weights and thresholds are dimensionally incompatible"

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

        return PolynomialNeuralNetwork(new_weights, new_thresh)

    def tropical(self):
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
                P = [p.power(a, lazy=True) for a, p in zip(row, polys)]
                prod = P[0]
                for p in P[1:]:
                    prod = prod * p

                prod = prod + thresh
                new_polys += [prod]

            polys = new_polys

        # In depth two case, manually set the zonotope generators,
        # since we know them a-priori. In other depths, this would be
        # invalid since the newton polytope is not necessarily a Zonotope
        if self.depth == 2:
            for j, f in enumerate(polys):
                A = self.weights[0]
                B = self.weights[1][j]
                f.Qz = np.array([B[i] * A[i] for i in range(len(A))]).T

        return polys

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


def test_equal(f1, f2, input_dim, n_samples=10000):
    """Test if two functions are equal pointwise by checking a bunch of
    random points inside [-500,500]^d.
    """
    for _ in range(n_samples):
        x = 1000 * np.random.rand(input_dim) - 500
        if abs(f1(x) - f2(x)) > 1e-10:
            print(f"Failed at x = {x}")
            return False
    return True
