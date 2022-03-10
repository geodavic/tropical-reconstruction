from scipy.spatial import ConvexHull
import numpy as np
from tzlp import TZLP_Solver

from utils import get_upper_hull, all_subsets


class TropicalPolynomial:
    """ A max-plus tropical polynomial.
    """
    def __init__(self,monomials,coeffs):
        """
        Parameters
        ---------
        monomials : List[tuple]
            The monomials of the tropical polynomial. 
        coeffs : List[float]
            The coefficients of the monomials.
        """
        self.poly = {m:c for m,c in zip(monomials,coeffs)}
        self.input_dim = len(monomials[0])
        self.Qz = None # if Newt(f) is a zonotope, this is the generator matrix (see self.zonotope())

        assert len(monomials) == len(coeffs), "Monomials and coefficients must be bijection."
        for m,c in zip(monomials,coeffs):
            mon_np = np.array(m)
            assert (mon_np>=0).all(), "Monomials must all have non-negative powers."
            assert len(m) == self.input_dim, "Monomials must all be in the same dimension."
            assert c > -np.infty, "Coefficients must be finite."
            assert c < np.infty, "Coefficients must be finite."

    def newton_polytope(self):
        """ Return the Newton polytope.
        """
        pts = [np.array(m) for m,_ in self.poly.items()]
        hull = ConvexHull(pts)
        return hull

    def lifted_newton_polytope(self):
        """ Return the lifted Newton polytope.
        """
        pts = [np.array(m+(c,)) for m,c in self.poly.items()]
        hull = ConvexHull(pts)
        return hull

    def legendre(self):
        """ Return the legendre transform of the polynomial represented
        as a collection of hyperplanes.
        """
        hull = self.lifted_newton_polytope()
        return get_upper_hull(hull)

    def legendre_vertices(self):
        """ Return the vertices of the legendre transform (the vertices
        of the upper hull of the lifted newton polytopoe).

        Uses a precision of 1e-10.
        """
        upper_hull_vertices = []
        equations = self.legendre()
        newt = self.lifted_newton_polytope()
        vertices = newt.points[newt.vertices]
        for v in vertices:
            for eq in equations:
                found_v = False
                if abs(v@eq[:-1] + eq[-1]) < 1e-10:
                    upper_hull_vertices.append(v)
                    found_v = True
                if found_v:
                    break
        return np.array(upper_hull_vertices)

    def zonotope(self):
        """ Checks if the newton polytope of f is a zonotope. If yes,
        then return the generators of that zonotope. Otherwise return None.
        """
        if self.Qz is not None:
            return self.Qz
        else:
            return self._get_zonotope()

    def _get_zonotope(self):
        """ Get the zonotope representation of Newt(f) (if it exists).
        """
        raise NotImplementedError

    def _get_tzlp_data(self):
        """ Get the datum (Qz,U,Epsilon,z0) necessary to set up the TZLP associated to
        this polynomial.
        """
        if self.zonotope() is not None:
            upper_hull_vertices = self.legendre_vertices()
            U = [list(v) for v in upper_hull_vertices if sum(v[:-1])]
            Qz = self.zonotope().astype(np.float64)
            d,n = Qz.shape
            z0 = [0]*d + [self.constant_term()]
            Epsilon = []

            subsets = all_subsets(n)
            for u in U:
                m = u[:-1]
                for eps in subsets:
                    if np.sum(np.abs(Qz@eps - m)) < 1e-10:
                        Epsilon.append(eps)
                        break
            return Qz.tolist(),U,z0,Epsilon
        else:
            return None

    def _solve_tzlp(self,verbose=True):
        data = self._get_tzlp_data()
        Qz = np.array(data[0])
        d,n = Qz.shape
        if n<= d+1:
            print("Warning: n <= d+1, so assumptions of TZLP are not satisfied")
        if data is not None:
            tzlp = TZLP_Solver(*data)
            sol = tzlp.solve(verbose=verbose,numpy=True)
            
            if sol is not None:
                x,c = sol
                return (np.append(Qz,[x],axis=0),c)
            else:
                return None
        else:
            if verbose:
                print("Newton polytope of this polynomial is not a zonotope.")
            return None

    def _solve_algebraic_reconstruction(self,QZ,c,z0,b=None):
        """ Solve the algebraic reconstruction problem given a solution (Q_Z,c,z_0)
        to the geometric reconstruction problem.
        """
        if b is None:
            b = np.ones(QZ.shape[1])

        A1 = np.array([r/b for r in QZ[:-1]]).T
        A2 = np.array([b])
        t1 = -QZ[-1]/b
        t2 = np.array([z0[-1]])
        return [A1,A2],[t1,t2]

    def neural_network(self,b=None,verbose=True):
        """ Return a (d,n,1) NeuralNetwork representation of f, is possible. This solves the
        TZLP associated to f, and then the algebraic reconstruction problem.

        TODO: allow for different architectures.

        TODO: the t^2 term is wrong when going NN -> f -> NN', but NN = NN' pointwise still ..?
        """
        if self.input_dim > 1:
            QZ,c = self._solve_tzlp(verbose=verbose)
            z0 = np.array([0]*(QZ.shape[0]-1) + [self.constant_term()])
            weights,thresholds = self._solve_algebraic_reconstruction(QZ,c,z0,b=b)
        else:
            raise NotImplementedError("Only implemented for (d,n,1) for d>1")

        return PolynomialNeuralNetwork(weights,thresholds)

    def coef(self,mon):
        """ Retrieve the coefficient of the specified monomial.
        """
        return self.poly.get(tuple(mon),-np.infty)

    def constant_term(self):
        """ Retrieve the constant term of the polynomial.
        """
        mon = (0,)*self.input_dim
        return self.coef(mon)

    def __call__(self,x):
        """
        Evaluate polynomial on x.
        """
        if type(x) != tuple:
            try:
                x = tuple(x)
            except:
                x = (x,)
        assert len(x) == self.input_dim, "Input has improper dimension."
        evals = [c+np.array(m)@x for m,c in self.poly.items()]
        return max(evals)

    def __len__(self):
        return len(self.poly)



class PolynomialNeuralNetwork:
    """
    A fully connected homogenous Neural Network with non-negative weights
    and ReLU activation.
    """
    def __init__(self,weights,thresholds):
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
        
        assert len(weights) == len(thresholds), "Weights and thresholds must be in bijection."
        for A,t in zip(self.weights,self.thresholds):
            assert (A>=0).all(), "All weights must be nonnegative."
            assert len(A) == len(t), "At least one pair (A,t) of weights and thresholds are dimensionally incompatible"

    def component(self,i):
        """ Return a neural network function which is the ith component
        of the current network.
        """
        if self.output_dim == 1:
            return self

        new_weights = []
        new_thresh = []
        c = 0
        for A,t in zip(self.weights,self.thresholds):
            if c == len(self.weights):
                new_weights += [np.copy(A[i])]
                new_thresh += [np.copy(t[i])]
            else:
                new_weights += np.copy(A)
                new_thresh += np.copy(t)
            c += 1
        
        return PolynomialNeuralNetwork(new_weights,new_thresh)

    def tropical(self):
        """ Return the associated tropical polynomial(s) to the network.
        This is only valid for depth <= 2.
        """
        if self.depth <= 2:
            if self.output_dim > 1:
                polys = [self.tropical(self.component(i))[0] for i in range(self.output_dim)]
            else:
                polys = [self._get_poly(self.weights,self.thresholds)]
        else:
            raise NotImplementedError
        return polys

    def _get_poly(self,weights,thresholds):
        """ Get the tropical polynomial of the neural network assuming its output
        dimension is 1. Only implemented for depth 2.
        """
        mons = []
        coeffs = []

        if len(weights) == 1 or len(weights) > 2:
            raise NotImplementedError

        # collect all monomials and their coefficients.
        else:
            A = weights[0]
            B = weights[1][0]
            t = thresholds[0]

            # loop over subsets of len(A)
            for N in range(2**len(A)):
                subset = [B[i]*A[i] for i in range(len(A)) if (N>>i)%2]
                subset_c = [B[i]*t[i] for i in range(len(A)) if not (N>>i)%2]

                # deal with empty set
                if N:
                    mons.append(sum(subset))
                else:
                    mons.append([0]*len(A[0]))
                coeffs.append(sum(subset_c))

        # combine like terms
        poly = {(0,)*len(A[0]):thresholds[-1][0]}
        for m,c in zip(mons,coeffs):
            tup = tuple(m)
            if tup in poly:
                poly[tup] = max(poly[tup],c)
            else:
                poly[tup] = c

        # create the tropical polynomial
        f = TropicalPolynomial(list(poly.keys()),list(poly.values()))

        # manually set the zonotope generators, since we know them a-priori.
        f.Qz = np.array([B[i]*A[i] for i in range(len(A))]).T

        return f

    def __call__(self,x):
        """ Evaluate the Neural Network at x.
        """
        assert len(x) == self.input_dim, "Input has incorrect dimension."
        if type(x) == np.ndarray:
            ret = np.copy(x)
        else:
            ret = np.array(x)

        for A,t in zip(self.weights,self.thresholds):
            ret = np.maximum(A@ret,t)
        return ret


def test_equal(f1,f2,input_dim,n_samples=10000):
    """ Test if two functions are equal pointwise by checking a bunch of 
    random points inside [-500,500]^d.
    """
    for _ in range(n_samples):
        x = 1000*np.random.rand(input_dim)-500
        if abs(f1(x)-f2(x)) > 1e-10:
            return False 
    return True

