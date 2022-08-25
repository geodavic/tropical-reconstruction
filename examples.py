import numpy as np
from tzlp import TZLP_Solver
from utils import test_tzlp, generate_witness, generate_LP_example
from function import TropicalPolynomial, PolynomialNeuralNetwork, test_equal


def clean_float(fl):
    if abs((fl - np.round(fl))) < 1e-6:
        return int(np.round(fl))
    else:
        return fl


class TropicalExampleB:
    """Tropical polynomial:
    f(x,y) = 10 + 12x^4 + 9x^2y + 11x^6y + 14x^2y^4 + 16x^6y^4 + 13x^4y^5 + 15x^8y^5 + (-5)y^6
             + (-3)x^4y^6 + (-6)x^2y^7 + (-4)x^6y^7 + (-1)x^2y^10 + 1x^6y^10 + (-2)x^4y^11 + 0x^8y^11
    """

    monomials = [
        (0, 0),
        (4, 0),
        (2, 1),
        (6, 1),
        (2, 4),
        (6, 4),
        (4, 5),
        (8, 5),
        (0, 6),
        (4, 6),
        (2, 7),
        (6, 7),
        (2, 10),
        (6, 10),
        (4, 11),
        (8, 11),
    ]
    coeffs = [10, 12, 9, 11, 14, 16, 13, 15, -5, -3, -6, -4, -1, 1, -2, 0]
    f = TropicalPolynomial(monomials, coeffs)
    f.Qz = np.array([[4, 2, 2, 0], [0, 1, 4, 6]])


class NeuralNetworkExampleB:
    """A Neural network pointwise equal to TropicalExampleB."""

    A = np.array([[2, 0], [2, 1], [1, 2], [0, 2]])
    B = np.array([[2, 1, 2, 3]])
    t = np.array([-1, 1, -2, 5])
    r = np.array([5])
    NN = PolynomialNeuralNetwork([A, B], [t, r])


class TZLPExampleB:
    """This TZLP data comes from converting TropicalExampleB to a Neural Network."""

    f = TropicalExampleB.f
    Qz, U, z0, Epsilon = f._get_tzlp_data()
    # make entries of U and Qz integer (for cleaner latex rendering)
    Qz = [[clean_float(e) for e in r] for r in Qz]
    U = [[clean_float(e) for e in r] for r in U]
    data = Qz, U, z0, Epsilon


class NeuralNetworkExampleC:
    A = np.array([[2, 0], [2, 1], [1, 2], [0, 2]])
    B = np.array([[1, 1, 1, 1]])
    t = np.array([0.75, 1.5, 0.5, 2.5])
    r = np.array([8])
    NN = PolynomialNeuralNetwork([A, B], [t, r])


class TropicalExampleC:
    f = NeuralNetworkExampleC.NN.tropical()[0]
    f.Qz = NeuralNetworkExampleC.A.T


class TZLPExampleC:
    """This TZLP data comes from converting TropicalExampleC to a Neural Network."""

    f = TropicalExampleC.f
    Qz, U, z0, Epsilon = f._get_tzlp_data()
    # make entries of U and Qz cleaner (for cleaner latex rendering)
    Qz = [[clean_float(e) for e in r] for r in Qz]
    U = [[clean_float(e) for e in r] for r in U]
    data = Qz, U, z0, Epsilon


class NeuralNetworkExampleD:
    """A depth three neural network (2,4,2,1)"""

    A = np.array([[2, 0], [2, 1], [1, 2], [0, 2]])
    B = np.array([[0, 1, 2, 1], [1, 0, 2, 1]])
    C = np.array([[1, 2]])
    t = np.array([0.75, 1.5, 0.5, 2.5])
    r = np.array([3, -1])
    s = np.array([-3])
    NN = PolynomialNeuralNetwork([A, B, C], [t, r, s])


class TropicalExampleD:
    f = NeuralNetworkExampleD.NN.tropical()[0]


class RandomNeuralNetwork:
    """A random depth 2 ReLU Polynomial Neural Network. Weights are chosen uniformly
    inside [0,MAX] and thresholds are chosen uniformly in [-MAX/2,MAX/2].
    """

    def __init__(self, architecture, MAX=None, integer=False):
        self.MAX = MAX or 10 / len(architecture)
        self.integer = integer
        self.set_params(architecture)

    def set_params(self, architecture):
        weights = []
        thresholds = []
        for L, K in zip(architecture, architecture[1:]):
            A = self.MAX * np.random.rand(K, L)
            t = 2 * self.MAX * np.random.rand(K) - self.MAX
            if self.integer:
                A = A.round(decimals=0)
                t = t.round(decimals=0)
            weights.append(A)
            thresholds.append(t)
        self.NN = PolynomialNeuralNetwork(weights, thresholds)


class RandomTZLP:
    """Data for a random TZLP."""

    def __init__(self, n=None, d=None, z0=None):
        self.n = n
        self.d = d
        self.z0 = z0

        params = generate_LP_example(n, d, z0=self.z0)
        self.Qz, self.QZ, self.U, self.z0, self.Epsilon = params
