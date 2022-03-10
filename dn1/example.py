import numpy as np
from tzlp import TZLP_Solver
from utils import test_tzlp, generate_witness, generate_LP_example
from function import TropicalPolynomial, PolynomialNeuralNetwork, test_equal
from template import TeX_Writer
import tqdm

class TropicalExampleB:
    """ Tropical polynomial:
    f(x,y) = 10 + 12x^4 + 9x^2y + 11x^6y + 14x^2y^4 + 16x^6y^4 + 13x^4y^5 + 15x^8y^5 + (-5)y^6 
             + (-3)x^4y^6 + (-6)x^2y^7 + (-4)x^6y^7 + (-1)x^2y^10 + 1x^6y^10 + (-2)x^4y^11 + 0x^8y^11
    """

    monomials = [(0, 0), (4, 0), (2, 1), (6, 1), (2, 4), (6, 4), (4, 5), (8, 5), (0, 6), (4, 6), (2, 7), (6, 7), (2, 10), (6, 10), (4, 11), (8, 11)]
    coeffs = [10, 12, 9, 11, 14, 16, 13, 15, -5, -3, -6, -4, -1, 1, -2, 0]
    f = TropicalPolynomial(monomials,coeffs)
    f.Qz = np.array([[4,2,2,0],[0,1,4,6]])

class NeuralNetworkExampleB:
    """ A Neural network pointwise equal to TropicalExampleB.
    """
    A = np.array([[2,0],[2,1],[1,2],[0,2]])
    B = np.array([[2,1,2,3]])
    t = np.array([-1,1,-2,5])
    r = np.array([5])
    NN = PolynomialNeuralNetwork([A,B],[t,r])


class TZLPExampleB:
    """ This TZLP data comes from converting TropicalExampleB to a Neural Network.
    """
    f = TropicalExampleB.f
    Qz,U,z0,Epsilon = f._get_tzlp_data()
    # make entries of U and Qz integer (for cleaner latex rendering)
    Qz = [[int(e) for e in r] for r in Qz]
    U = [[int(e) for e in r] for r in U]
    data = Qz,U,z0,Epsilon


class RandomNeuralNetwork:
    """ A random depth 2 ReLU Polynomial Neural Network. Weights are chosen uniformly
    inside [0,20] and thresholds are chosen uniformly in [-10,10].
    """
    def __init__(self,d,n,integer=False):
        self.MAX = 5
        if integer:
            self.set_integer_params(d,n)
        else:
            self.set_params(d,n)

    def set_integer_params(self,d,n):
        A = np.array([np.random.randint(1,self.MAX) for _ in range(n*d)]).reshape(n,d)
        B = np.array([[np.random.randint(1,self.MAX) for _ in range(n)]])
        t = np.array([np.random.randint(-self.MAX//2,self.MAX//2) for _ in range(n)])
        r = np.array([np.random.randint(-self.MAX//2,self.MAX//2)])
        self.NN = PolynomialNeuralNetwork([A,B],[t,r])

    def set_params(self,d,n):
        A = self.MAX*np.random.rand(n,d)
        B = self.MAX*np.random.rand(1,n)
        t = 2*self.MAX*np.random.rand(n) - self.MAX
        r = 2*self.MAX*np.random.rand(1) - self.MAX
        self.NN = PolynomialNeuralNetwork([A,B],[t,r])


class RandomTZLP:
    """ Data for a random TZLP.
    """
    def __init__(self,n=None,d=None,z0=None):
        self.n = n
        self.d = d
        self.z0 = z0
        
        params = generate_LP_example(n,d,z0=self.z0)
        self.Qz,self.QZ,self.U,self.z0,self.Epsilon = params


def test_composition(d,n,num):
    for _ in tqdm.tqdm(range(num)):
            
        N = RandomNeuralNetwork(d,n).NN
        f = N.tropical()[0]
        N2 = f.neural_network(verbose=False)
        if N2 is not None:
            passed = test_equal(N,f,d) and test_equal(f,N2,d)
            if not passed:
                print("Failed equality test")
        else:   
            print("Unable to solve for N")
            passed = False
        
        if not passed:
            break

    if not passed:
        print("Problem data:")
        print(N.weights)
        print(N.thresholds)
    else:
        print("All examples passed")


if __name__ == "__main__":
    test_composition(2,4,10)
