from tropical_reconstruction.examples import RandomNeuralNetwork
from tropical_reconstruction.function import test_equal
import tqdm
import unittest

class test_reconstruction(unittest.TestCase):
    def setUp(self):
        self.num_times = 5
    
    def test_351(self):
        self._test_composition(3,5,self.num_times)

    def test_241(self):
        self._test_composition(2,4,self.num_times)

    def test_361(self):
        self._test_composition(3,6,self.num_times)
        
    def _test_composition(self, d, n, num):
        """Generate a random neural network N, convert it to a
        tropical polynomial f, and the convert f back to a neural
        network N2 (using TZLP framework). Then check N == N2 pointwise
        everywhere. Repeat num times.
        """
        for _ in tqdm.tqdm(range(num)):

            N = RandomNeuralNetwork((d, n, 1)).NN
            f = N.tropical()[0]
            N2 = f.neural_network(verbose=False)
            if N2 is not None:
                passed = test_equal(N, f, d) and test_equal(f, N2, d)
                if not passed:
                    print("Failed equality test")
            else:
                print("Unable to solve for N")
                passed = False

            if not passed:
                break

        assert passed, f"Composition testing failed for networks of architecture ({d},{n},1)"
