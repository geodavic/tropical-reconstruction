from tropical_reconstruction.polytope import Zonotope, Halfspace, TOLERANCE
from tropical_reconstruction.optim import ZonotopeFacetGradient
from tropical_reconstruction.metrics.hausdorff import distance_to_polytope
import numpy as np
import unittest

class test_ZonotopeFacetGradient(unittest.TestCase):
    def setUp(self):
        self.G1 = self._setup1()
        self.G2 = self._setup2()

    def _setup1(self):
        """
        Unit cube in dimension 3 viewed as a zonotope
        """
        generators = np.eye(3,3)
        Z = Zonotope(generators=generators)
        hyperplane = Halfspace(a=np.array([1,0,0]),c=-1)
        return ZonotopeFacetGradient(Z,hyperplane)

    def _setup2(self):
        """
        Random rank 4 zonotope in 3 dimensions. Return 
        hyperplane 
        """
        generators = np.random.rand(4,3)
        mu = np.random.rand(3)
        Z = Zonotope(generators=generators,mu=mu)
    
        # Pick arbitrary facet
        hyperplane = Z.incident_hyperplanes(Z.vertices[-1])[0]

        return ZonotopeFacetGradient(Z,hyperplane)

    def test_symbolic_normal_vector(self):
        """ 
        Test that the ZonotopeFacetGradient normal vector is correct
        """

        nu = np.array(self.G1._facet_normal(evaluate=True))
        target_nu = self.G1.hyperplane.a / np.linalg.norm(self.G1.hyperplane.a)

        assert np.linalg.norm(nu - target_nu) < TOLERANCE

        nu = np.array(self.G2._facet_normal(evaluate=True))
        target_nu = self.G2.hyperplane.a / np.linalg.norm(self.G2.hyperplane.a)

        assert np.linalg.norm(nu - target_nu) < TOLERANCE


    def test_symbolic_offset(self):
        """
        Test that the ZonotopeFacetGradient offset value is correct
        """
        c = self.G1._offset(evaluate=True)
        target_c = -self.G1.hyperplane.c # sign convention is opposite for Hyperplane class

        assert np.abs(c-target_c) < TOLERANCE

        c = self.G2._offset(evaluate=True)
        target_c = -self.G2.hyperplane.c # sign convention is opposite for Hyperplane class

        assert np.abs(c-target_c) < TOLERANCE
