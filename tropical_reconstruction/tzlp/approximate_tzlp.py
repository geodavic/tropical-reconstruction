from tropical_reconstruction.optim.gradient import ZonotopeBoundaryPointGradient
from tropical_reconstruction.polytope import Zonotope
from tropical_reconstruction.metrics import distance_to_polytope_l2
import numpy as np
from tqdm import tqdm


class ApproximateTZLP_Solver:
    """
    Solve a TZLP approximately using gradient descent.
    """

    def __init__(self, optimizer: ZonotopeOptimizer, Qz=None, U=None, z0=None):
        self.Qz = Qz
        self.U = U
        self.z0 = z0

        self.d = len(Qz)
        self.n = len(Qz[0])
        self.M = len(U)
        self.Y = Polytope(pts=self.U + [self.z0])
        
        self.optimizer = optimizer

    def _initialize_solution(self):
        return np.random.rand(self.n)

    def solve(self, steps: int, stop_thresh: float = None):
        start = self._initialize_solution()
        QZ = np.append(self.Qz, [start], axis=0)
        lifted_zonotope = Zonotope(generators=QZ)

        if stop_thresh is None:
            stop_thresh = 0

        pbar = tqdm(range(num_steps))
        for step in pbar:
             dist, type_, lifted_zonotope = self._single_step(lifted_zonotope)

             if dist < stop_thresh:
                return lifted_zonotope

             pbar.set_description(f"d = {dist}, lr = {self.optimizer.lrscheduler.lr}, type = {type_}")

        return lifted_zonotope

    def _single_step(self, lifted_zonotope):
        """
        Perform single gradient step.
        """
        return self.optimizer.step(self.Y, lifted_zonotope)
