from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope
from tropical_reconstruction.metrics import hausdorff_distance
from tropical_reconstruction.optim import ZonotopeOptimizer
import numpy as np
from tqdm import tqdm


class ZonotopeTrainer:
    def __init__(
        self,
        target_polytope: Polytope,
        optimizer: ZonotopeOptimizer,
        zonotope_rank: int = None,
        start_zonotope: Zonotope = None,
        seed: int = None,
        warmstart: bool = True,
    ):
        self.metric = 2  # Only support L^2 metric
        self.optimizer = optimizer
        self.warmstart = warmstart
        self.seed = seed or np.random.randint(2**32)
        np.random.seed(self.seed)

        assert (
            zonotope_rank or start_zonotope
        ), "Must specify a starting zonotope or a desired zonotope rank"

        self.target_polytope = target_polytope
        self.zonotope = start_zonotope or self._initialize_zonotope(zonotope_rank)

    def _initialize_zonotope(self, rank: int):
        """
        Create an intial zonotope from which to start optimization.
        """

        if self.warmstart:
            # Smallest zonotope containing the target
            Z = self.target_polytope.sym()
        else:
            # random zonotope
            Z = random_zonotope(rank, self.target_polytope.dim)
            Z = Z.translate(self.target_polytope.barycenter - Z.barycenter)

        return Z

    def _single_train_step(self, target_pt, control_pt, multiplicity):
        return self.optimizer.step(
            target_pt,
            control_pt,
            self.target_polytope,
            self.zonotope,
            multiplicity=multiplicity,
        )

    def train(
        self,
        num_steps: int,
        stop_thresh: float = None,
        multiplicity_thresh: float = None,
    ):
        if stop_thresh is None:
            stop_thresh = 0
        if multiplicity_thresh is None:
            multiplicity_thresh = 1

        pbar = tqdm(range(num_steps))
        for step in pbar:
            distance, target_pt, control_pt, multiplicity = hausdorff_distance(
                self.target_polytope,
                self.zonotope,
                full=False,
                metric=self.metric,
                thresh=multiplicity_thresh,
            )


            if distance < stop_thresh:
                return self.zonotope

            type_,Z = self._single_train_step(target_pt, control_pt, multiplicity)
            pbar.set_description(f"d = {distance}, mult = {multiplicity}, lr = {self.optimizer.lrscheduler.lr}, type = {type_}")

            if Z is not None:
                self.zonotope = Z
            else:
                return self.zonotope

        return self.zonotope
