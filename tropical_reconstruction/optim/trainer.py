from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope, approximate_polytope
from tropical_reconstruction.metrics import hausdorff_distance
from tropical_reconstruction.optim import GradientOptimizer, ZonotopeOptimizer
from tropical_reconstruction.optim.lrschedulers import MultiplicityLRScheduler
from tropical_reconstruction.utils.draw import render_polytopes, render_polytopes_close_ties
import numpy as np
import cv2
from tqdm import tqdm


def create_zonotope_gradient_trainer(
    P: Polytope,
    lr: float,
    rank: int,
    warmstart: bool,
    seed: int,
    normalize_grad: bool,
    video_out: bool = False,
    render_start: int = 0
):
    lrscheduler = MultiplicityLRScheduler(start=lr)
    opt = GradientOptimizer(lrscheduler, normalize_grad=normalize_grad, use_facet_gradient=True)

    if video_out:
        trainer = ZonotopeTrainerWithVideo(P, opt, zonotope_rank=rank, warmstart=warmstart, seed=seed, render_start=render_start)
    else:
        trainer = ZonotopeTrainer(P, opt, zonotope_rank=rank, warmstart=warmstart, seed=seed)

    return trainer


class ZonotopeTrainer:
    def __init__(
        self,
        target_polytope: Polytope,
        optimizer: ZonotopeOptimizer,
        zonotope_rank: int = None,
        start_zonotope: Zonotope = None,
        seed: int = None,
        warmstart: bool = True,
        **kwargs,
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
            Z = approximate_polytope(self.target_polytope, rank, outer=False)
        else:
            # random zonotope
            Z = random_zonotope(rank, self.target_polytope.dim)
            Z = Z.translate(self.target_polytope.barycenter - Z.barycenter)

        return Z

    def _single_train_step(self, target_pt, control_pt, multiplicity, step_number):
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
        save_losses: str = None
    ):
        if stop_thresh is None:
            stop_thresh = 0
        if multiplicity_thresh is None:
            multiplicity_thresh = 1

        losses = []
        pbar = tqdm(range(num_steps))
        for step in pbar:
            distance, target_pt, control_pt, multiplicity = hausdorff_distance(
                self.target_polytope,
                self.zonotope,
                full=False,
                metric=self.metric,
                thresh=multiplicity_thresh,
            )

            losses += [distance]
            if distance < stop_thresh:
                return self.zonotope

            type_,Z = self._single_train_step(target_pt, control_pt, multiplicity, step)
            pbar.set_description(f"d = {distance:10.10f}, mult = {multiplicity}, lr = {self.optimizer.lrscheduler.lr:6.6f}, type = {type_}")

            if Z is not None:
                self.zonotope = Z
            else:
                return self.zonotope
        if save_losses is not None:
            np.save("losses.npy", np.array(losses))
        return self.zonotope


class ZonotopeTrainerWithVideo(ZonotopeTrainer):

    def __init__(
        self,
        target_polytope: Polytope,
        optimizer: ZonotopeOptimizer,
        zonotope_rank: int = None,
        start_zonotope: Zonotope = None,
        seed: int = None,
        warmstart: bool = True,
        filename: str = None,
        render_start: int = 0
    ):

        super().__init__(
            target_polytope=target_polytope,
            optimizer=optimizer,
            zonotope_rank=zonotope_rank,
            start_zonotope=start_zonotope,
            seed=seed,
            warmstart=warmstart
        )
        self.filename = filename or "out.mp4"
        self.out = self._setup_video()
        self.render_start = render_start

    def _setup_video(self):
        self.frame_size = (1000, 1000)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.filename, fourcc, 10, self.frame_size)
        return out

    def _single_train_step(self, target_pt, control_pt, multiplicity, step_number):
        type_, Z = self.optimizer.step(
            target_pt,
            control_pt,
            self.target_polytope,
            self.zonotope,
            multiplicity=multiplicity,
        )
        if Z is None:
            self.out.release() 
        elif step_number >= self.render_start:
            render_polytopes_close_ties(self.target_polytope, self.zonotope, name=f".tmp.png")
            img = cv2.imread(".tmp.png")
            img = cv2.resize(img, self.frame_size)
            self.out.write(img)
        else:
            pass

        return type_, Z

    def train(
        self,
        **kwargs,
    ):
        Z = super().train(**kwargs)
        render_polytopes_close_ties(self.target_polytope, self.zonotope, name=f"out.png")
        self.out.release()
        return Z
