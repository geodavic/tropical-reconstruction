from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope, approximate_polytope
from tropical_reconstruction.metrics import hausdorff_distance, coarse_hausdorff_distance
from tropical_reconstruction.optim import GradientOptimizer, ZonotopeOptimizer
from tropical_reconstruction.optim.lrschedulers import MultiplicityLRScheduler
from tropical_reconstruction.utils.draw import render_polytopes, render_polytopes_close_ties
import numpy as np
import matplotlib.pyplot as plt
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
            Z = random_zonotope(rank, self.target_polytope.dim, scale=0.5) # 5
            Z = Z.translate(self.target_polytope.barycenter - Z.barycenter)

        return Z

    def _single_train_step(self, target_pt, control_pt, multiplicity, step_number, total_steps):
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
        save_losses: bool = False,
        save_ratios: bool = False,
        render_last: bool = False
    ):
        if stop_thresh is None:
            stop_thresh = 0
        if multiplicity_thresh is None:
            multiplicity_thresh = 1

        self.losses = []
        ratios = []
        pbar = tqdm(range(num_steps))
        for step in pbar:
            distance, target_pt, control_pt, multiplicity = hausdorff_distance(
                self.target_polytope,
                self.zonotope,
                full=False,
                metric=self.metric,
                thresh=multiplicity_thresh,
            )

            self.losses += [distance]
            if save_ratios:
                ratios += [coarse_hausdorff_distance(self.target_polytope,self.zonotope)/distance]

            if distance < stop_thresh:
                return self.zonotope

            type_,Z = self._single_train_step(target_pt, control_pt, multiplicity, step, num_steps)
            pbar.set_description(f"d = {distance:10.10f}, mult = {multiplicity}, lr = {self.optimizer.lrscheduler.lr:6.6f}, type = {type_}")

            if Z is not None:
                self.zonotope = Z
            else:
                return self.zonotope
        if save_losses:
            np.save("losses.npy", np.array(self.losses))
        if save_ratios:
            np.save("ratios.npy", np.array(ratios))
        if render_last:
            render_polytopes_close_ties(self.target_polytope, self.zonotope, name=f"out.png")
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
        out = cv2.VideoWriter(self.filename, fourcc, 30, self.frame_size)
        return out

    def _single_train_step(self, target_pt, control_pt, multiplicity, step_number, total_steps):
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
            self._plot_losses(".losses.png",total_steps)
            img = cv2.imread(".tmp.png")
            img = cv2.resize(img, self.frame_size)

            loss_size_factor = 3
            loss_size = (int(self.frame_size[0]/loss_size_factor), int(self.frame_size[1]/loss_size_factor))
            img2 = cv2.imread(".losses.png")
            img2 = cv2.resize(img2,loss_size)
            y_offset = x_offset = int((1-1/loss_size_factor)*self.frame_size[0])
            img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
            self.out.write(img)
        else:
            pass

        return type_, Z

    def _plot_losses(self, output_file, total_steps):
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(self.losses, color='r')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0,total_steps)
        ax.set_ylim(0,max(self.losses))
        plt.savefig(output_file)

    def train(
        self,
        **kwargs,
    ):
        Z = super().train(**kwargs)
        self.out.release()
        return Z
