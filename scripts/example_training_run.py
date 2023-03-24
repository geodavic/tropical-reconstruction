from tropical_reconstruction.optim.trainer import ZonotopeTrainer
from tropical_reconstruction.polytope import (
    Polytope,
    Zonotope,
    random_polytope,
    random_zonotope,
)
from tropical_reconstruction.optim import GradientOptimizer, ZonotopeOptimizer
from tropical_reconstruction.optim.lrschedulers import MultiplicityLRScheduler
import numpy as np


def create_gradient_trainer(
    P: Polytope,
    lr: float,
    rank: int,
    warmstart: bool,
    seed: int,
    normalize_grad: bool,
    video_out=False,
):
    lrscheduler = MultiplicityLRScheduler(start=lr)
    opt = GradientOptimizer(lrscheduler, normalize_grad=normalize_grad)

    if video_out:
        raise NotImplementedError
    else:
        trainer = ZonotopeTrainer(
            P, opt, zonotope_rank=rank, warmstart=warmstart, seed=seed
        )

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--z_seed", type=int, default=None)
    parser.add_argument("--p_seed", type=int, default=None)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--stop_thresh", type=float, default=None)
    parser.add_argument("--multiplicity_thresh", type=float, default=0.96)

    args = parser.parse_args()

    if args.z_seed is None:
        args.z_seed = np.random.randint(2**32)

    if args.p_seed is None:
        args.p_seed = np.random.randint(2**32)

    print(f"Target polytope seed: {args.p_seed}",flush=True)
    print(f"Starting zonotope seed: {args.z_seed}",flush=True)

    np.random.seed(args.p_seed)
    P = random_polytope(10, args.dimension)

    trainer = create_gradient_trainer(P,args.lr,args.rank, args.warmstart, args.z_seed, args.normalize_grad, args.render)

    trainer.train(args.steps, args.stop_thresh, args.multiplicity_thresh)
