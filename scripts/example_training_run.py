from tropical_reconstruction.optim.trainer import create_zonotope_gradient_trainer
from tropical_reconstruction.polytope import (
    Polytope,
    random_polytope,
)
import numpy as np


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
    parser.add_argument("--render_start", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--stop_thresh", type=float, default=None)
    parser.add_argument("--save_losses", action="store_true")
    parser.add_argument("--save_ratios", action="store_true")
    parser.add_argument("--render_last", action="store_true")
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

    trainer = create_zonotope_gradient_trainer(P,args.lr,args.rank, args.warmstart, args.z_seed, args.normalize_grad, args.render, args.render_start)
   
    print(args.save_ratios)
    Z = trainer.train(num_steps=args.steps, stop_thresh=args.stop_thresh, multiplicity_thresh=args.multiplicity_thresh, save_losses=args.save_losses, save_ratios=args.save_ratios, render_last=args.render_last)
