from tropical_reconstruction.optim import approximate_by_zonotope, GradientOptimizer
from tropical_reconstruction.optim.lrschedulers import MultiplicityLRScheduler
from tropical_reconstruction.polytope import (
    Polytope,
    Zonotope,
    random_polytope,
    random_zonotope,
)
import os

if not os.path.exists("vid"):
    os.makedirs("vid")
if not os.path.exists("img"):
    os.makedirs("img")

polytope = True
warmstart = False

if polytope:
    P = random_polytope(10, 2)
else:
    P = Polytope(pts=random_zonotope(8, 2).vertices)
    warmstart = False

lrscheduler = MultiplicityLRScheduler(start=0.01, scale=1)
opt_kwargs = {"normalize_grad": True, "lrscheduler": lrscheduler}
Z = approximate_by_zonotope(
    P,
    GradientOptimizer,
    600,
    5,
    opt_kwargs=opt_kwargs,
    closeness_thresh=0.96,
    warmstart=warmstart,
)
