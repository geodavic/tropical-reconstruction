from tropical_reconstruction.optim import approximate_by_zonotope, GradientOptimizer
from tropical_reconstruction.optim.lrschedulers import MultiplicityLRScheduler
from tropical_reconstruction.polytope import Polytope, Zonotope, random_polytope

P = random_polytope(10,2)
lrscheduler = MultiplicityLRScheduler(start=0.005,scale=1)
opt_kwargs = {"normalize_grad":True, "lrscheduler":lrscheduler}
Z = approximate_by_zonotope(P, GradientOptimizer, 600, 5, opt_kwargs=opt_kwargs, closeness_thresh=0.96)
