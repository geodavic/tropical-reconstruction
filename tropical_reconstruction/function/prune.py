from tropical_reconstruction.polytope import Zonotope
from tropical_reconstruction.function import PolynomialNeuralNetwork
from tropical_reconstruction.optim import ZonotopeTrainer, GradientOptimizer
from tropical_reconstruction.lrchedulers import MultiplictyLRScheler, LRScheduler


def prune_network(N: PolynomialNeuralNetwork, width: int, train_steps: int) -> PolynomialNeuralNetwork:
    """
    Prune a PolynomialNeuralNetwork into a smaller network with one hidden layer.
    """

    f = N.tropical()
    P = f.newton_polytope()

    warmstart = False
    lrscheduler = MultiplicityLRScheduler(lr=0.01)
    opt = GradientOptimizer(lrscheduler=lrscheduler, normalize_grad=True)
    trainer = ZonotopeTrainer(target_polytope=P, optimizer=opt, zonotope_rank=width, warmstart=warmstart)

    Z = ZonotopeTrainer.train(num_steps=train_steps)


