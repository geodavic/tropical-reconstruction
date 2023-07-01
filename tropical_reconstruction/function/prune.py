from tropical_reconstruction.polytope import Zonotope
from tropical_reconstruction.function import NeuralNetwork, PolynomialNeuralNetwork
from tropical_reconstruction.optim.trainer import create_zonotope_gradient_trainer
from tropical_reconstruction.tzlp import solve_algebraic_reconstruction
import numpy as np


def prune_network(N: NeuralNetwork, width: int, train_steps: int):
    """
    Prune a PolynomialNeuralNetwork into a smaller network with one hidden layer.
    """

    h = N.tropical()[0]
    P1 = h.numerator.lifted_newton_polytope
    P2 = h.denominator.lifted_newton_polytope

    warmstart = False
    lr = 0.1
    normalize_grad = True
    seed1 = 1
    seed2 = 5
    multiplicity_thresh=0.98
    stop_thresh1 = 3.1
    stop_thresh2 = None


    trainer1 = create_zonotope_gradient_trainer(P1,lr,width,warmstart,seed1,normalize_grad)
    trainer2 = create_zonotope_gradient_trainer(P2,lr,width,warmstart,seed2,normalize_grad)

    Z1 = trainer1.train(num_steps=train_steps,multiplicity_thresh=multiplicity_thresh,stop_thresh=stop_thresh1)
    Z2 = trainer2.train(num_steps=train_steps,multiplicity_thresh=multiplicity_thresh,stop_thresh=stop_thresh2)
    
    z0 = np.array( [0]*(Z1.dim-1) + [-np.infty])
    weights1, thresh1 = solve_algebraic_reconstruction(Z1.generators.T, None, z0)
    weights2, thresh2 = solve_algebraic_reconstruction(Z2.generators.T, None, z0)

    nu1 = NeuralNetwork(weights1, thresh1)
    nu2 = NeuralNetwork(weights2, thresh2)
    return nu1,nu2
