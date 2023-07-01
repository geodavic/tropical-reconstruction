from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope, remove_duplicate_generators
from tropical_reconstruction.utils import all_subsets
from scipy.optimize import linprog
import numpy as np
import random

def approximate_polytope(polytope: Polytope, rank: int, outer=False):
    """ Return a zonotope Z of given rank that contains polytope that is approximately minimal

    Proceeds by picking the smalest directions between vertices of the polytope and then
    calculating the minimal oriented enclosing zonotope along those directions.
    """
    """
    dirs = []
    for i,p1 in enumerate(polytope.vertices):
        for j,p2 in enumerate(polytope.vertices):
            if j > i:
                dirs += [p1-p2]

    dirs = sorted(dirs, key=lambda x: np.linalg.norm(x), reverse=True)
    directions = random.sample(dirs,rank)
    #directions = dirs[:rank]
    """
    directions = random_zonotope(rank, polytope.dim, positive=False).generators
    Z = minimal_oriented_enclosing_zonotope(directions, polytope)

    if not outer:
        center = Z.barycenter
        Z=Z.translate(-center)
        scale = polytope.radius/Z.radius
        Z = scale*Z
        Z=Z.translate(center)
    
    return Z

def minimal_oriented_enclosing_zonotope(directions, polytope: Polytope):
    """ Compute the discrete oriented enclosing zonotope (Guibas et. al)

    No element of `directions` should be a scalar multiple of another.
    """
    
    #Preprocess adding negatives and normalizing
    directions_normalized = []
    for D in directions:
        directions_normalized += [D/np.linalg.norm(D)]
        directions_normalized += [-D/np.linalg.norm(D)]

    n = len(directions_normalized)
    d = polytope.dim
    N = len(polytope.vertices)
   
    objective = np.zeros(n+d+n*N)
    for i in range(n):
        objective[i] = 1

    Aub = []
    for i in range(n):
        for j in range(N):
            z = np.zeros(n+d+n*N)
            z[n+d+N*i + j] = -1
            Aub += [z]
            z = np.zeros(n+d+n*N)
            z[n+d+N*i + j] = 1
            z[i] = -1
            Aub += [z]

    Aub = np.array(Aub)
    bub = np.zeros(2*n*N)

    Aeq = []
    beq = []
    for j in range(N):
        for l in range(d):
            beq += [polytope.vertices[j][l]]
            z = np.zeros(n+d+n*N)
            z[n+l] = 1
            for i in range(n):
                z[n+d+N*i + j] = directions_normalized[i][l]
            Aeq += [z]
    Aeq = np.array(Aeq)
    beq = np.array(beq)

    lp = linprog(c=objective, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=None, method="highs-ipm")

    sol = lp.x
    generators = np.array([sol[i]*directions_normalized[i] for i in range(n)])
    mu = np.array(sol[n:n+d])
    Z = Zonotope(generators=generators,mu=mu)

    return remove_duplicate_generators(Z) 

