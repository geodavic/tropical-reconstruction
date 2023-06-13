from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from qpsolvers import solve_qp
import numpy as np
import scipy as sp
from tropical_reconstruction.utils import unit_ball

def distance_to_polytope_l2(x, P):
    """
    Find the distance to from x to P in L^2 norm.
    
    Returns the distance, the point p in P closest to x,
    and the coefficients of the sum representing p
    as a convex sum of vertices of P.
    """
    x = np.array(x, np.float64)
    P_vertices = P.vertices

    Q = sp.linalg.block_diag(
        np.eye(len(x)), np.zeros((len(P_vertices), len(P_vertices)))
    )
    q = -np.concatenate([x, np.zeros(len(P_vertices))])
    A = np.concatenate([np.eye(len(x)), -P_vertices.T], axis=1)
    A = np.concatenate([A, [[0] * len(x) + [1] * len(P_vertices)]], axis=0)
    b = np.array([0] * len(x) + [1], np.float64)
    lb = np.array([-np.inf] * len(x) + [0] * len(P_vertices), np.float64)
    ub = np.array([np.inf] * (len(x) + len(P_vertices)), np.float64)
    sol = solve_qp(
        Q,
        q,
        A=A,
        b=b,
        lb=lb,
        ub=ub,
        solver="osqp",
        polish=1,
        eps_abs=1e-11,
        eps_rel=1e-11,  # TODO: should this be THRESH?
        max_iter=4000000,
    )
    p = sol[: len(x)]
    coeffs = sol[len(x) :]
    dist = np.linalg.norm(p - x)
    return dist, p, coeffs

def distance_to_polytope(x, P, metric=2):
    """Compute the L1 Hausdorff distance between x and P,
    where P is a polytope. Returns the distance as well as
    the point p in P achieving that distance.

    If metric == 2, then use quadratic solver.
    If metric == 1 or infty, use an LP.

    TODO: use py-hausdorff?
    """
    x = np.array(x, np.float64)
    P_vertices = P.vertices

    if metric == 2:
        dist, sol, _ = distance_to_polytope_l2(x,P)
        return dist, sol

    else:
        B_vertices = unit_ball(len(x), p=metric)
        lambda_coefs = np.concatenate(
            [P_vertices.T, [np.ones(len(P_vertices))], [np.zeros(len(P_vertices))]],
            axis=0,
        )
        mu_coefs = np.concatenate(
            [B_vertices.T, [np.zeros(len(B_vertices))], [np.ones(len(B_vertices))]],
            axis=0,
        )

        rho_coefs = np.zeros(len(x) + 2)
        rho_coefs[-1] = -1
        rho_coefs = np.expand_dims(rho_coefs, axis=0).T

        Aeq = np.concatenate([lambda_coefs, mu_coefs, rho_coefs], axis=1)
        beq = np.concatenate([x, [1, 0]])
        c = np.zeros(len(Aeq[0]))
        c[-1] = 1

        prog = linprog(c, A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq)
        if prog.status != 0:
            raise Exception(
                "LP to determine Hausdorff distance is infeasible or unbounded"
            )

        dist = prog.x[-1]
        sol = np.sum(
            [prog.x[i] * P_vertices[i] for i in range(len(P_vertices))], axis=0
        )
        return dist, sol


def hausdorff_distance(P, Q, metric=2, full=True, thresh=1.0):
    """Compute the Hausdorff distance between polytopes P
    and Q. This is max(min(d(x,P)),min(d(y,Q))).

    Thresh determines the multiplicity return value (i.e. number
    of pairs of points achieving thresh*hausdorff distance)
    """

    distances = []

    distP = -np.inf
    qP = None
    qQ = None
    for p in P.vertices:
        dist, _ = distance_to_polytope(p, Q, metric=metric)
        distances += [dist]
        if dist > distP:
            distP = dist
            qP = p
            qQ = _

    distQ = -np.inf
    pQ = None
    pP = None
    for q in Q.vertices:
        dist, _ = distance_to_polytope(q, P, metric=metric)
        distances += [dist]
        if dist > distQ:
            distQ = dist
            pQ = q
            pP = _

    # calculate multiplicity
    mult = 0
    distances = reversed(sorted(distances))
    for d in distances:
        if thresh * max(distP, distQ) > d:
            break
        mult += 1

    if full:
        return ((distP, qP, qQ), (distQ, pP, pQ))
    else:
        if distP > distQ:
            return (distP, qP, qQ, mult)
        else:
            return (distQ, pP, pQ, mult)


def hausdorff_distance_close(P, Q, thresh, metric=2):
    """Return all pairs (p,q) \in PxQ such that at least one of p or q is a vertex
    and d(p,q) >= thresh*hausdorff_distance(P,Q)
    """

    dist, _, _, _ = hausdorff_distance(P, Q, full=False)

    pairs = []
    for p in P.vertices:
        distp, q = distance_to_polytope(p, Q, metric=metric)
        if distp >= thresh * dist:
            pairs += [(p, q)]

    for q in Q.vertices:
        distq, p = distance_to_polytope(q, P, metric=metric)
        if distq >= thresh * dist:
            pairs += [(p, q)]

    return pairs
