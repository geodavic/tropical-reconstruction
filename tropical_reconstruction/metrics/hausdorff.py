from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from qpsolvers import solve_qp
import numpy as np
import scipy as sp
from tropical_reconstruction.utils import unit_ball


def distance_to_polytope(x, P, metric=2):
    """Compute the L1 Hausdorff distance between x and P,
    where P is a polytope. Returns the distance as well as
    the point p in P achieving that distance.

    If metric == 2, then use quadratic solver.
    If metric == 1 or infty, use an LP.
    """
    x = np.array(x, np.float64)
    P_vertices = P.vertices

    if metric == 2:
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
            eps_abs=1e-5,
            eps_rel=1e-5,
        )
        sol = sol[: len(x)]
        dist = np.linalg.norm(sol - x)
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


def hausdorff_distance(P, Q, metric=2, full=True):
    """Compute the Hausdorff distance between polytopes P
    and Q. This is max(min(d(x,P)),min(d(y,Q))).
    """

    distP = -np.inf
    qP = None
    qQ = None
    for p in P.vertices:
        dist, _ = distance_to_polytope(p, Q, metric=metric)
        if dist > distP:
            distP = dist
            qP = p
            qQ = _

    distQ = -np.inf
    pQ = None
    pP = None
    for q in Q.vertices:
        dist, _ = distance_to_polytope(q, P, metric=metric)
        if dist > distQ:
            distQ = dist
            pQ = q
            pP = _

    if full:
        return ((distP, qP, qQ), (distQ, pP, pQ))
    else:
        if distP > distQ:
            return (distP, qP, qQ)
        else:
            return (distQ, pP, pQ)


def hausdorff_distance_close(P, Q, thresh, metric=2):
    """Return all pairs (p,q) \in PxQ such that at least one of p or q is a vertex
    and d(p,q) >= thresh*hausdorff_distance(P,Q)
    """

    dist, _, _ = hausdorff_distance(P, Q, full=False)

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
