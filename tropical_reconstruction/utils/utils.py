import numpy as np
from scipy.optimize import linprog
import tqdm
from itertools import combinations
from scipy.spatial import ConvexHull


def all_subsets(n):
    subsets = []
    for i in range(2**n):
        L = list(bin(i)[2:])
        L = [0] * (n - len(L)) + [int(l) for l in L]
        subsets += [L]
    return subsets


def binary_to_subset(L):
    subset = [i for i in range(len(L)) if L[i]]
    return subset


def elements(N, n):
    rval = []
    for i in range(n):
        if 1 & (N >> i):
            rval.append(1)
        else:
            rval.append(0)
    return rval

def print_poly(poly: dict):
    mons = list(poly.keys())
    dim = len(mons[0])
    variables = ["x","y","z"]
    get_var = lambda i : variables[i%3]+(str(i//3+1) if i > 2 else "")

    mons = []
    for m,c in poly.items():
        s = ""
        if c < 0:
            s+= f"({str(c)})"
        else:
            s+= str(c)

        if sum(m):
            s+= "•"
        for i,ex in enumerate(m):
            if ex > 1:
                s += get_var(i)+"^"+str(ex)
            elif ex == 1:
                s += get_var(i)
            elif ex <= 0:
                pass # assuming non negative moonomials
        mons += [s]

    return " + ".join(mons)

def mathematica_print(A):
    """Print 2d array in mathematica format"""
    if type(A) != np.ndarray:
        return str(A)
    else:
        return "{" + ",".join([mathematica_print(a) for a in A]) + "}"


def unit_ball(d, p=1):
    """Return vertices of the unit l^p ball in R^d."""
    if p == 1:
        V = []
        for i in range(d):
            v = np.zeros(d)
            v[i] = 1
            w = np.zeros(d)
            w[i] = -1
            V.append(v)
            V.append(w)
        V = np.array(V)
    else:
        raise NotImplementedError(f"Metric {p} is not supported")

    return V


def generate_LP_example(n, d, z0=1):
    """Generate a random QZ, project it to Qz. Then set U to be the
    set of vertices of Qz (with the lifted points from QZ) and return the tzlp.
    """
    Qz = np.random.rand(d, n)
    last_row = np.random.rand(n)
    QZ = np.append(Qz, [last_row], axis=0)

    all_vertices = np.array([elements(i, n) for i in range(2**n)])

    # create vertices of Qz
    Qz_cubical_vertices = []
    for v in all_vertices:
        Qz_cubical_vertices.append(Qz @ v)
    Qz_cubical_vertices = np.array(Qz_cubical_vertices)

    Qz_hull = ConvexHull(Qz_cubical_vertices)
    Epsilon = [list(v) for v in all_vertices[Qz_hull.vertices] if np.sum(v)]
    U = [list(QZ @ e) for e in Epsilon]

    z0 = [0] * d + [z0]

    return Qz, QZ, U, z0, Epsilon


################### Old Code #######################


def sample_polytope(A, b, N=1, outside=False):
    # sample from polytope of the form Ax <= b
    if N > 1:
        return [sample_polytope(A, b, N=1, outside=outside) for _ in range(N)]
    lp = linprog(c=np.zeros(A.shape[1]), A_ub=A, b_ub=b, bounds=(None, None))
    start = lp.x
    if lp.status == 2:
        if outside:
            return np.random.rand(A.shape[1])
        else:
            raise Exception("Infeasible equations passed")
    for _ in range(5):
        direction = 2 * np.random.rand(3) - 1
        pts = [
            a
            for a in np.linspace(-5, 5, 500)
            if (A @ (start + a * direction) <= b).all()
        ]
        start = start + np.random.choice(pts) * direction

    if outside:
        while True:
            try:
                direction = 2 * np.random.rand(3) - 1
                pts = [
                    a
                    for a in np.linspace(-10, 10, 500)
                    if (A @ (start + a * direction) > b).any()
                ]
                start = start + np.random.choice(pts) * direction
                return start
            except:
                continue

    return start


def generate_witness(QZ, Epsilon, y_namer, w_namer, x_namer, N=1, outside=False):
    # Given a solution x (encoded as the last row of QZ), generate witness values
    # (y,w) for the SLP constraints. If outside=True, this generates non-witness values
    # (i.e. values that should not satisfy the SLP constraints)
    if N > 1:
        return [
            generate_witness(
                QZ, Epsilon, y_namer, w_namer, x_namer, N=1, outside=outside
            )
            for _ in range(N)
        ]
    He = lambda e: np.array([(-1) ** e[j] * QZ[:, j] for j in range(4)] + [[0, 0, -1]])
    out = {x_namer(i): q for i, q in enumerate(QZ[-1])}
    for i, e in enumerate(Epsilon):
        # Ax >= b
        A = He(e)
        b = np.array([0, 0, 0, 0, 1])
        # negate A and b to get it in the form A' x <= b'
        pt = sample_polytope(-A, -b, outside=outside)
        out.update({y: v for y, v in zip(y_namer(i), pt)})
        out.update({w_namer(i): pt[-1]})
    return out


def test_tzlp(N, QZ, Epsilon, tzlp):
    total_errors = 0

    print("Testing feasible points")
    for _ in tqdm.tqdm(range(N)):
        pos = generate_witness(QZ, Epsilon, tzlp.y_name, tzlp.w_name, tzlp.x_name, N=1)
        if not tzlp.evaluator.is_feasible(pos):
            print(
                "Error! The following variable should be feasible but isn't according to the tzlp"
            )
            print(pos)
            print("\n")
            total_errors += 1

    print("Testing infeasible points")
    for _ in tqdm.tqdm(range(N)):
        neg = generate_witness(
            QZ, Epsilon, tzlp.y_name, tzlp.w_name, tzlp.x_name, N=1, outside=True
        )
        if tzlp.evaluator.is_feasible(neg):
            print(
                "Error! The following variable should not be feasible but is according to the tzlp"
            )
            print(neg)
            print("\n")
            total_errors += 1
    print("Total errors: {}".format(total_errors))
    return
