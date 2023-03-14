from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from tropical_reconstruction.polytope import Polytope
from copy import copy, deepcopy
import math
import numpy as np

from tropical_reconstruction.utils import all_subsets


def solve_tzlp(data, verbose=True):
    """Solve a TZLP.

    Parameters
    ----------
    data: tuple
        Tuple of inputs passed to TZLP_Solver. This is the necessary
        data for the solver to produce a solution.
    verbose: bool
        If true, will print steps of TZLP algorithm.
    """
    # Check dimensionality assumption
    Qz = np.array(data[0])
    d, n = Qz.shape
    if n <= d + 1:
        print("Warning: n <= d+1, so assumptions of TZLP are not satisfied")

    # Call solver
    tzlp = TZLP_Solver(*data)
    sol = tzlp.solve(verbose=verbose, numpy=True)

    if sol is not None:
        x, c = sol
        return (np.append(Qz, [x], axis=0), c)
    return None


def solve_algebraic_reconstruction(QZ, c, z0, b=None):
    """Solve the algebraic reconstruction problem given a solution (Q_Z,c,z_0)
    to the geometric reconstruction problem.
    """
    if b is None:
        b = np.ones(QZ.shape[1])

    A1 = np.array([r / b for r in QZ[:-1]]).T
    A2 = np.array([b])
    t1 = -QZ[-1] / b
    t2 = np.array([z0[-1]])
    return [A1, A2], [t1, t2]


class TZLP_Solver:
    """Basic class for a TZLP solver."""

    def __init__(self, Qz=None, U=None, z0=None, Epsilon=None, latex_names=False):
        self.Qz = Qz
        self.U = U
        self.z0 = z0
        self.Epsilon = Epsilon

        self.d = len(Qz)
        self.n = len(Qz[0])
        self.M = len(U)
        self.Y = Polytope(pts=self.U + [self.z0])

        # variable names
        self.latex_names = latex_names
        self.ynames = [
            self.y_name(i=i, l=l) for i in range(self.M) for l in range(self.d)
        ]
        self.wnames = [self.w_name(i) for i in range(self.M)]
        self.xnames = [self.x_name(j) for j in range(self.n)]

        # check that the passed parameters are compatible
        self._check_compatible()

        # get the equations
        self.equations = self._get_equations()

    def y_name(self, i=None, l=None):
        """Name of the y variables"""
        if l is not None:
            assert i < self.M and l < self.d, "Indicies provided out of range"
            if self.latex_names:
                return f"y^{{({i})}}_{{{l}}}"
            else:
                return f"y^{i}_{l}"
        else:
            assert i < self.M, "Indicies provided out of range"
            return [self.y_name(i=i, l=l) for l in range(self.d)]

    def w_name(self, i):
        """Name of the w variables"""
        if self.latex_names:
            return f"w^{{({i})}}"
        else:
            return f"w^{i}"

    def x_name(self, j):
        """Name of the x (primal) variables."""
        if self.latex_names:
            return f"x_{{{j}}}"
        else:
            return f"x_{j}"

    def _check_compatible(self):
        """
        Check that the init parameters are all compatible.

        TODO
        """
        return None

    def _get_equations(self):
        """Get the equations of the TZLP, stored in `LP`, `SLP` and `TLP`
        groups.
        """
        equations = {"LP": [], "SLP": [], "TLP": []}
        variables = self.xnames + ["C"] + self.ynames + self.wnames

        # LP
        for eps, y in zip(self.Epsilon, self.U):
            rhs = y[-1]
            lin_expr = [self.xnames + ["C"], eps + [1]]
            equations["LP"].append(
                Equation(rhs, "E", variables, lin=dict(zip(*lin_expr)))
            )

        # SLP
        qz = np.array(self.Qz)
        Gij = lambda i, j: ((-1) ** (self.Epsilon[i][j]) * qz[:, j]).tolist()
        for i in range(self.M):
            for j in range(self.n):
                lin_expr = [self.y_name(i), Gij(i, j)]
                quad_val = (-1) ** self.Epsilon[i][j]
                equations["SLP"].append(
                    Equation(
                        0,
                        "G",
                        variables,
                        lin=dict(zip(*lin_expr)),
                        quad={(self.w_name(i), self.x_name(j)): quad_val},
                    )
                )

        # TLP
        qz = np.array(self.Qz)
        self.Epsilon_complement = [
            s for s in all_subsets(self.n) if not s in self.Epsilon and sum(s) > 0
        ]

        self.Y_upper_hull_inequalities = self.Y.upper_hull
        self.N = len(self.Y_upper_hull_inequalities)
        P = lambda k, eta: (
            (eta + [1]) * np.array(self.Y_upper_hull_inequalities[k].a[-1])
        ).tolist()

        def r(k, eta):
            d_k = -self.Y_upper_hull_inequalities[k].c
            q_k = np.array(self.Y_upper_hull_inequalities[k].a[:-1])
            return d_k - q_k @ (qz @ np.array(eta))

        for eta in self.Epsilon_complement:
            for k in range(len(self.Y_upper_hull_inequalities)):
                lin_expr = [self.xnames + ["C"], P(k, eta)]
                rhs = r(k, eta)
                equations["TLP"].append(
                    Equation(rhs, "L", variables, lin=dict(zip(*lin_expr)))
                )
        return equations

    def solve_primal(self, verbose=True):
        """Solve for the primal variable x \in \R^n and offset c \in \R. This
        is the solution to ZLP_Z(U).
        """
        if verbose:
            print("Solving for primal variable x...")
        A = np.array(self.Epsilon)
        A = np.append(A, 1 * np.ones((A.shape[0], 1)), axis=1)
        b = [y[-1] for y in self.U]
        sol = np.linalg.lstsq(A, b, rcond=None)

        if abs(sol[1]) <= 1e-10:
            c = sol[0][-1]
            sol = {n: v for n, v in zip(self.xnames, sol[0][:-1])}
            sol["C"] = c
        else:
            if verbose:
                print(f"No solution to ZLP_Z(U) exists (residual: {sol[1]})")
            return None
        return sol

    def check_feasible(self, sol, verbose=True):
        """Check that the TZLP is feasible for a given primal
        value x.
        """

        # Substitute x into SLP equations.
        substituted_equations = copy(self.equations)
        substitute = lambda eq: eq.substitute(sol)
        substituted_equations["SLP"] = list(
            map(substitute, substituted_equations["SLP"])
        )

        if verbose:
            print("Checking that TLP is feasible at x...")

        # Verify there are no quadratics left
        for eq in substituted_equations["SLP"]:
            if eq.quad:
                raise Exception("There are still quadratic terms after substituting")

        # First check TLP equations are feasible:
        tlp_bools = [not eq.is_true_at(sol) for eq in substituted_equations["TLP"]]
        if sum(tlp_bools):
            if verbose:
                perc = 100 * (sum(tlp_bools)) / len(tlp_bools)
                print(f"TLP is not feasible (failed {perc}% of TLP equations)")
            return False

        if verbose:
            print("Checking that SLP is feasible at x...")

        # Now check SLP equations. This requires an LP solver.
        SLP_eqs = np.array(
            [eq.numpy()[0] for eq in substituted_equations["SLP"]], dtype=np.float64
        )
        SLP_rhs = np.array(
            [eq.numpy()[1] for eq in substituted_equations["SLP"]], dtype=np.float64
        )
        lp = linprog(
            c=np.zeros(SLP_eqs.shape[1]),
            A_ub=SLP_eqs,
            b_ub=SLP_rhs,
            bounds=(None, None),
        )
        feasible_pt = lp.x
        if lp.status == 2:
            return False
        elif lp.status in [0, 3]:
            return True
        else:
            raise Exception(lp.status)

    def solve(self, verbose=True, numpy=False):
        """Solve the TZLP by solving the primal (LP) problem
        and then the remaining feasibility problems.
        """
        try:
            sol = self.solve_primal(verbose=verbose)
        except Exception as e:
            if verbose:
                print(e)
            return None
        if self.check_feasible(sol, verbose=verbose):
            if verbose:
                print("Success! Solution to the TZLP:")
            if numpy:
                c = sol.pop("C")
                x = np.array(list(sol.values()))
                sol = (x, c)
            return sol
        else:
            return None


class Equation:
    """Equation class. Consists of linear parts, quadratic parts,
    the right hand side, and sense (>=,<=,=).
    """

    def __init__(self, rhs, sense, variables, lin=None, quad=None):
        self.lin = lin
        self.quad = quad
        self.rhs = rhs
        self.sense = sense
        self.variables = variables
        assert sense in ("G", "L", "E"), f"Unsupported sense {sense}"

    def is_true_at(self, pt):
        """Check if equation is true at a point."""
        linear_val = self._eval_linear(pt, self.lin)
        quad_val = self._eval_quad(pt, self.quad)
        lhs = linear_val + quad_val
        if self.sense == "E":
            return lhs == self.rhs
        if self.sense == "G":
            return lhs >= self.rhs
        if self.sense == "L":
            return lhs <= self.rhs

    def _eval_linear(self, pt, lin):
        """Evaluate a linear equation."""
        if lin is None:
            return 0
        assert set(lin.keys()).issubset(
            set(pt.keys())
        ), "More variables required to evaluate linear term {}".format(lin)
        result = 0
        for var, coef in lin.items():
            result += coef * pt[var]
        return result

    def _eval_quad(self, pt, quad):
        """Evaluate a quadratic equation (with no linear part)."""
        if quad is None:
            return 0
        req_vars = set([v for k in quad for v in k])
        assert set(req_vars).issubset(
            set(pt.keys())
        ), "More vairables required to evaluate quadratic term {}".format(quad)
        result = 0

        for mon, coef in quad.items():
            result += coef * math.prod([pt[v] for v in mon])
        return result

    def substitute(self, pt):
        """Substitute a variable into the equation. Similar to is_true_at(), except
        it doesn't need to substitute all values. Hence the return is another equation,
        not a boolean.
        """
        rhs = deepcopy(self.rhs)
        quad = deepcopy(self.quad)
        lin = deepcopy(self.lin)
        sense = deepcopy(self.sense)

        if self.quad is not None:
            for mon, coef in self.quad.items():
                subs = set(pt.keys()).intersection(set(mon))
                if len(subs) == 2:
                    v1, v2 = subs
                    rhs -= coef * pt[v1] * pt[v2]
                    quad.pop(mon)
                if len(subs) == 1:
                    if len(set(mon)) == 1:
                        rhs -= coef * pt[mon[0]] ** 2
                        quad.pop(mon)
                    else:
                        val = subs.pop()
                        var = mon[0] if mon[0] != val else mon[1]
                        if var in lin:
                            lin[var] += coef * pt[val]
                        else:
                            lin[var] = coef * pt[val]
                        quad.pop(mon)
        if self.lin is not None:
            for var in pt:
                if var in self.lin:
                    rhs -= self.lin[var] * pt[var]
                    lin.pop(var)
        if quad == {}:
            quad = None
        if lin == {}:
            lin = None
        return Equation(rhs, sense, self.variables, lin=lin, quad=quad)

    def items(self):
        """Return items iterator of dictionary representation of the equation."""
        eq = {"lin": self.lin, "quad": self.quad, "sense": self.sense, "rhs": self.rhs}
        return eq.items()

    def numpy(self):
        """Convert equation to numpy array (only if there is no quadratic part).
        Returns it in format lhs <= rhs
        """
        assert self.quad is None, "Cannot cast a quadratic equation to a numpy array"

        array = np.zeros(len(self.variables))
        for i, var in enumerate(self.variables):
            if var in self.lin:
                array[i] = self.lin[var]
        if self.sense == "G":
            return -array, -self.rhs
        else:
            return array, self.rhs

    def _format_coef(self, c, float_precision):
        """Format a coefficient (used in self.texify())"""
        if c == 1:
            return ""
        elif int(c) == c:
            return str(c)
        else:
            return str(np.round(c, decimals=float_precision))

    def texify(self, float_precision=4, align=True, use_sense=True):
        """Render equation into LaTeX format."""
        fstr = lambda f: str(np.round(f, decimals=float_precision))
        align_str = ""
        if align:
            align_str = "&"

        s = ""
        if self.lin is not None:
            for v, c in self.lin.items():
                if c < 0:
                    s += " - "
                if c > 0:
                    s += " + "
                if c != 0:
                    s += self._format_coef(abs(c), float_precision=float_precision) + v

        if self.quad is not None:
            for m, c in self.quad.items():
                if c < 0:
                    s += " - "
                if c > 0:
                    s += " + "
                if c != 0:
                    s += self._format_coef(
                        abs(c), float_precision=float_precision
                    ) + "".join(m)

        if use_sense:
            if self.sense == "E":
                s += " {}= ".format(align_str)
            if self.sense == "G":
                s += " {}\geq ".format(align_str)
            if self.sense == "L":
                s += " {}\leq ".format(align_str)

            s += fstr(self.rhs)

        s = s.strip()
        if s[0] == "+":
            s = s[1:].strip()

        return s
