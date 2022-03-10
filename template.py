""" This is used to render the LaTeX code for the TZLP Example in Marco.tex.
"""

import jinja2
import os
import numpy as np
from tzlp import TZLP_Solver, Equation

example_dir = "../../marco/example/"
latex_jinja_env = jinja2.Environment(
    block_start_string="\BLOCK{",
    block_end_string="}",
    variable_start_string="\VAR{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(os.path.abspath(example_dir + "template")),
)

TEMPLATE_FILES = list(os.listdir(example_dir + "template"))


def render_template(file, kwargs):
    """Render jinja template to a file."""
    template = latex_jinja_env.get_template(file)
    out_str = template.render(**kwargs)
    with open(
        os.path.abspath(example_dir + file.replace(".template", ".tex")), "w"
    ) as fp:
        fp.write(out_str)

class TeX_Writer:
    """Class to handle writing of LaTeX templates for the 241 example."""

    def __init__(self, tzlp):
        self.tzlp = tzlp
        self.equations = tzlp.equations
        self.index_example = 2
        self.eps_example_idx = len(tzlp.Epsilon_complement) - 2

    def render_equations(self):
        lp_equations = [eq.texify() for eq in self.equations["LP"]]
        slp_equations = [eq.texify() for eq in self.equations["SLP"]]
        tlp_equations = [eq.texify() for eq in self.equations["TLP"]]
        kwargs = {
            "lp_equations": lp_equations,
            "slp_equations": slp_equations,
            "tlp_equations": tlp_equations,
        }
        render_template("241_equations.template", kwargs)

    def render_setup(self):
        kwargs = {"U": [tuple(z) for z in self.tzlp.U], "z0": tuple(self.tzlp.z0)}
        render_template("241_setup.template", kwargs)

    def render_qz(self):
        kwargs = {"qz": self.tzlp.Qz, "n": self.tzlp.n}
        render_template("qz.template", kwargs)

    def render_eps(self):
        i = self.index_example
        kwargs = {
            "m2": tuple(self.tzlp.U[i][:2]),
            "eps": tuple(self.tzlp.Epsilon[i]),
            "c2": self.tzlp.U[i][-1],
            "i": i,
        }
        render_template("i=2_eps.template", kwargs)

    def render_zlp_eq(self):
        i = self.index_example
        kwargs = {"eq_str": self.equations["LP"][i].texify(align=False)}
        render_template("i=2_zlp_equation.template", kwargs)

    def render_slp_eq(self):
        i = self.index_example
        n = self.tzlp.n
        qz = np.array(self.tzlp.Qz)
        Gij = lambda i, j: ((-1) ** (self.tzlp.Epsilon[i][j]) * qz[:, j]).tolist()
        eqs = [
            self.equations["SLP"][i * n + j] for j in range(n)
        ]
        quad_eqs = [Equation(0,"E",None,quad=e.quad) for e in eqs]
        kwargs = {
            "i": i,
            "G": [Gij(2, j) for j in range(self.tzlp.n)],
            "ynames": self.tzlp.y_name(i),
            "quad_terms": [eq.texify(align=False, use_sense=False) for eq in quad_eqs],
            "eqs": [eq.texify(align=False, use_sense=False) for eq in eqs],
        }
        render_template("i=2_slp_equation.template", kwargs)

    def render_tlp_eq(self):
        eta_idx = self.eps_example_idx
        N = self.tzlp.N
        eqs = [self.equations["TLP"][eta_idx * N + k] for k in range(N)]
        eta = tuple(self.tzlp.Epsilon_complement[eta_idx])
        kwargs = {"tlp_eqs": [eq.texify() for eq in eqs], "eta": eta, "N": N}
        render_template("tlp_equation.template", kwargs)

    def render_Yhull(self):
        rounding = 3
        triples = [
            (
                np.round(e[:-2], decimals=rounding),
                np.round(e[-2], decimals=rounding),
                np.round(e[-1], decimals=rounding),
            )
            for e in self.tzlp.Y_upper_hull_inequalities
        ]
        kwargs = {"hull_data": enumerate(triples)}
        render_template("i=2_Y_upper_hull_eqs.template", kwargs)

    def render_solution(self):
        alpha = np.round(self.tzlp.solve(numpy=True,verbose=False)[0],decimals=2)
        x = str(tuple(alpha))
        qzbar = self.tzlp.Qz + [alpha]
        kwargs = {"qzbar" : qzbar, "x" : x}
        render_template("solution.template",kwargs)
        render_template("qzbar.template",kwargs)

    def render_all(self):
        self.render_equations()
        self.render_setup()
        self.render_qz()
        self.render_eps()
        self.render_zlp_eq()
        self.render_slp_eq()
        self.render_tlp_eq()
        self.render_Yhull()
        self.render_solution()


if __name__ == "__main__":
    from example import TZLPExampleB
    ex = TZLPExampleB
    tzlp = TZLP_Solver(*ex.data,latex_names=True)
    
    writer = TeX_Writer(tzlp)
    writer.render_all()
