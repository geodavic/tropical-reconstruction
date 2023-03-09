from hausdorff import hausdorff_distance
from polytope import Polytope, Zonotope, random_zonotope, random_polytope
from gradient import ZonotopePointGradient, ZonotopeFacetGradient
from draw import render_polytopes, render_polytopes_close_ties
import numpy as np
import cv2

METRIC = 2


def approximate_by_zonotope(
    P, opt_cls, steps, rank, seed=None, startZ=None, animate=True, opt_kwargs={}
):
    """Find a zonotope of rank n that approximates P
    in terms of Hausdorff distance.
    """
    if seed is None:
        seed = np.random.randint(2**32)
    np.random.seed(seed)
    print(seed)

    if startZ is not None:
        Z = startZ
    else:
        Z = random_zonotope(rank, P.dim)
    Z = Z.translate(P.barycenter - Z.barycenter)
    # Z = P.sym()
    # P.translate(Z.barycenter-P.barycenter)
    # Z = (P.radius/Z.radius)*Z
    opt = opt_cls(Z, P, **opt_kwargs)

    if animate:
        frame_size = (1000, 1000)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("vid/out.mp4", fourcc, 10, frame_size)

    for _ in range(steps):
        dist, p, q = hausdorff_distance(opt.P, opt.Z, full=False, metric=METRIC)
        # opt.step(p,q)
        try:
            opt.step(p, q)
            # pass
        except Exception as e:
            if animate:
                out.release()
            opt.print(e)
            return opt.Z
        opt.print(f"step: {_}, distance: {dist}")
        if animate:
            render_polytopes_close_ties(opt.P, opt.Z, name=f"img/frame.png")
            img = cv2.imread("img/frame.png")
            img = cv2.resize(img, frame_size)
            out.write(img)
    if animate:
        out.release()
    return opt.Z


def approximate_by_zonotope_silent(
    P, opt_cls, steps, rank, seed=None, startZ=None, opt_kwargs={}
):
    if seed is None:
        seed = np.random.randint(2**32)
    np.random.seed(seed)

    if startZ is not None:
        Z = startZ
    else:
        Z = random_zonotope(rank, P.dim)
    # Z = Z.translate(P.barycenter-Z.barycenter)
    # Z = P.sym()
    # P.translate(Z.barycenter-P.barycenter)
    # Z = (P.radius/Z.radius)*Z
    opt = opt_cls(Z, P, **opt_kwargs)

    for _ in range(steps):
        dist, p, q = hausdorff_distance(opt.P, opt.Z, full=False, metric=METRIC)
        try:
            opt.step(p, q)
        except Exception as e:
            return _, opt.Z
        opt.print(f"step: {_}, distance: {dist}")
    return _, opt.Z


def get_direction_to_subspace(x, p, polytope):
    """Get the direction vector between a point x and the subspace
    spanned by the smallest polytope face containing p.
    """
    incidents = polytope.incident_hyperplanes(p)
    A = np.array([h.a for h in incidents])
    c = np.array([-h.c for h in incidents])
    direction = np.linalg.lstsq(A, c - A @ x)[0]
    return direction


class ZonotopeOptimizer:
    """Base class for optimization of Hausdorff distance."""

    def __init__(
        self,
        Z: Zonotope,
        P: Polytope,
        stepping_rate=0.01,
        normalize_grad=False,
        verbose=True,
    ):
        self.Z = Z
        self.P = P
        self.stepping_rate = stepping_rate
        self.normalize_grad = normalize_grad
        self.verbose = verbose
        self.grads = []

    def step(self, target_pt, control_pt):
        raise NotImplementedError

    def _prepare_gradient(self, grad):
        v = np.copy(grad)
        if self.normalize_grad:
            v /= np.linalg.norm(v)

        v *= self.stepping_rate
        return v


class GradientOptimizer(ZonotopeOptimizer):
    """Direct gradient descent optimizer"""

    def print(self, s):
        if self.verbose:
            print(s)

    def step(self, target_pt, control_pt):
        if self.P.has_vertex(target_pt) and self.Z.has_vertex(control_pt):
            self.print("type 1")
            control_idx = self.Z.get_pt_idx(control_pt, force=True)
            control_subset = self.Z.pts_subsets(control_idx, binary=False)
            # Not sure about this part
            normal = control_pt - target_pt
            grad = -ZonotopePointGradient(self.Z, control_subset, normal)()
        elif self.Z.has_vertex(control_pt):
            self.print("type 2")
            control_idx = self.Z.get_pt_idx(control_pt, force=True)
            control_subset = self.Z.pts_subsets(control_idx, binary=False)
            normal = get_direction_to_subspace(control_pt, target_pt, self.P)
            grad = ZonotopePointGradient(self.Z, control_subset, normal)()
        else:
            self.print("type 3")
            grads = []
            facets = self.Z.incident_hyperplanes(control_pt)
            for facet in facets:
                for v in self.Z.vertices:
                    if facet.boundary_contains(v):
                        idx = self.Z.get_pt_idx(v, force=True)
                        control_subset = self.Z.pts_subsets(idx, binary=False)
                        # self.print(control_subset)
                        break
                facet_generators_subset = self.Z.get_facet_generators(facet)
                new_grad = ZonotopeFacetGradient(
                    self.Z, control_subset, facet_generators_subset, target_pt
                )()
                grads += [new_grad]

            # normal cone hack
            grad = sum(grads) / len(grads)

        grad = self._prepare_gradient(grad)
        # TODO: decide on a sign for grad. For now below will suffice.
        Zf = self._apply_grad(grad)
        Zb = self._apply_grad(-grad)

        distf, _, _ = hausdorff_distance(self.P, Zf, full=False, metric=METRIC)
        distb, _, _ = hausdorff_distance(self.P, Zb, full=False, metric=METRIC)

        if distf < distb:
            self.grads.append(grad)
            self.Z = Zf
        else:
            self.print("flipped grad")
            self.grads.append(-grad)
            self.Z = Zb

    def _apply_grad(self, grad):
        new_generators = self.Z.generators + grad[:-1]
        new_mu = self.Z.mu + 1 * grad[-1]
        return Zonotope(generators=new_generators, mu=new_mu)


class SymmetryOptimizer(ZonotopeOptimizer):
    """L^2 optimizer for a 2d zonotope that operates on vertices
    rather than on generators
    """

    def __init__(
        self,
        Z: Zonotope,
        P: Polytope,
        stepping_rate=0.01,
        normalize_grad=False,
        move_body=True,
    ):
        assert (
            Z.dim == 2
        ), "The SymmetryOptimizer only works for zonotopes of dimension 2"
        super().__init__(
            Z, P, stepping_rate=stepping_rate, normalize_grad=normalize_grad
        )
        self.move_body = move_body

    def _single_step(self, control_idx, direction):
        v = self._prepare_gradient(direction)
        if self.move_body:
            self.Z = self.Z.translate(v / 2)
            self.Z = self.Z.move_pt(control_idx, v / 2)
        else:
            self.Z = self.Z.move_pt(control_idx, v)

    def step(self, target_pt, control_pt):
        # Move Point to Point
        if self.P.has_vertex(target_pt) and self.Z.has_vertex(control_pt):
            control_idx = self.Z.get_pt_idx(control_pt)
            direction = target_pt - control_pt
            self._single_step(control_idx, direction)
            return

        # Move Point to subspace
        if self.Z.has_vertex(control_pt):
            control_idx = self.Z.get_pt_idx(control_pt)
            direction = get_direction_to_subspace(control_pt, target_pt, self.P)
            self._single_step(control_idx, direction)
            return

        # Move Subspace to point
        pts_moving = []
        incidents = self.Z.incident_hyperplanes(control_pt)
        for i in self.Z.hull.vertices:
            v = self.Z.pts[i]
            for plane in incidents:
                if plane.boundary_contains(v):
                    pts_moving.append(i)
        direction = -get_direction_to_subspace(target_pt, control_pt, self.Z)

        for pt_idx in pts_moving:
            self._single_step(pt_idx, direction)
