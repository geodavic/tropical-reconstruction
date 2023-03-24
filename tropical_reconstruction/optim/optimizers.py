from tropical_reconstruction.metrics import hausdorff_distance
from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope, random_polytope, get_direction_to_subspace
from tropical_reconstruction.optim.gradient import ZonotopePointGradient, ZonotopeFacetGradient
from tropical_reconstruction.optim.lrschedulers import LRScheduler
from tropical_reconstruction.utils.draw import render_polytopes, render_polytopes_close_ties
import numpy as np
import cv2


class ZonotopeOptimizer:
    """Base class for optimization of Hausdorff distance."""

    def __init__(
        self,
        lrscheduler: LRScheduler,
        normalize_grad=False,
    ):
        self.normalize_grad = normalize_grad
        self.lrscheduler = lrscheduler
        self.grads = []
        
    def step(self, target_pt, control_pt) -> (int, Zonotope):
        raise NotImplementedError

    def _prepare_gradient(self, grad):
        v = np.copy(grad)
        if self.normalize_grad:
            v /= np.linalg.norm(v)

        v *= self.lrscheduler.lr
        return v


class GradientOptimizer(ZonotopeOptimizer):
    """Direct gradient descent optimizer"""

    def step(self, target_pt, control_pt, P, Z, multiplicity=1, **kwargs):
        self.lrscheduler.step(multiplicity)

        if P.has_vertex(target_pt) and Z.has_vertex(control_pt):
            type_ = 1
            control_idx = Z.get_pt_idx(control_pt, force=True)
            control_subset = Z.pts_subsets(control_idx, binary=False)
            # Not sure about this part
            normal = control_pt - target_pt
            grad = -ZonotopePointGradient(Z, control_subset, normal)()
        elif Z.has_vertex(control_pt):
            type_ = 2
            control_idx = Z.get_pt_idx(control_pt, force=True)
            control_subset = Z.pts_subsets(control_idx, binary=False)
            normal = get_direction_to_subspace(control_pt, target_pt, P)
            grad = ZonotopePointGradient(Z, control_subset, normal)()
        else:
            type_ = 3
            grads = []
            facets = Z.incident_hyperplanes(control_pt)
            for facet in facets:
                new_grad = ZonotopeFacetGradient(
                    Z, facet
                )(target_pt)
                grads += [new_grad]

            # normal cone hack
            grad = - sum(grads) / len(grads)

        grad = self._prepare_gradient(grad)

        #self.grads.append(grad)
        return type_, self._apply_grad(grad, Z)

    def _apply_grad(self, grad, Z):
        new_generators = Z.generators + grad[:-1]
        new_mu = Z.mu + 1 * grad[-1]
        return Zonotope(generators=new_generators, mu=new_mu)


class SymmetryOptimizer(ZonotopeOptimizer):
    """L^2 optimizer for a 2d zonotope that operates on vertices
    rather than on generators
    """

    def __init__(
        self,
        normalize_grad=False,
        move_body=True,
    ):
        super().__init__(
            LRScheduler(),
            normalize_grad=normalize_grad
        )
        self.move_body = move_body

    def _single_step(self, control_idx, direction, Z):
        v = self._prepare_gradient(direction)
        if self.move_body:
            Z = Z.translate(v / 2)
            Z = Z.move_pt(control_idx, v / 2)
        else:
            Z = Z.move_pt(control_idx, v)

    def step(self, target_pt, control_pt, P, Z, **kwargs):
        self.lrscheduler.step(multiplicity)

        # Move Point to Point
        if P.has_vertex(target_pt) and Z.has_vertex(control_pt):
            control_idx = Z.get_pt_idx(control_pt)
            direction = target_pt - control_pt
            return 1,self._single_step(control_idx, direction)

        # Move Point to subspace
        if Z.has_vertex(control_pt):
            control_idx = Z.get_pt_idx(control_pt)
            direction = get_direction_to_subspace(control_pt, target_pt, P)
            return 2,self._single_step(control_idx, direction)

        # Move Subspace to point
        pts_moving = []
        incidents = Z.incident_hyperplanes(control_pt)
        for i in Z.hull.vertices:
            v = Z.pts[i]
            for plane in incidents:
                if plane.boundary_contains(v):
                    pts_moving.append(i)
        direction = -get_direction_to_subspace(target_pt, control_pt, Z)

        for pt_idx in pts_moving:
            Z = self._single_step(pt_idx, direction, Z)

        return 3,Z
