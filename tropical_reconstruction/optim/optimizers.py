from tropical_reconstruction.metrics import hausdorff_distance
from tropical_reconstruction.polytope import Polytope, Zonotope, random_zonotope, random_polytope, get_direction_to_subspace, polytope_height_at
from tropical_reconstruction.optim.gradient import ZonotopeVertexGradient, ZonotopeFacetGradient, ZonotopeBoundaryPointGradient
from tropical_reconstruction.optim.lrschedulers import LRScheduler
import numpy as np
import cv2


class ZonotopeOptimizer:
    """Base class for optimization of a Zonotope."""

    def __init__(
        self,
        lrscheduler: LRScheduler,
        normalize_grad=False,
    ):
        self.normalize_grad = normalize_grad
        self.lrscheduler = lrscheduler
        self.grads = []
        
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_gradient(self, grad):
        v = np.copy(grad)
        if self.normalize_grad:
            v /= np.linalg.norm(v)

        v *= self.lrscheduler.lr
        return v

    def _apply_grad(self, grad, Z: Zonotope):
        grad = self._prepare_gradient(grad)
        new_generators = Z.generators + grad[:-1]
        new_mu = Z.mu + 1 * grad[-1]
        return Zonotope(generators=new_generators, mu=new_mu)


class TZLPOptimizer(ZonotopeOptimizer):
    """ Optimizer for solving the approximate TZLP """
    
    def step(self, P, Z):
        Z_dist = 0
        for v in Z.upper_hull_vertices():
            delta = polytope_height_at(P, v[:-1]) - v[-1]
            if abs(delta) > Z_dist:
                Z_dist = abs(delta)
                control_pt_I = np.copy(v)
                target_pt_I = np.copy(v)
                target_pt_I[-1] += delta

        P_dist = 0
        for v in P.upper_hull_vertices():
            delta = polytope_height_at(Z, v[:-1]) - v[-1]
            if abs(delta) > P_dist:
                P_dist = abs(delta)
                target_pt_II = np.copy(v)
                control_pt_II = np.copy(v)
                target_pt_II[-1] += delta

        if Z_dist > P_dist:
            target_pt = target_pt_I
            control_pt = control_pt_I
            dist = Z_dist
            type_=1
        else:
            target_pt = target_pt_II
            control_pt = control_pt_II
            dist = P_dist
            type_=2

        grad = ZonotopeBoundaryPointGradient(Z,control_pt)(target_pt)
        return dist, type_, self._apply_grad(grad)

    def _prepare_gradient(self, grad):
        """ Modify existing method to project gradient onto last component """
        grad[-1] = np.zeros_like(grad[-1]) # do not translate the zonotope
        grad[:,:-1] = np.zeros_like(grad[:,:-1]) # do not move in R^d, only last dimension
        return super()._prepare_gradient(grad)


class GradientOptimizer(ZonotopeOptimizer):
    """Direct gradient descent optimizer for zonotope fitting"""

    def __init__(
        self,
        lrscheduler: LRScheduler,
        normalize_grad=False,
        use_facet_gradient=True
    ):
        super().__init__(lrscheduler, normalize_grad=normalize_grad)
        self.use_facet_gradient = use_facet_gradient

    def step(self, target_pt, control_pt, P, Z, multiplicity=1, **kwargs):
        self.lrscheduler.step(multiplicity=multiplicity)

        if P.has_vertex(target_pt) and Z.has_vertex(control_pt):
            type_ = 1
            control_idx = Z.get_pt_idx(control_pt, force=True)
            control_subset = Z.pts_subsets(control_idx, binary=False)
            # Not sure about this part. SHould normal be normalized?
            normal = control_pt - target_pt
            grad = -ZonotopeVertexGradient(Z, control_subset, normal)()
        elif Z.has_vertex(control_pt):
            type_ = 2
            control_idx = Z.get_pt_idx(control_pt, force=True)
            control_subset = Z.pts_subsets(control_idx, binary=False)
            normal = get_direction_to_subspace(control_pt, target_pt, P)
            # Should normal be normalized?
            grad = ZonotopeVertexGradient(Z, control_subset, normal)()
        else:
            type_ = 3
            grads = []
            if self.use_facet_gradient:
                facets = Z.incident_hyperplanes(control_pt)
                for facet in facets:
                    new_grad = ZonotopeFacetGradient(
                        Z, facet
                    )(target_pt)
                    grads += [new_grad]
            else:
                new_grad = ZonotopeBoundaryPointGradient(Z,control_pt)(target_pt)
                grads += [new_grad]

            # normal cone hack
            grad = - sum(grads) / len(grads)

        #self.grads.append(grad)
        return type_, self._apply_grad(grad, Z)


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
        return Z

    def step(self, target_pt, control_pt, P, Z, **kwargs):
        self.lrscheduler.step(multiplicity)

        # Move Point to Point
        if P.has_vertex(target_pt) and Z.has_vertex(control_pt):
            control_idx = Z.get_pt_idx(control_pt)
            direction = target_pt - control_pt
            return 1,self._single_step(control_idx, direction,Z)

        # Move Point to subspace
        if Z.has_vertex(control_pt):
            control_idx = Z.get_pt_idx(control_pt)
            direction = get_direction_to_subspace(control_pt, target_pt, P)
            return 2,self._single_step(control_idx, direction,Z)

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
