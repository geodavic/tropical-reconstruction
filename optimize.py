from hausdorff import hausdorff_distance
from polytope import Polytope, Zonotope, random_zonotope, random_polytope
from draw import render_polytopes
import numpy as np
import cv2

def approximate_by_zonotope(P,n,steps,seed=None,animate=True,random_start=False,opt_kwargs={}):
    """ Find a zonotope of rank n that approximates P
    in terms of Hausdorff distance.
    """
    if seed is not None:
        np.random.seed(seed)

    if random_start:
        Z = random_zonotope()
    else:
        Z = P.sym()

    opt = ZonotopeOptimizer(Z,P,**opt_kwargs)
    opt.recenter_zonotope()

    if animate:
        frame_size = (500,500)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter('out.mp4',fourcc,10,frame_size) 

    for _ in range(steps):
        dist,p,q = hausdorff_distance(P,Z,full=False)
        try:
            opt.step(p,q)
        except Exception as e:
            if animate:
                out.release()
            print(e)
            return Z
        print(f"step: {_}, distance: {dist}")
        if animate:
            render_polytopes(P,Z,name=f"frame.png") 
            img = cv2.imread("frame.png")
            img = cv2.resize(img,frame_size)
            out.write(img)
    if animate:
        out.release()
    return Z


class ZonotopeOptimizer:
    """ Class that handles optimization of L^2 Hausdorff distance.
    """

    def __init__(self,Z : Zonotope, P: Polytope, stepping_rate=0.01, normalize_grad=False, move_body=True):
        self.Z = Z
        self.P = P
        self.stepping_rate = stepping_rate
        self.normalize_grad = normalize_grad
        self.move_body = move_body
    
    def step(self, target_pt, control_pt):

        # Move Point to Point
        if self.P.has_vertex(target_pt) and self.Z.has_vertex(control_pt):
            control_idx = self.Z.get_pt_idx(control_pt)
            direction = target_pt - control_pt
            self._single_step(control_idx,direction)
            return
        
        # Move Point to subspace
        if self.Z.has_vertex(control_pt):
            control_idx = self.Z.get_pt_idx(control_pt)
            direction = self.get_direction_to_subspace(control_pt,target_pt,self.P)
            self._single_step(control_idx,direction)
            return
        
        # Move Subspace to point
        pts_moving = []
        incidents = self.Z.incident_hyperplanes(control_pt)
        for i in self.Z.hull.vertices:
            v = self.Z.pts[i]
            for plane in incidents:
                if plane.boundary_contains(v):
                    pts_moving.append(i)
        direction = -self.get_direction_to_subspace(target_pt,control_pt,self.Z)

        for pt_idx in pts_moving:
            self._single_step(pt_idx,direction)

    def _single_step(self,control_idx,direction):
       
        v = direction
        if self.normalize_grad:
            v /= np.linalg.norm(v)
        
        v*= self.stepping_rate
        if self.move_body:
            self.Z.translate(v/2)
            self.Z.move_pt(control_idx,v/2)
        else:
            self.Z.move_pt(control_idx,v)

    def recenter_zonotope(self):
        diff = self.P.barycenter - self.Z.barycenter
        self.Z.translate(diff)

    def get_direction_to_subspace(self,x,p,polytope):
        """ Get the direction vector between a point x and the subspace
        spanned by the smallest polytope face containing p.
        """
        incidents = polytope.incident_hyperplanes(p) 
        A = np.array([h.a for h in incidents])
        c = np.array([-h.c for h in incidents])
        direction = np.linalg.lstsq(A,c-A@x)[0]
        return direction

