from tropical_reconstruction.metrics import hausdorff_distance, coarse_hausdorff_distance
from tropical_reconstruction.polytope import random_zonotope, random_polytope

rank = 5
scale = 0.1
dim = 3
num = 1000

P = random_polytope(10,dim)
print(P.vertices)

ratios = []
for _ in range(num):
    Z = random_zonotope(rank,dim,random_anchor=True,scale=scale)
    d = hausdorff_distance(P,Z,full=False)[0]
    dc = coarse_hausdorff_distance(P,Z)
    ratios += [dc/d]
