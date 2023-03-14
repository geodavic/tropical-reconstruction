import matplotlib.pyplot as plt
from tropical_reconstruction.polytope import Polytope,Zonotope
from tropical_reconstruction.metrics import hausdorff_distance, hausdorff_distance_close

def draw_polytope_lines(P: Polytope, ax, marker='o', color='b',linewidth=1, markersize=3):
    n = len(P.vertices)
    for i in range(n):
        p1 = P.vertices[i]
        p2 = P.vertices[(i+1)%n]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],marker=marker,color=color,linewidth=linewidth,markersize=markersize)

def draw_polytope(P: Polytope, subdivision=None, name=None):
    """ Draw a polytope in 2D
    """
    plt.clf()

    # Set axes to fit P
    plt.gca().set_aspect('equal') 
    maxP,minP = P.bounds()
    factor = 0.6
    width = factor*max(maxP[0]-minP[0],maxP[1]-minP[1])
    plt.xlim(min(minP)-width,max(maxP)+width)
    plt.ylim(min(minP)-width,max(maxP)+width)
   
    # plot subdivision if passed
    if subdivision is not None:
        for F in subdivision:
            FP = Polytope(pts=F)
            draw_polytope_lines(FP,plt,linewidth=0.5,markersize=2,color='k')

    # Plot vertices
    draw_polytope_lines(P,plt)

    # Save
    if name is None:
        name = "render.png"
    plt.savefig(name,bbox_inches='tight')


def render_polytopes(P: Polytope,Z: Zonotope, metric=2,name=None):
    """ Draw P and Z as well as a line for their Hausdorff
    distance.
    """
    plt.clf()
    assert P.dim == 2 and Z.dim == 2, "Can only render 2-dimensional polytopes"

    # Set axes to fit P
    plt.gca().set_aspect('equal') 
    maxP,minP = P.bounds()
    factor = 0.6
    width = factor*max(maxP[0]-minP[0],maxP[1]-minP[1])
    plt.xlim(min(minP)-width,max(maxP)+width)
    plt.ylim(min(minP)-width,max(maxP)+width)
    
    n = len(P.vertices)
    for i in range(n):
        p1 = P.vertices[i]
        p2 = P.vertices[(i+1)%n]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],marker='o',color='b')

    m = len(Z.vertices)
    for j in range(m):
        p1 = Z.vertices[j]
        p2 = Z.vertices[(j+1)%m]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],marker='o',color='r')

    dist,p,q = hausdorff_distance(P,Z,metric=metric,full=False)
    plt.plot([p[0],q[0]],[p[1],q[1]],marker='*',color='k')

    if name is None:
        name = "render.png"
    plt.savefig(name,bbox_inches='tight')


def render_polytopes_close_ties(P: Polytope, Z: Zonotope, metric=2, name=None, thresh=0.96):
    """ Draw P and Z as well as lines for the close tie Hausdorff distances
    """
    render_polytopes(P,Z,metric=metric,name=name)

    close_pts = hausdorff_distance_close(P,Z,thresh,metric=metric)
    for p,q in close_pts:
        plt.plot([p[0],q[0]],[p[1],q[1]],marker="*",color='y')

    dist,p,q = hausdorff_distance(P,Z,metric=metric,full=False)
    plt.plot([p[0],q[0]],[p[1],q[1]],marker='*',color='k')

    if name is None:
        name = "render.png"
    plt.savefig(name,bbox_inches='tight')



if __name__ == "__main__":
    from examples import RandomNeuralNetwork
    from function import test_equal
    import sys
   
    verbose = True
    while True:
        NN = RandomNeuralNetwork((2,4,5,1),MAX=1,integer=False).NN
        h = NN.tropical(verbose=verbose)[0]
        try:
            MM = h.neural_network(verbose=verbose)
            break
        except:
            if len(sys.argv) > 1:
                print("failed")
                break
            continue
    sub = h.dual_diagram

    draw_polytope(h.newton_polytope, subdivision=sub, name='depth3.svg')
