import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')

# starting tropical polyonmial (monomial, coefficient) pairs
f = {0:-2,1:1,2:1,3:0}

def active_monomials(f):
    # get the active monomials of f
    degree = max(f.keys())

    active_monomials = {0:f[0]}
    tail = 0
    while tail < degree:
        highest_slope = -np.Inf 
        for m in f.keys():
            if m > tail:
                new_slope = (f[m]-f[tail])/(m-tail)
                if new_slope > highest_slope:
                    tip = m
                    highest_slope = new_slope
        active_monomials[tip] = f[tip]
        tail = tip
                
    return active_monomials 
        
def principal_components(f):
    assert f.get(0) is not None
    assert f.get(max(f.keys())) == 0

    f = active_monomials(f)
    h = f[0]
    
    pts = [[k,v] for k,v in f.items()]
    pts.sort(key=lambda l:l[0])
    pts = np.array(pts)

    diffs = []
    for i in range(1,len(pts)):
       diffs += [pts[i]-pts[i-1]]
    diffs = np.array(diffs).T

    return diffs[0],diffs[1],h

def solve_NN(f,b=None):
    X,Y,h = principal_components(f)
    if b is None:
        b = np.ones_like(X)

    a = X/b
    c = -Y/b
    return a,b,c,f[0]

def random_complexity(n,c,samples=10000):
    complexities = []
    for _ in range(samples):
        f = random_poly(n,c=c,show=False)
        complexities += [len(solve_NN(f)[0])]
    complexities = np.array(complexities) 

    hist = [[d,len(complexities[complexities==d])] for d in range(n+1)]
    hist = np.array(hist).T
    return hist

def plot_curves(bottom,top,samples=10000):
    for deg in range(bottom,top+1):
        hist = random_complexity(deg,1,samples=samples)
        plt.plot(hist[0],hist[1],label="degree {}".format(deg))
    #plt.savefig("hist.png",dpi=500)
    plt.legend()
    plt.show()

def print_poly(f):
    s = ""
    for d,c in f.items():
        if c >= 0:
           s+="+ "
        s+=str(c)[:5]+"x^{} ".format(d)
    if s[0] == "+":
        s = s[2:]
    print(s)

def plot_pair(f,NN):
    plt.clf()
    inp = np.arange(-1,1,0.05)

    f_l = lambda x: evaluate_poly(f,x)
    NN_l = lambda x: evaluate_NN(NN[0],NN[1],NN[2],NN[3],x)

    f_vals = np.array(list(map(f_l,inp)))
    NN_vals = np.array(list(map(NN_l,inp)))

    fig,ax = plt.subplots(1,2)
    ax[0].plot(inp,f_vals)
    ax[0].set_title("Graph of Polynomial")
    ax[1].plot(inp,NN_vals)
    ax[1].set_title("Graph of Neural Network")
    fig.show()

def evaluate_poly(f,x):
    terms = [p*x+c for p,c in f.items()]
    return max(terms)

def evaluate_NN(a,b,c,d,x):
    z1 = a*x
    z2 = np.array([max(zz,cc) for zz,cc in zip(z1,c)])
    return max(b@z2,d)

def random_poly(n,c=2,show=True):
    coefs = [c*np.random.rand() for _ in range(n)]
    coefs+= [0]
    f = {i:coefs[i] for i in range(n+1)}
    if show:
        print_poly(f)
    return f

def check_solver(f):
    a,b,c,d = solve_NN(f)
    test_set = np.arange(-50,50,0.5)
    tester = lambda x : evaluate_poly(f,x)-evaluate_NN(a,b,c,d,x)

    return sum([tester(x) for x in test_set])

