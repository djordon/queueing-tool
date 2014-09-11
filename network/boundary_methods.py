from   numpy import dot, array
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

np.set_printoptions(precision=2,suppress=True,threshold=2000)

EPS = 1e-14
# Boundary points
points  = np.array([[0,1],
                    [0.5, 2],
                    [1, 1.75],
                    [2, .5],
                    [1.5, 0],
                    [0.25, 0.25]])


## This function tests to see if the line segment connecting
## the first pair of points (v1 and v2), intersects
## with the line connecting the second pair (v3 and v4) 

def intersects(v1, v2, v3, v4) :
    M   = np.array([v1 - v2, v4 - v3]).T
    if np.linalg.det(M) == 0 :
        return False
    y   = v4 - v2
    x   = np.linalg.solve(M,y)
    return (-EPS <= x[0] <= 1+EPS) and (-EPS <= x[1] <= 1+EPS)


def convex_boundary(points) :
    if points.shape[0] < 4 :
        if points.shape[0] == 3 :
            edges   = np.array([[0,1], [0,2], [1,2]])
            sortedp = [0, 1, 2]
            return edges, sortedp
        else :
            ## Throw some flag
            return
    
    pairs   = []
    nPoints = points.shape[0]
    nPairs  = int( nPoints*(nPoints-1)/2 )
    bndry   = np.ones(nPairs, bool)
    
    for k in range(nPoints-1) :
        for j in range(k+1, nPoints) :
            pairs.append([k, j])
    
    for k in range(nPairs-1) :
        pair = pairs[k]
        for j in range(k+1, nPairs) :
            if not pairs[j][0] in pair and not pairs[j][1] in pair :
                p1, p2  = points[pair]
                p3, p4  = points[pairs[j]]
                if intersects(p1, p2, p3, p4) :
                    bndry[[k, j]] = False
    
    eindex  = np.where(bndry)[0]
    edges   = np.array(pairs)[eindex]
    adj     = [ [] for k in range(nPoints)]
    
    for pair in edges :
        adj[pair[0]].append(pair[1])
        adj[pair[1]].append(pair[0])
    
    a, b    = edges[0][0], adj[edges[0][0]][0]
    sortedp = []
    sortedp.append(edges[0][0])
    sortedp.append(adj[edges[0][0]][0])
    if False :
        for k in range( edges.shape[0]-2 ) :
            if adj[b][0] == a :
                sortedp.append( adj[b][1] )
                a, b = b, adj[b][1]
            else :
                sortedp.append( adj[b][0] )
                a, b = b, adj[b][0]
    
    return edges, sortedp


def convex_distance(ab, *args) :
    a, b  = ab
    p, x  = args[:2]
    y, z  = args[2:]
    ans   = b * x + a*(1-b) * y + (1-a)*(1-b) * z - p
    return dot(ans, ans)

## Test whether a point is within a region. The region is the
## convex hull of the points supplied in the variable region.

def point_region( region, point ) :
    nPoints, p1     = len(region), region[0]
    edges, sortedp  = convex_boundary( region )
    in_region       = False
    
    for k in range(2, nPoints) :
        p2, p3  = region[(k-1):(k+1)]
        ans     = op.minimize(convex_boundary, x0=array([0.5, 0.5]), args=(point,p1,p2,p3),
                              method='L-BFGS-B', bounds=[(0,1), (0,1)] )
        if ans.fun < EPS :
            in_region = True
            break
    
    return in_region

def test_convexity( points ) :
    pass






points      = 10*np.random.random( (20,2) )
edges, ps   = convex_boundary( points ) 
fig, ax = plt.subplots()
ax.set_xlim([-0.25, 10.25])
ax.set_ylim([-0.25, 10.25])
for k in points[edges] :
    ax.plot(k[:,0], k[:,1], 'k')

ax.plot(points[:,0], points[:,1], 'o')
plt.show()







"""
import cProfile

p     = array([5,5])
bnds  = ( (0,1), (0,1) )
pr    = cProfile.Profile()
pr.enable()
for k in range(500):
    ans = op.minimize( convex_min, x0=array([0.5, 0.5]), args=(p,x0,x1,x2), method='L-BFGS-B', bounds=bnds)

pr.disable()
pr.print_stats(sort='time')


SLSQP       0.716 seconds
CG          0.897 seconds
BFGS        0.922 seconds
L-BFGS-B    0.494 seconds
TNC         0.703 seconds
COBYLA      0.467 seconds
Powell      3.320 seconds
Nelder-Mead 1.789 seconds

def jacobian(y, *args) :
    a, b  = y
    p, x  = args[:2]
    y, z  = args[2:]
    fun   =  b * x + a*(1-b) * y + (1-a)*(1-b) * z - p
    ans   = array([0,0])
    ans[0]= 2 * dot( fun, (1-b)*(y-z) )
    ans[1]= 2 * dot( fun, x - a*y - (1-a)*z ) 
    return ans
"""






