from   numpy import dot, array
import numpy as np
import scipy.optimize as op

np.set_printoptions(precision=2,suppress=True,threshold=2000)

# Boundary points
bps = array([[1,0], [0,1], [-1,-1]])

p   = array([0, 0]) # point supplied by user

# pick any point in the set, then
# pick two other point in the set
x0  = bps[0] 
x1  = bps[1]
x2  = bps[2]

def convex_min(y, *args) :
    a, b  = y
    p, x  = args[:2]
    y, z  = args[2:]
    ans   = b * x + a*(1-b) * y + (1-a)*(1-b) * z - p
    return dot(ans, ans)

bnds  = ( (0,1), (0,1) )
ans   = op.minimize( convex_min, x0=array([0.5, 0.5]), args=(p,x0,x1,x2), method='L-BFGS-B', bounds=bnds)


def convex_boundary( points ) :
    if points.shape[0] < 4 :
        if points.shape[0] == 3 :
            edges   = np.array([[0,1], [0,2], [1,2]])
            sortedp = [0, 1, 2]
            return edges, sortedp
        else :
            ## Throw some flag
            return
    
    ## This function tests to see if the line connecting
    ## the first pair of points (v1 and v2), intersects
    ## with the line connecting the second pair (v3 and v4) 
    def intersect( v1, v2, v3, v4 ) :
        M   = np.array([v2 - v1, v4 - v3]).T
        if np.linalg.det(M) == 0 :
          return False
        y   = v4 - v1
        x   = np.linalg.solve(M,y)
        return 0 <= x[0] <= 1 and 0 <= x[1] <= 1
    
    pairs   = []
    nPoints = points.shape[0]
    nPairs  = int( nPoints*(nPoints-1)/2 )
    inter   = np.zeros( (nPairs, nPairs), bool)
    
    for k in range(nPoints) :
        if k < n-1:
            for j in range(k+1, nPoints) :
                pairs.append( [k,j] )
    
    for k in range( nPairs ) :
        if k < nPoints-1:
            pair1 = pairs[k]
            for j in range(k+1, nPairs ) :
                if not pairs[j][0] in pair1 and not pairs[j][1] in pair1 :
                    pair2   = pairs[j]
                    p1, p2  = points[pair1[0]], points[pair1[1]]
                    p3, p4  = points[pair2[0]], points[pair2[1]]
                    if intersect( p1, p2, p3, p4 ) :
                        inter[k, j] = True
                        inter[j, k] = True
    
    eindex  = np.where( np.sum(inter, 1) == 0 )[0]
    edges   = np.array( pairs )[eindex]
    adj     = [ [] for k in range(nPoints) ]
    
    for pair in edges :
        adj[pair[0]].append( pair[1] )
        adj[pair[1]].append( pair[0] )
    
    a, b    = 0, adj[0][0]
    sortedp = []
    sortedp.append(0)
    sortedp.append(adj[0][0])
    
    for k in range(nPoints-2) :
        if adj[b][0] == a :
            sortedp.append( adj[b][1] )
            a, b = b, adj[b][1]
        else :
            sortedp.append( adj[b][0] )
            a, b = b, adj[b][0]
    
    return edges, sortedp



def test_convexity( points ) :
    pass



import numpy as np
import matplotlib.pyplot as plt

points  = np.array([[0,1],
                    [0.5, 2],
                    [1, 1.75],
                    [2, .5],
                    [1.5, 0],
                    [0.25, 0.25]])


fig, ax = plt.subplots()
ax.set_xlim([-0.25, 2.25])
ax.set_ylim([-0.25, 2.25])
ax.plot(points[tmp,0], points[tmp,1])
plt.show()

fig, ax = plt.subplots()
t   = np.asmatrix( np.arange(-1,2,.1) )
ps  = np.array( np.asmatrix(points[0]).T * t + np.asmatrix(points[1]).T * (1-t) ).T
ax.plot(ps[:,0], ps[:,1], 'o')
plt.show()


v1  = points[0]
v2  = points[2]
v3  = points[1]
v4  = points[-1]
v5  = points[-2]




ans = np.linalg.solve(M2,y)

fig, ax = plt.subplots()
ax.set_xlim([-0.5, 2.25])
ax.set_ylim([-0.5, 2.25])
ax.plot(points[:,0], points[:,1], 'o')


pstar = (1-ans[0]) * v1 + ans[0] * v3
ax.plot( pstar[0], pstar[1], 'o' )

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






