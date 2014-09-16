import scipy.optimize as op
import numpy as np

## intersects(v1, v2, v3, v4)
## This function tests to see if the line segment connecting
## the first pair of points (v1 and v2), intersects
## with the line connecting the second pair (v3 and v4)
## Parallel are returned as non-intersecting (even if they are 
## on the same line)

## point_region(point, points, edges)
## Test whether a point is within a region. The region is assumed to
## the convex hull of the points supplied in the variable 'points'.

## extreme_points(points)
## Given a list of points, find the points that make up the boundary
## of the given points. That is, every other point is in the convex
## hull of the returned points. A list of edges that make up the 
## boundary is returned.

EPS = 1e-7


def intersects(v1, v2, v3, v4) :
    M   = np.array([v1 - v2, v4 - v3]).T
    det = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    if det == 0 :
        return False
    M[0, 0], M[1, 1] = M[1, 1], M[0, 0]
    M[0, 1] = -M[0, 1]
    M[1, 0] = -M[1, 0]
    y = v4 - v2
    x = np.dot(M, y) / det
    return (0 <= x[0] <= 1) and (0 <= x[1] <= 1)


def _convex_distance(ab, *args) :
    a, b  = ab
    p, x  = args[:2]
    y, z  = args[2:]
    ans   = b * x + a*(1-b) * y + (1-a)*(1-b) * z - p
    return np.dot(ans, ans)


def _convex_boundary(points) :
    pairs   = []
    nPoints = len(points)
    nPairs  = int(nPoints * (nPoints-1) / 2)
    bndry   = np.ones(nPairs, bool)
    
    for k in range(nPoints-1) :
        for j in range(k+1, nPoints) :
            pairs.append([k, j])
    
    for k in range(nPairs-1) :
        pair = pairs[k]
        for j in range(k+1, nPairs) :
            if not bndry[k] and not bndry[j] :
                continue
            if not pairs[j][0] in pair and not pairs[j][1] in pair :
                p1, p2  = points[pair]
                p3, p4  = points[pairs[j]]
                if intersects(p1, p2, p3, p4) :
                    bndry[[k, j]] = False
    
    eindex  = np.where(bndry)[0]
    edges   = np.array(pairs)[eindex]
    return edges


def _region_test(p, points, edges) :
    p1  = points[p]
    ept = set(np.unique(edges))
    reg = {p : False for p in ept}
    ept = ept - {p}
    
    for edge in edges :
        if p in edge :
            continue
        for point in ept :
            if point in edge :
                continue
            pt      = points[point]
            p2, p3  = points[edge]
            optim   = op.minimize(_convex_distance, x0=np.array([0.5, 0.5]),
                                  args=(pt, p1, p2, p3), method='L-BFGS-B',
                                  bounds=[(0,1), (0,1)] )
            if optim.fun < EPS :
                reg[point] = True
    
    return np.array([e for e in edges if not (reg[e[0]] or reg[e[1]]) ])


def extreme_points(points) :
    if len(points) < 4 :
        if len(points) == 3 :
            edges   = np.array([[0,1], [0,2], [1,2]])
            return edges
        else :
            raise Exception("Need 3 of more points for extreme_points; given %s." % (len(points)))
    
    edges = _convex_boundary(points)
    edges = _region_test(edges[0][0], points, edges)
    edges = _region_test(edges[0][1], points, edges)
    return edges


def point_region(pts, points, edges=None) :
    if edges == None :
        edges = extreme_points(points)
    if len(pts.shape) == 1 :
        pts = (pts,)
    p1  = points[edges[0][0]]
    ans = np.zeros(len(pts), bool)
    
    for k in range(len(pts)) :
        for edge in edges :
            if edges[0][0] in edge :
                continue
            p2, p3  = points[edge]
            optim   = op.minimize(_convex_distance, x0=np.array([0.5, 0.5]), 
                                  args=(pts[k], p1, p2, p3), method='L-BFGS-B',
                                  bounds=[(0,1), (0,1)])
            if optim.fun < EPS :
                ans[k] = True
                break
    
    return ans if len(pts) > 1 else ans[0]
