from cython.parallel import prange, parallel

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp

ctypedef cnp.int64_t I64_t
ctypedef cnp.float64_t F64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _pairwise_distance(cnp.ndarray[F64_t, ndim=2] pts, int n):
    cdef I64_t [:, ::1] edges = np.empty((n * (n - 1) // 2, 2), int)
    cdef F64_t [::1] dists = np.empty(n * (n - 1) // 2)
    cdef F64_t x, y
    cdef int i, j, k

    num_threads = max(openmp.omp_get_max_threads() - 2, 1)

    with nogil, parallel(num_threads=num_threads):
        for k in prange(n-1):
            for j in range(k+1, n):
                x = pts[k, 0] - pts[j, 0]
                y = pts[k, 1] - pts[j, 1]
                dists[i] = x**2 + y**2
                edges[i, 0] = k
                edges[i, 1] = j
                i += 1

    return np.array(dists), np.array(edges)
