import numpy as np
cimport numpy as cnp
cimport cython


ctypedef cnp.int64_t I64_t
ctypedef cnp.float64_t F64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _choice(cnp.ndarray[F64_t, ndim=1] pr, cnp.float64_t u, Py_ssize_t n):
    cdef cnp.float64_t z
    cdef int k

    if u <= pr[0]:
        k = 0
    else:
        z = pr[0]
        with nogil:
            for k in range(1, n):
                if u <= z + pr[k]:
                    break
                else:
                    z += pr[k]

    return k


@cython.boundscheck(False)
@cython.wraparound(False)
def _argmin(list a):
    cdef int minv, amin, k, v
    minv = a[0]
    amin = -1
    for k, v in enumerate(a[1:]):
        if v < minv:
            minv = v
            amin = k
    return amin + 1
