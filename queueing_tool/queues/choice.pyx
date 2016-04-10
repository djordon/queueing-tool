cimport cython
from cpython cimport array
import array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _choice(double [::1] pr, double u, Py_ssize_t n):
    cdef double z
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
@cython.nonecheck(False)
def _argmin(list a):
    cdef int minv, amin, k, v
    minv = a[0]
    amin = -1
    for k, v in enumerate(a[1:]):
        if v < minv:
            minv = v
            amin = k
    return amin + 1
