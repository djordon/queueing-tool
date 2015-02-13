import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _choice(list pr, np.float64_t u, Py_ssize_t n) :
    cdef np.float64_t z
    cdef int k

    if u <= pr[0] :
        k = 0
    else :
        z = pr[0]
        for k in range(1, n) :
            if u <= z + pr[k] :
                break
            else :
                z  += pr[k]

    return k
