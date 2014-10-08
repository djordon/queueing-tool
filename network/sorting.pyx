cimport numpy as np
cimport cython

#ctypedef np.object_ DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def insertionsort_special(object a) :
    cdef unsigned int k, j
    for k in range(1, len(a)) :
        j = k
        while j > 0 and a[j] < a[j-1] :
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
def insertionsort_special2(object a) :
    cdef unsigned int k, j
    cdef unsigned int m = 0
    cdef bint swapped
    for k in range(1, len(a)) :
        j = k
        swapped = False
        while j > 0 and a[j,0] < a[j-1,0] :
            a[j], a[j-1] = a[j-1], a[j]
            swapped = True
            j = j - 1
        if swapped :
            m = m + 1
            if m == 2 :
                break
