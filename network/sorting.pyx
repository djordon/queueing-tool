from heapq import heappush
cimport numpy as np
cimport cython

#ctypedef np.object_ DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def insertionsort_special(np.ndarray[object, ndim=1] a) :
    cdef Py_ssize_t k, j
    for k in range(1, len(a)) :
        j = k
        while j > 0 and a[j] < a[j-1] :
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
def insertionsort_special2(object a, int n) :
    cdef Py_ssize_t k, j
    for k in range(1, n) :
        j = k
        while j > 0 and a[j] < a[j-1] :
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
def twosort(object a, int n) :
    cdef Py_ssize_t k, j
    cdef bint one, two
    for k in range(1, n) :
        if a[k] < a[k-1] :
            j = k
            if one :
                two = True
            while j > 0 and a[j] < a[j-1] :
                a[j], a[j-1] = a[j-1], a[j]
                j = j - 1
            one = True
            if one and two :
                return

@cython.boundscheck(False)
@cython.wraparound(False)
def onesort(object a, int n) :
    cdef Py_ssize_t k
    for k in range(1, n) :
        if a[k] < a[k-1] :
            while k > 0 and a[k] < a[k-1] :
                a[k], a[k-1] = a[k-1], a[k]
                k = k - 1
                return
