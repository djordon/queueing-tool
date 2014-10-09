cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def departsort(object a, object q, Py_ssize_t pos) :
    cdef Py_ssize_t k, j, ppos
    cdef bint issorted = False
    ppos  = pos - 1

    for k in range(1, pos) :
        j = k
        while j > 0 and a[j] < a[j-1] :
            issorted = True
            a[j], a[j-1] = a[j-1], a[j]
            j -= 1
        if issorted :
            break

    while ppos > 0 and  q < a[ppos] :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q < a[pos-1] :
        pos = pos - 1

    a.insert(pos, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def onesort(object a, object q, Py_ssize_t pos) :
    cdef Py_ssize_t ppos
    ppos  = pos - 1
    
    while ppos > 0 and  q < a[ppos] :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q < a[pos-1] :
        pos = pos - 1

    a.insert(pos, q)

