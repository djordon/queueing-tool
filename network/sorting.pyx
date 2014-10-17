import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def zeroOneSort(list a, object q, np.float64_t t, Py_ssize_t n) :
    cdef Py_ssize_t k, j, ppos, pos, b
    cdef bint issorted = False
    b     = 0
    pos   = n
    ppos  = n - 1

    while ppos - b > 0 and a[ppos-1].time < a[ppos].time :
        pos  = ppos
        ppos = ((pos - 1 - b) >> 1) + b
        if t > a[ppos].time :
            b    = pos
            ppos = pos

    q2    = a.pop(pos)
    ppos  = pos - 1
    b     = 0

    while ppos - b > 0 :
        pos  = ppos
        ppos = ((pos - 1 - b) >> 1) + b
        if q2.time > a[ppos].time :
            b    = pos
            ppos = pos

    a.insert(pos, q2)

    ppos  = n - 1
    pos   = n
    b     = 0

    while ppos - b > 0 :
        pos  = ppos
        ppos = ((pos - 1 - b) >> 1) + b
        if q.time > a[ppos].time :
            b    = pos
            ppos = pos

    a.insert(pos, q)

@cython.boundscheck(False)
@cython.wraparound(False)
def zeroSort(list a, Py_ssize_t n) :
    cdef Py_ssize_t k, j
    cdef bint issorted = False

    for k in range(1, n) :
        j = k
        while j > 0 and a[j].time < a[j-1].time :
            issorted = True
            a[j], a[j-1] = a[j-1], a[j]
            j = j - 1
        if issorted :
            break


@cython.boundscheck(False)
@cython.wraparound(False)
def oneSort(list a, object q, Py_ssize_t pos) :
    cdef Py_ssize_t ppos
    ppos  = pos - 1
    
    while ppos > 0 and  q.time < a[ppos].time :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q.time < a[pos-1].time :
        pos = pos - 1

    a.insert(pos, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def twoSort(list a, object q1, object q2, Py_ssize_t n) :
    cdef Py_ssize_t ppos, pos
    pos   = n
    ppos  = pos - 1
    
    while ppos > 0 and  q1.time < a[ppos].time :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q1.time < a[pos-1].time :
        pos = pos - 1

    a.insert(pos, q1)
    pos   = n + 1
    ppos  = pos - 1
    
    while ppos > 0 and  q2.time < a[ppos].time :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q2.time < a[pos-1].time :
        pos = pos - 1

    a.insert(pos, q2)
