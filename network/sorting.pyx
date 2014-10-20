import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def oneSort(list a, np.float64_t t, Py_ssize_t n) :
    cdef Py_ssize_t ppos, pos, b
    cdef object q
    b     = 0
    pos   = n - 1
    ppos  = n - 1

    if t < a[pos].time :
        while True :
            pos  = ppos
            ppos = (pos + b) >> 1
            if b == ppos :
                pos = b
                break
            if t > a[ppos].time :
                b    = ppos
                ppos = pos

    q     = a.pop(pos)
    ppos  = n - 2
    pos   = n - 2
    b     = 0

    if q.time < a[pos].time :
        while True :
            pos  = ppos
            ppos = (pos + b) >> 1
            if b == ppos :
                pos = b
                break
            if q.time > a[ppos].time :
                b    = ppos
                ppos = pos

    if q.time < a[pos].time :
        a.insert(pos, q)
    else :
        a.insert(pos+1, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def bisectSort(list a, object q, Py_ssize_t n) :
    cdef Py_ssize_t ppos, pos, b
    b     = 0
    pos   = n - 1
    ppos  = n - 1

    if q.time < a[pos].time :
        while True :
            pos  = ppos
            ppos = (pos + b) >> 1
            if b == ppos :
                pos = b
                break
            if q.time > a[ppos].time :
                b    = ppos
                ppos = pos

    if q.time < a[pos].time :
        a.insert(pos, q)
    else :
        a.insert(pos+1, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def twoSort(list a, object q1, object q2, Py_ssize_t n) :
    bisectSort(a, q1, n)
    bisectSort(a, q2, n+1)

@cython.boundscheck(False)
@cython.wraparound(False)
def oneBisectSort(list a, object q, np.float64_t t, Py_ssize_t n) :
    oneSort(a, t, n)
    bisectSort(a, q, n)
