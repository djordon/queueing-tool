import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def oneSort(list a, np.float64_t t, Py_ssize_t n) :
    cdef Py_ssize_t pos, top, bot
    cdef object q
    top = n - 1
    bot = 0
    pos = 0

    if t < a[bot]._time :
        while True :
            pos = (top + bot) >> 1
            if bot == pos :
                break
            if t < a[pos]._time :
                bot = pos
            else :
                top = pos

        q = a.pop(top)
    else :
        q = a.pop(0)

    top = n - 2
    bot = 0
    pos = 0

    if q._time < a[bot]._time :
        while True :
            pos = (top + bot) >> 1
            if bot == pos :
                break
            if q._time < a[pos]._time :
                bot = pos
            else :
                top = pos

        if q._time < a[top]._time :
            a.insert(top+1, q)
        else :
            a.insert(top, q)
    else :
        a.insert(0, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def bisectSort(list a, object q, Py_ssize_t n) :
    cdef Py_ssize_t pos, top, bot
    top = n - 1
    bot = 0
    pos = 0

    if q._time < a[bot]._time :
        while True :
            pos = (top + bot) >> 1
            if bot == pos :
                break
            if q._time < a[pos]._time :
                bot = pos
            else :
                top = pos

        if q._time < a[top]._time :
            a.insert(top+1, q)
        else :
            a.insert(top, q)
    else :
        a.insert(0, q)


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
