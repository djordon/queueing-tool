cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def departsort(list a, object q, Py_ssize_t pos) :
    cdef Py_ssize_t k, j, ppos
    cdef bint issorted = False
    ppos  = pos - 1

    for k in range(1, pos) :
        j = k
        while j > 0 and a[j].time < a[j-1].time :
            issorted = True
            a[j], a[j-1] = a[j-1], a[j]
            j = j - 1
        if issorted :
            break

    while ppos > 0 and  q.time < a[ppos].time :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q.time < a[pos-1].time :
        pos = pos - 1

    a.insert(pos, q)


@cython.boundscheck(False)
@cython.wraparound(False)
def arrivalsort(list a, object q, Py_ssize_t pos) :
    cdef Py_ssize_t ppos
    ppos  = pos - 1
    
    while ppos > 0 and  q.time < a[ppos].time :
        pos  = ppos
        ppos = (pos - 1) >> 1

    while pos > 0 and q.time < a[pos-1].time :
        pos = pos - 1

    a.insert(pos, q)

