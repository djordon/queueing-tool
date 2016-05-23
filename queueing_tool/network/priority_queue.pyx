cimport cython
from cpython cimport array
import array


cdef class PriorityQueue:
    """A specialized version of a priority queue for a QueueNetwork"""

    cdef public array.array array_times, array_edges
    cdef double [::1] sorted_times
    cdef double [::1] q_times
    cdef int [::1] sorted_edges
    cdef int actual_size
    cdef public int size, next_node
    cdef public double next_time

    def __cinit__(self, object keys=None, int n=0):
        pass

    def __init__(self, object keys=None, int n=0):
        cdef tuple key

        if keys is None:
            keys = []

        self.array_times = array.array('d', [key[0] for key in keys])
        self.array_edges = array.array('i', [key[1] for key in keys])

        self.sorted_times = self.array_times
        self.sorted_edges = self.array_edges

        self.size = len(keys)
        self.actual_size = self.size

        heapify(self.sorted_times, self.sorted_edges, self.size)

        self.q_times = array.array('d', [float('inf') for k in range(n)])

        for key in keys:
            self.q_times[key[1]] = key[0]

    @property
    def times(self):
        return self.array_times[:self.size]

    @property
    def edges(self):
        return self.array_edges[:self.size]

    def q_map(self):
        return array.array('d', self.q_times)

    @property
    def arraysize(self):
        return self.actual_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def pop(self):
        cdef double t
        cdef int k

        self.heappop()
        with nogil:
            while self.next_time != self.q_times[self.next_node]:
                if self.size == 0:
                    self.size -= 1
                    break
                self.heappop()

        if self.size < 0:
            self.size += 1
            return None
        return self.next_time, self.next_node

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def push(self, double t, int k):

        self.q_times[k] = t
        if self.size < self.actual_size:
            self.sorted_times[self.size] = t
            self.sorted_edges[self.size] = k

        else:
            array.resize_smart(self.array_times, 2 * self.size)
            array.resize_smart(self.array_edges, 2 * self.size)

            self.actual_size *= 2
            self.sorted_times = self.array_times
            self.sorted_edges = self.array_edges

            self.sorted_times[self.size] = t
            self.sorted_edges[self.size] = k

        self.size += 1
        _siftdown(self.sorted_times, self.sorted_edges, 0, self.size-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef void heappop(self) nogil:
        cdef double last_time, return_time
        cdef int last_node, return_node

        self.size -= 1
        last_time = self.sorted_times[self.size]
        last_node = self.sorted_edges[self.size]

        if self.size > 0:
            return_time = self.sorted_times[0]
            return_node = self.sorted_edges[0]

            self.sorted_times[0] = last_time
            self.sorted_edges[0] = last_node

            _siftup(self.sorted_times, self.sorted_edges, 0, self.size)

            self.next_time = return_time
            self.next_node = return_node
        else:
            self.next_time = last_time
            self.next_node = last_node


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _siftdown(double[:] heap, int[:] hmap, int startpos, int pos) nogil:
    cdef double newitem, parent
    cdef int parentpos, mapitem, father

    newitem = heap[pos]
    mapitem = hmap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        father = hmap[parentpos]

        if newitem < parent:
            heap[pos] = parent
            hmap[pos] = father
            pos = parentpos
            continue
        break

    heap[pos] = newitem
    hmap[pos] = mapitem


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _siftup(double[:] heap, int[:] hmap, int pos, int endpos) nogil:
    cdef double newitem
    cdef int childpos, rightpos, mapitem
    cdef int startpos = pos

    newitem = heap[pos]
    mapitem = hmap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position

    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1

        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos

        # Move the smaller child up.
        heap[pos] = heap[childpos]
        hmap[pos] = hmap[childpos]
        pos = childpos
        childpos = 2*pos + 1

    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    hmap[pos] = mapitem
    _siftdown(heap, hmap, startpos, pos)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void heapify(double[:] heap, int[:] hmap, int n) nogil:
    cdef int i
    for i in reversed(range(n//2)):
        _siftup(heap, hmap, i, n)
