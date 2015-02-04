class UnionFind :
    """The union-find data structure with union by rank and path compression.

    The UnionFind data structure is a collection of objects that supports
    the union and find operations (described below). Each object in the collection
    belongs to a set, which is identified by its leader. To sets can be fused
    together to form a new set (union), and for any element in the collection,
    we can find the leader of the set to which it belongs (find)

    Parameters
    ----------
    S : set (can be any object that is iterable whose elements are hashable by dict)
        A collection of objects.

    Attributes
    ----------
    nClusters : int
        The number of clusters contained in the data-structure.
    """
    def __init__(self, S) :
        self._leader     = dict( (s, s)   for s in S)
        self._size       = dict( (s, 1)   for s in S)
        self._rank       = dict( (s, 0)   for s in S)
        self.nClusters  = len(S)


    def size(self, s):
        """Returns the number of elements in the set that ``s`` belongs to.

        Parameters
        ----------
        s : object
            An object

        Returns
        -------
        output : int
            The number of elements in the set that ``s`` belongs to.
        """
        leader = self.find(s)
        return self._size[leader]


    def find(self, s) :
        """Find the leader of the element ``s`` with path compression.

        Locates the leader to which the element ``s`` belongs.

        Parameters
        ----------
        s : object
            An object.

        Returns
        -------
        output : object
            The leader of the set that contains ``s``.
        """
        pSet    = [s]
        parent  = self._leader[s]

        while parent != self._leader[parent] :
            pSet.append(parent)
            parent = self._leader[parent]

        if len(pSet) > 1 :
            for a in pSet :
                self._leader[a] = parent

        return parent


    def union(self, a, b) :
        """Union the set that contains ``a`` with the set that contains ``b``.

        Merges the set that contains ``a`` with the set that contains ``b``.

        Parameters
        ----------
        a, b : objects
            Two objects whose sets are to be merged.
        """
        s1, s2  = self.find(a), self.find(b)
        if s1 != s2 :
            r1, r2  = self._rank[s1], self._rank[s2]
            if r2 > r1 :
                r1, r2  = r2, r1
                s1, s2  = s2, s1
            if r1 == r2 :
                self._rank[s1]  += 1

            self._leader[s2]  = s1
            self._size[s1]   += self._size[s2]
            self.nClusters   -= 1
