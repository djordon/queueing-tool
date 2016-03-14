import collections
import copy
import numbers
from heapq import heappush, heappop

from numpy.random import uniform, exponential
from numpy import infty
import numpy as np

from queueing_tool.queues.agents import Agent, InftyAgent


def poisson_random_measure(rate, rate_max, t):
    """A function that returns the arrival time of the next arrival for a
    Poisson random measure.

    Parameters
    ----------
    rate : function
        The *intensity function* for the measure, where ``rate(t)`` is the
        expected arrival rate at time ``t``. This function should be bounded.
    rate_max : float
        The maximum value of the ``rate`` function.
    t : float
        The start time from which to simulate the next arrival time.

    Returns
    -------
    float
        The time of the next arrival.

    Notes
    -----
    This function returns the time of the next arrival, where the distribution
    of the number of arrivals between times :math:`t` and :math:`t+s` is
    Poisson with mean

    .. math::

       \int_{t}^{t+s} dx \, r(x)

    where :math:`r(t)` is the supplied ``rate`` function. This function can
    only simulate processes that have bounded intensity functions. See chapter
    6 of [3]_ for more on the mathematics behind Poisson random measures; the
    book's publisher, Springer, has that chapter available online (`pdf`_\).

    A Poisson random measure is sometimes called a non-homogeneous Poisson
    process. A Poisson process is a special type of Poisson random measure.

    .. _pdf: http://www.springer.com/cda/content/document/cda_downloaddocument/9780387878584-c1.pdf

    Examples
    --------
    Suppose you wanted to modeled the arrival process as a Poisson random
    measure with rate function :math:`r(t) = 2 + \sin( 2\pi t)`. Then you could
    do so as follows:

    >>> import queueing_tool as qt
    >>> import numpy as np
    >>> rate  = lambda t: 2 + np.sin(2 * np.pi * t)
    >>> arr_f = lambda t: qt.poisson_random_measure(rate, 3, t)

    References
    ----------
    .. [3] Cinlar, Erhan. *Probability and stochastics*. Graduate Texts in\
           Mathematics. Vol. 261. Springer, New York, 2011.\
           :doi:`10.1007/978-0-387-87859-1`
    """
    scale = 1.0 / rate_max
    t     = t + exponential(scale)
    while rate_max * uniform() > rate(t):
        t   = t + exponential(scale)
    return t



class QueueServer(object):
    """The base queue-server class.

    Built to work with the :class:`.QueueNetwork` class, but can stand alone
    as a multi-server queue. It supports a capped pool of potential arrivals
    using the ``active_cap`` attribute, as well as the stopping time attribute,
    ``deactive_t``, after which no more arrivals can enter the queue. When
    connected to a network of queues via a :class:`.QueueNetwork`\, the
    ``active_cap`` and ``deactive_t`` attributes applies only to arrivals from
    outside the network, the ``QueueServer`` instance always accepts arrivals
    from other queues inside the :class:`.QueueNetwork`\.

    This class supports arbitrary arrival and service distribution functions
    (that can depend on time but do not on the state of the system).

    Note that of the following parameters are assigned to an attribute of the
    same name.

    Parameters
    ----------
    nServers : int or :const:`.numpy.infty` (optional, the default is ``1``)
        The number of servers servicing agents.
    arrival_f : function (optional, default: ``lambda t: t + exponential(1)``)
        A function that returns the time of next arrival from outside the
        network. When this function is called, ``t`` is always taken to be the
        current time. **Should not return any values less than** ``t``, that is,
        ``arrival_f(t) >= t`` should always be true.
    service_f : function (optional, default: ``lambda t: t + exponential(0.9)``)
        A function that returns the time of an agent's service time. When this
        function is called, ``t`` is the time the agent is entering service.
        **Should not return any values less than** ``t``, that is,
        ``service_f(t) >= t`` should always be true.
    edge : 4-:class:`.tuple` of int (optional, the default is ``(0,0,0,1)``)
        A tuple that uniquely identifies which edge this queue lays on and the
        edge type. The first slot of the tuple is the source vertex, the second
        slot is the target vertex, and the third slot is the ``edge_index`` of
        that edge, and the last slot is the edge type for this queue. This is
        automatically created when a :class:`.QueueNetwork` instance is created.
    AgentClass : class (optional, the default is the :class:`~Agent` class)
        A class object for an :class:`.Agent` or any class object that has
        inherited the :class:`.Agent` class.
    active_cap : int (the default is :const:`.numpy.infty`\)
        The maximum number of arrivals the queue will accept from outside the
        network.
    deactive_t : float (the default is :const:`.numpy.infty`\)
        Sets a stopping time, after which no more arrivals (from outside the
        network) will attempt to enter the ``QueueServer``.
    collect_data : bool (the default is ``False``)
        A bool that defines whether the queue collects each :class:`.Agent`\'s
        arrival, service start, and departure times, as well as the number of
        other agents in the queue upon arrival.
    colors : dict (optional)
        A dictionary of the colors used when drawing the graph. The possible
        colors are:

            ``edge_loop_color``
                The default color of the edge if the edge is a loop.
            ``edge_color``
                The normal color a non-loop edge.
            ``vertex_fill_color``
                The normal fill color for a vertex; this also colors the target
                vertex in the graph.
            ``vertex_color``
                The color of the vertex pen of the target vertex.

        The defaults are listed in the notes.
    seed : int (optional)
        If supplied ``seed`` is used to initialize numpy's psuedorandom
        number generator.

    Attributes
    ----------
    active : bool
        Returns whether the queue accepts arrivals from outside the network
        (the queue will always accept arrivals from inside the network). The
        default is false. To change call :meth:`.set_active`\.
    current_time : float
        The time of the last event.
    time : float
        The time of the next event.
    nDepartures : int
        The total number of departures from the queue.
    nSystem : int
        The number of agents in the entire ``QueueServer`` -- this includes
        those being served and those waiting to be served.
    nArrivals : list
        A list with two entries. The first slot is the total number of arrivals,
        while the second slot is the number of arrivals from outside the network.
    data : dict
        Keeps track of each :class:`.Agent`\'s arrival, service start, and
        departure times, as well as how many other agents were waiting to be
        served and the total number agents in the system (upon arrival). The
        keys are the :class:`.Agent`\'s unique ``issn``\, and the values is a
        list of lists. Each time an agent arrives at the queue it appends this
        data to the end of the list. Use :meth:`.fetch_data` to retrieve a
        formated version of this data.

    Examples
    --------
    The following code constructs an :math:`\\text{M}_t/\\text{GI}/5`
    ``QueueServer`` with mean utilization rate :math:`\\rho = 0.8`. The
    arrivals are modeled as a Poisson random measure with rate function
    :math:`r(t) = 2 + 16 \sin^2(\pi t / 8)` and a service distribution that is
    gamma with shape and scale parameters 4 and 0.1 respectively. To create
    such a queue run:

    >>> import queueing_tool as qt
    >>> import numpy as np
    >>> rate = lambda t: 2 + 16 * np.sin(np.pi * t / 8)**2
    >>> arr = lambda t: qt.poisson_random_measure(rate, 18, t)
    >>> ser = lambda t : t + np.random.gamma(4, 0.1)
    >>> q = qt.QueueServer(5, arrival_f=arr, service_f=ser, seed=13)

    Before you can simulate the queue, it must be set to active; also, no data
    is collected by default, we change these with the following:

    >>> q.set_active()
    >>> q.collect_data = True

    To simulate 12000 events and collect the data run

    >>> q.simulate(n=12000)
    >>> data = q.fetch_data()

    Notes
    -----
    This is a generic multi-server queue implimentation (see [4]_).
    In `Kendall's notation`_\, this is a
    :math:`\\text{GI}_t/\\text{GI}_t/c/\infty/N/\\text{FIFO}` queue class,
    where :math:`c` is set by ``nServers`` and :math:`N` is set by
    ``active_cap``. See chapter 1 of [3]_ (pdfs from `the author`_ and
    `the publisher`_) for a good introduction to the theory behind the
    multi-server queue.

    Each queue sits on an edge in a graph. When drawing the graph, the queue
    colors the edges. If the target vertex does not have any loops, the number
    of agents in this queue affects the target vertex's color as well.

    Some defaults:

    >>> default_colors = {'edge_loop_color'   : [0, 0, 0, 0],
    ...                   'edge_color'        : [0.9, 0.9, 0.9, 0.5],
    ...                   'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
    ...                   'vertex_color'      : [0.0, 0.5, 1.0, 1.0]}


    References
    ----------
    .. [3] Harchol-Balter, Mor. *Performance Modeling and Design of Computer\
           Systems: Queueing Theory in Action*. Cambridge University Press,\
           2013. ISBN: `9781107027503`_.

    .. [4] *Queueing Theory*, Wikipedia `<http://en.wikipedia.org/wiki/Queueing_theory>`_.


    .. _Kendall's notation: http://en.wikipedia.org/wiki/Kendall%27s_notation
    .. _the author: http://www.cs.cmu.edu/~harchol/PerformanceModeling/chpt1.pdf
    .. _the publisher: http://assets.cambridge.org/97811070/27503/excerpt/9781107027503_excerpt.pdf
    .. _9781107027503: http://www.cambridge.org/us/9781107027503
    """

    _default_colors = {
        'edge_loop_color'  : [0, 0, 0, 0],
        'edge_color'       : [0.9, 0.9, 0.9, 0.5],
        'vertex_fill_color': [1.0, 1.0, 1.0, 1.0],
        'vertex_color'     : [0.0, 0.5, 1.0, 1.0]
    }

    def __init__(self, nServers=1, arrival_f=lambda t: t + exponential(1),
                 service_f=lambda t: t + exponential(0.9), edge=(0,0,0,1),
                 AgentClass=Agent, collect_data=False, active_cap=infty,
                 deactive_t=infty, colors=None, seed=None, **kwargs):

        if not isinstance(nServers, numbers.Integral) and nServers is not infty:
            msg = "nServers must be an integer or infinity."
            raise TypeError(msg)
        elif nServers <= 0:
            msg = "nServers must be a positive integer or infinity."
            raise ValueError(msg)

        self.edge         = edge
        self.nServers     = nServers
        self.nDepartures  = 0
        self.nSystem      = 0
        self.data         = {}   # times; issn : [arrival, service start, departure]

        self.arrival_f    = arrival_f
        self.service_f    = service_f
        self.AgentClass   = AgentClass
        self.collect_data = collect_data
        self.active_cap   = active_cap
        self.deactive_t   = deactive_t

        inftyAgent        = InftyAgent()
        self._arrivals    = [inftyAgent]    # A list of arriving agents.
        self._departures  = [inftyAgent]    # A list of departing agents.
        self._queue       = collections.deque()
        self._nArrivals   = 0
        self._oArrivals   = 0
        self._nTotal      = 0         # The number of agents scheduled to arrive + nSystem
        self._active      = False
        self._current_t   = 0         # The time of the last event.
        self._time        = infty     # The time of the next event.
        self._next_ct     = 0         # The next time an arrival from outside the network can arrive.
        self._black_cap   = 5.        # Used to help color edges and vertices.

        if isinstance(seed, numbers.Integral):
            np.random.seed(seed)

        if colors is not None:
            self.colors = colors
            for col in set(self._default_colors.keys()) - set(self.colors.keys()):
                self.colors[col] = self._default_colors[col]
        else:
            self.colors = self._default_colors


    @property
    def active(self):
        return self._active

    @property
    def time(self):
        return self._time

    @property
    def current_time(self):
        return self._current_t

    @property
    def nArrivals(self):
        return [self._nArrivals, self._oArrivals]


    def __repr__(self):
        my_str = ("QueueServer:{0}. Servers: {1}, queued: {2}, arrivals: {3}, "
                  "departures: {4}, next time: {5}")
        arg = (self.edge[2], self.nServers, len(self._queue), self.nArrivals,\
               self.nDepartures, round(self._time, 3))
        return my_str.format(*arg)

    def __lt__(a, b):
        return a._time < b._time

    def __gt__(a, b):
        return a._time > b._time

    def __eq__(a, b):
        return a._time == b._time

    def __le__(a, b):
        return a._time <= b._time

    def __ge__(a, b):
        return a._time >= b._time


    def at_capacity(self):
        """Returns whether the queue is at capacity or not.

        Returns
        -------
        bool
            Always returns ``False``, since the ``QueueServer`` class has
            infinite capacity.
        """
        return False


    def set_active(self):
        """Changes the ``active`` attribute to True. The queue now has arrivals
        arriving from outside the network.
        """
        self._active = True
        self._add_arrival()


    def set_inactive(self):
        """Changes the ``active`` attribute to False."""
        self._active = False


    def set_nServers(self, n):
        """Change the number of servers in the queue to ``n``.

        Parameters
        ----------
        n : int or :const:`numpy.infty`
            A positive integer (or ``np.infty``) to set the number of queues in
            the system to.

        Raises
        ------
        RuntimeError
            If ``n`` is not a positive integer or infinity then this error is
            raised.
        """
        if not isinstance(n, numbers.Integral) and n is not infty:
            the_str = "n must be an integer or infinity.\n{0}"
            raise TypeError(the_str.format(str(self)))
        elif n <= 0:
            the_str = "n must be a positive integer or infinity.\n{0}"
            raise ValueError( the_str.format(str(self)) )
        else:
            self.nServers = n


    def nQueued(self):
        """Returns the number of agents waiting in line to be served.

        Returns
        -------
        int
            The number of agents waiting in line to be served.
        """
        return len(self._queue)


    def fetch_data(self):
        """Fetches data from the queue.

        Returns
        -------
        data : :class:`~numpy.ndarray`
            A six column :class:`~numpy.ndarray` of all the data. The first,
            second, and third columns represent, respectively, the arrival,
            service start, and departure times of each ``Agent`` that has
            visited the queue. The fourth column identifies how many other
            agents were waiting to be serviced upon arrival, the fifth
            column identifies the number of agents in the system, and the sixth
            column specifies this queue by its edge index.
        """

        qdata = []
        for d in self.data.values():
            qdata.extend(d)

        dat = np.zeros((len(qdata), 6))
        if len(qdata) > 0:
            dat[:,:5] = np.array(qdata)
            dat[:, 5] = self.edge[2]

            dType = [
                ('a', float),
                ('s', float),
                ('d', float),
                ('q', float),
                ('n', float),
                ('id', float)
            ]
            dat = np.array([tuple(d) for d in dat], dtype=dType)
            dat = np.sort(dat, order='a')
            dat = np.array([tuple(d) for d in dat])

        return dat


    def _add_arrival(self, agent=None):
        if agent is not None:
            self._nTotal += 1
            heappush(self._arrivals, agent)
        else:
            if self._current_t >= self._next_ct:
                self._next_ct = self.arrival_f(self._current_t)

                if self._next_ct >= self.deactive_t:
                    self._active = False
                    return

                self._nTotal += 1
                new_agent = self.AgentClass((self.edge[2], self._oArrivals))
                new_agent._time = self._next_ct
                heappush(self._arrivals, new_agent)

                self._oArrivals += 1

                if self._oArrivals >= self.active_cap:
                    self._active = False

        if self._arrivals[0]._time < self._departures[0]._time:
            self._time = self._arrivals[0]._time


    def delay_service(self, t=None):
        """Adds an extra service time to the next departing agent's service
        time.

        Parameters
        ----------
        t : float (optional)
            Specifies when the departing time for the agent scheduled to depart
            next. If ``t`` is not given, then an additional service time is
            added to the next departing agent.
        """
        if len(self._departures) > 1:
            agent = heappop(self._departures)

            if t is None:
                agent._time = self.service_f(agent._time)
            else:
                agent._time = t

            heappush(self._departures, agent)

            if self._arrivals[0]._time < self._departures[0]._time:
                self._time = self._arrivals[0]._time
            else:
                self._time = self._departures[0]._time


    def next_event_description(self):
        """Returns an integer representing whether the next event is an arrival,
        a departure, or nothing.

        Returns
        -------
        out : int
            An integer representing whether the next event is an arrival or a
            departure: ``1`` corresponds to an arrival, ``2`` corresponds to a
            departure, and ``0`` corresponds to nothing scheduled to occur.
        """
        if self._departures[0]._time < self._arrivals[0]._time:
            return 2
        elif self._arrivals[0]._time < infty:
            return 1
        else:
            return 0


    def next_event(self):
        """Simulates the queue forward one event.

        Use :meth:`.simulate` instead.

        Returns
        -------
        out :
            If the next event is a departure then the departing agent is
            returned, otherwise nothing is returned.

        See Also
        --------
        :meth:`.simulate` : Simulates the queue forward.
        """
        if self._departures[0]._time < self._arrivals[0]._time :
            new_depart        = heappop(self._departures)
            self._current_t   = new_depart._time
            self._nTotal     -= 1
            self.nSystem     -= 1
            self.nDepartures += 1

            if self.collect_data and new_depart.issn in self.data :
                self.data[new_depart.issn][-1][2] = self._current_t

            if len(self._queue) > 0 :
                agent = self._queue.popleft()
                if self.collect_data and agent.issn in self.data :
                    self.data[agent.issn][-1][1] = self._current_t

                agent._time = self.service_f(self._current_t)
                agent.queue_action(self, 1)
                heappush(self._departures, agent)

            new_depart.queue_action(self, 2)

            if self._arrivals[0]._time < self._departures[0]._time :
                self._time = self._arrivals[0]._time
            else:
                self._time = self._departures[0]._time

            return new_depart

        elif self._arrivals[0]._time < infty :
            arrival = heappop(self._arrivals)
            self._current_t = arrival._time

            if self._active :
                self._add_arrival()

            self.nSystem    += 1
            self._nArrivals += 1

            if self.collect_data :
                b = 0 if self.nSystem <= self.nServers else 1
                if arrival.issn not in self.data :
                    self.data[arrival.issn] = [[arrival._time, 0, 0, len(self._queue)+b, self.nSystem]]
                else:
                    self.data[arrival.issn].append([arrival._time, 0, 0, len(self._queue)+b, self.nSystem])

            arrival.queue_action(self, 0)

            if self.nSystem <= self.nServers :
                if self.collect_data :
                    self.data[arrival.issn][-1][1] = arrival._time

                arrival._time = self.service_f(arrival._time)
                arrival.queue_action(self, 1)
                heappush(self._departures, arrival)
            else:
                self._queue.append(arrival)

            if self._arrivals[0]._time < self._departures[0]._time :
                self._time = self._arrivals[0]._time
            else:
                self._time = self._departures[0]._time


    def simulate(self, n=1, t=None, nA=None, nD=None):
        """This method simulates the queue forward for a specified amount of
        simulation time ``t``\, or for a specific number of events.

        Parameters
        ----------
        n : int (optional, the default is ``1``)
            The number of events to simulate. If ``t``, ``nA``, and ``nD`` are
            not given then this parameter is used.
        t : float (optional)
            The minimum amount of simulation time to simulate forward.
        nA : int (optional)
            Simulate until ``nA`` additional arrivals are observed.
        nD : int (optional)
            Simulate until ``nD`` additional departures are observed.

        Examples
        --------
        Before any simulations can take place the ``QueueServer`` must be
        activated:

        >>> import queueing_tool as qt
        >>> import numpy as np
        >>> rate = lambda t: 2 + 16 * np.sin(np.pi * t / 8)**2
        >>> arr = lambda t: qt.poisson_random_measure(rate, 18, t)
        >>> ser = lambda t: t + np.random.gamma(4, 0.1)
        >>> q = qt.QueueServer(5, arrival_f=arr, service_f=ser, seed=54)
        >>> q.set_active()

        To simulate 50000 events do the following:

        >>> q.simulate(50000)
        >>> nEvents = q.nArrivals[0] + q.nDepartures
        >>> nEvents
        50000

        To simulate a 75 units of simulation time, do the following:

        >>> t0 = q.time
        >>> q.simulate(t=75)
        >>> round(float(q.time - t0), 1)
        75.1
        >>> q.nArrivals[1] + q.nDepartures - nEvents
        1597

        To simulate forward until 1000 new departures are observed run:

        >>> nA0, nD0 = q.nArrivals[1], q.nDepartures
        >>> q.simulate(nD=1000)
        >>> q.nDepartures - nD0, q.nArrivals[1] - nA0
        (1000, 983)

        To simulate until 1000 new arrivals are observed run:

        >>> nA0, nD0 = q.nArrivals[1], q.nDepartures
        >>> q.simulate(nA=1000)
        >>> q.nDepartures - nD0, q.nArrivals[1] - nA0,
        (987, 1000)

        """
        if t is None and nD is None and nA is None:
            for k in range(n):
                tmp = self.next_event()
        elif t is not None:
            then = self._current_t + t
            while self._current_t < then and self._time < infty :
                tmp = self.next_event()
        elif nD is not None:
            nDepartures = self.nDepartures + nD
            while self.nDepartures < nDepartures and self._time < infty :
                tmp = self.next_event()
        elif nA is not None:
            nArrivals = self._oArrivals + nA
            while self._oArrivals < nArrivals and self._time < infty :
                tmp = self.next_event()


    def _current_color(self, which=0):
        """Returns a color for the queue.

        Parameters
        ----------
        which : int (optional, the default is ``0``)
            Specifies the type of color to return.

        Returns
        -------
        color : :class:`.list`
            Returns a RGBA color that is represented as a list with 4 entries
            where each entry can be any floating point number between 0 and 1.

            * If ``which`` is 1 then it returns the color of the edge as if it
              were a self loop. This is specified in
              ``colors['edge_loop_color']``\.
            * If ``which`` is 2 then it returns the color of the vertex pen color
              (defined as color/vertex_color in :func:`~graph_tool.draw.graph_draw`\).
              This is specified in ``colors['vertex_color']``\.
            * If ``which`` is anything else, then it returns the a shade of the
              edge that is proportional to the number of agents in the system
              -- which includes those being servered and those waiting to be
              served. More agents correspond to darker edge colors. Uses
              ``colors['vertex_fill_color']`` if the queue sits on a loop, and
              ``colors['edge_color']`` otherwise.
        """
        if which == 1 :
            color = self.colors['edge_loop_color']

        elif which == 2 :
            color = self.colors['vertex_color']

        else:
            nSy = self.nSystem
            cap = self.nServers
            tmp = 0.9 - min(nSy / (self._black_cap * cap), 0.9)

            if self.edge[0] == self.edge[1] :
                color    = [i * tmp / 0.9 for i in self.colors['vertex_fill_color']]
                color[3] = 1.0
            else:
                color    = [i * tmp / 0.9 for i in self.colors['edge_color']]
                color[3] = 0.7 - tmp / 1.8

        return color


    def clear(self):
        """Clears out the queue. Removes all arrivals, departures, and queued
        agents from the ``QueueServer``\, resets ``nArrivals``\, ``nDepartures``\,
        ``nSystem``\, and the clock to zero. It also clears any stored ``data``
        and the server is then set to inactive.
        """
        self.data        = {}
        self._nArrivals  = 0
        self._oArrivals  = 0
        self.nDepartures = 0
        self.nSystem     = 0
        self._nTotal     = 0
        self._current_t  = 0
        self._time       = infty
        self._next_ct    = 0
        self._active     = False
        self._queue      = collections.deque()
        inftyAgent       = InftyAgent()
        self._arrivals   = [inftyAgent]
        self._departures = [inftyAgent]


    def copy(self):
        """Returns a deep copy of itself."""
        return copy.deepcopy(self)


    def __deepcopy__(self, memo):
        new_server              = self.__class__()
        new_server.edge         = copy.copy(self.edge)
        new_server.nServers     = copy.copy(self.nServers)
        new_server.active_cap   = copy.copy(self.active_cap)
        new_server.deactive_t   = copy.copy(self.deactive_t)
        new_server.collect_data = copy.copy(self.collect_data)
        new_server.nDepartures  = copy.copy(self.nDepartures)
        new_server.nSystem      = copy.copy(self.nSystem)
        new_server._nTotal      = copy.copy(self._nTotal)
        new_server._active      = copy.copy(self._active)
        new_server._current_t   = copy.copy(self._current_t)
        new_server._time        = copy.copy(self._time)
        new_server._next_ct     = copy.copy(self._next_ct)
        new_server._nArrivals   = copy.copy(self._nArrivals)
        new_server._oArrivals   = copy.copy(self._oArrivals)
        new_server.data         = copy.deepcopy(self.data)
        new_server.colors       = copy.deepcopy(self.colors)
        new_server._queue       = copy.deepcopy(self._queue, memo)
        new_server._arrivals    = copy.deepcopy(self._arrivals, memo)
        new_server._departures  = copy.deepcopy(self._departures, memo)
        new_server.arrival_f    = self.arrival_f
        new_server.service_f    = self.service_f
        new_server.AgentClass   = self.AgentClass
        return new_server



class LossQueue(QueueServer):
    """A finite capacity queue.

    If the buffer is some finite value, then agents that arrive to the queue
    are lost.

    Parameters
    ----------
    qbuffer : int (optional, the default is 0)
        Specifies the length of the buffer i.e. specifies the maximum number
        of agents that can be waiting in line to be served.
    **kwargs
        Any :class:`~QueueServer` parameters.

    Attributes
    ----------
    nBlocked : int
        The number of times arriving agents have been blocked because the
        server was full.
    buffer : int
        Specifies the length of the buffer (i.e. specifies how long the
        size of the queue can be).

    Notes
    -----
    In `Kendall's notation`_\, this is a
    :math:`\\text{GI}_t/\\text{GI}_t/c/c+b/N/\\text{FIFO}` queue, where
    :math:`b` is the ``qbuffer``. If the default parameters are used then the
    instance is an :math:`\\text{M}/\\text{M}/1/1` queue.
    """

    _default_colors = {
        'edge_loop_color'  : [0, 0, 0, 0],
        'edge_color'       : [0.7, 0.7, 0.7, 0.5],
        'vertex_fill_color': [1.0, 1.0, 1.0, 1.0],
        'vertex_color'     : [0.133, 0.545, 0.133, 1.0]
    }

    def __init__(self, qbuffer=0, **kwargs):
        super(LossQueue, self).__init__(**kwargs)

        self.nBlocked   = 0
        self.buffer     = qbuffer
        self._black_cap = 1


    def __repr__(self):
        tmp = ("LossQueue:{0}. Servers: {1}, queued: {2}, arrivals: {3}, "
               "departures: {4}, next time: {5}")
        arg = (self.edge[2], self.nServers, len(self._queue), self.nArrivals,\
               self.nDepartures, round(self._time, 3))
        return tmp.format(*arg)


    def at_capacity(self):
        """Returns whether the queue is at capacity or not.

        Returns
        -------
        bool
            Returns whether the number of agents in the system --- the number
            of agents being serviced plus those waiting to be serviced --- the
            is greater than or equal to ``nServers + buffer``.
        """
        return self.nSystem >= self.nServers + self.buffer


    def next_event(self):
        """Simulates the queue forward one event.

        If the queue is at capacity, then the arriving agent is lost.

        Use :meth:`~QueueServer.simulate` for simulating instead.

        Returns
        -------
        out :
            If the next event is a departure then the departing agent is
            returned, otherwise nothing is returned.
        """
        if self._departures[0]._time < self._arrivals[0]._time :
            return super(LossQueue, self).next_event()
        elif self._arrivals[0]._time < infty :
            if self.nSystem < self.nServers + self.buffer :
                super(LossQueue, self).next_event()
            else:
                self.nBlocked += 1
                self._nTotal  -= 1

                arrival = heappop(self._arrivals)
                arrival.add_loss(self.edge)

                self._current_t = arrival._time

                if self._active :
                    self._add_arrival()

                if self.collect_data :
                    if arrival.issn in self.data :
                        self.data[arrival.issn].append([arrival._time, 0, 0, len(self._queue), self.nSystem])
                    else:
                        self.data[arrival.issn] = [[arrival._time, 0, 0, len(self._queue), self.nSystem]]

                if self._arrivals[0]._time < self._departures[0]._time :
                    self._time = self._arrivals[0]._time
                else:
                    self._time = self._departures[0]._time


    def clear(self):
        super(LossQueue, self).clear()
        self.nBlocked  = 0


    def __deepcopy__(self, memo):
        new_server = super(LossQueue, self).__deepcopy__(memo)
        new_server.nBlocked = copy.copy(self.nBlocked)
        new_server.buffer   = copy.copy(self.buffer)
        return new_server



class NullQueue(QueueServer):
    """A terminal queue.

    A queue that is used by the :class:`.QueueNetwork` class to represent
    agents leaving the network. It can collect the arrival times of every
    agent that arrives.

    Since the ``NullQueue`` is used to represent agents leaving the network,
    all agents that arrive to this queue are deleted.

    This class can collect data on arriving agents. With the exception of
    ``next_event_description``, ``nQueued``, and ``current_color``, all
    functions have been replaced with ``pass``. The methods
    :meth:`~QueueServer.next_event_description` and :meth:`~QueueServer.nQueued`
    will always return ``0``.
    """

    _default_colors = {
        'edge_loop_color'  : [0, 0, 0, 0],
        'edge_color'       : [0.7, 0.7, 0.7, 0.3],
        'vertex_fill_color': [1.0, 1.0, 1.0, 1.0],
        'vertex_color'     : [0.5, 0.5, 0.5, 0.5]
    }

    def __init__(self, *args, **kwargs):
        if 'edge' not in kwargs :
            kwargs['edge'] = (0, 0, 0, 0)

        super(NullQueue, self).__init__(**kwargs)
        self.nServers = 0

    def __repr__(self):
        return "NullQueue:{0}.".format(self.edge[2])

    def initialize(self, *args, **kwargs):
        pass

    def set_nServers(self, *args, **kwargs):
        pass

    def nQueued(self):
        return 0

    def _add_arrival(self, agent=None):
        if self.collect_data and agent is not None:
            if agent.issn not in self.data :
                self.data[agent.issn] = [[agent._time, 0, 0, 0, 0]]
            else:
                self.data[agent.issn].append([agent._time, 0, 0, 0, 0])

    def delay_service(self):
        pass

    def next_event_description(self):
        return 0

    def next_event(self):
        pass

    def _current_color(self, which=0):
        if which == 1 :
            color = self.colors['edge_loop_color']
        elif which == 2 :
            color = self.colors['vertex_color']
        else:
            if self.edge[0] == self.edge[1] :
                color = self.colors['vertex_fill_color']
            else:
                color = self.colors['edge_color']
        return color

    def clear(self):
        pass

    def __deepcopy__(self, memo):
        return super(NullQueue, self).__deepcopy__(memo)
