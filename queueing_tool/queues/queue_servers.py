from .agents        import Agent, InftyAgent
from numpy.random   import uniform, exponential
from numpy          import infty
from heapq          import heappush, heappop

import numpy        as np
import collections
import copy


def poisson_random_measure(rate, rate_max, t) :
    """A function that returns the arrival time of the next arrival for a Poisson 
    random measure (a Poisson process is a special case of a Poisson random measure).

    Parameters
    ----------
    rate : function
        The *intensity function* for the measure, where ``rate(t)`` is the expected 
        arrival rate at time ``t``.
    rate_max : float
        The maximume value of the intensity function.
    t : float
        The start time from which to simulate the next arrival time.

    Returns
    -------
    out : float
        The time of the next arrival.

    Notes
    -----
    This function can only simulate processes that have bounded intensity functions.
    """
    t   = t + exponential(rate_max)
    while rate_max * uniform() > rate(t) :
        t   = t + exponential(rate_max)
    return t





class QueueServer :
    """The base queue-server class.

    Creates an instance of a :math:`{GI}_t/{GI}_t/n` queue. Each of the
    supplied parameters are set to a corresponding class variable/attribute.

    Each queue sits on an edge in a graph. When drawing the graph, the queue colors 
    the edges and the edge's target vertex. 

    Note that each parameter is assigned to an attribute of the same name.

    Parameters
    ----------
    nServers : int (optional, the default is 1)
        The number of servers servicing agents.
    edge : tuple (optional, the default is (0,0,0))
        A tuple that identifies (uniquely) which edge this queue lies on. The
        first slot of the tuple is the source edge, the second slot is the
        target/destination edge, and the last slot is the ``edge_index`` of that
        edge. This is automatically created when a 
        :class:`~queueing_tool.network.QueueNetwork` is instantiated.
    eType : int (optional, the default is 0)
        The type of queue this is. Used by :class:`~queueing_tool.network.QueueNetwork`
        when instantiating a network.
    arrival_f : function (optional, the default is ``lambda t: t + exponential(1)``)
        A function that returns the time of next arrival from outside the network.
        When this function is called, ``t`` is always taken to be the current time.
        Should never return any values less than ``t``, that is, ``arrival_f(t) >= t``.
    service_f : function (optional, the default is ``lambda t: t + exponential(0.9)``)
        A function that returns the time of an agent's service time completes. When
        This function is called, ``t`` is the time the agent is entering service.
        Should never return any values less than ``t``, that is, ``service_f(t) >= t``.
    AgentClass : class (optional, the default is :class:`~Agent` class)
        A class object for an :class:`~Agent` or any class object that has inherited the
        :class:`~Agent` class
       
    Attributes
    ----------
    nDepartures : int
        The number of departures from the queue.
    nSystem : int
        The number of agents in the entire queue (including those currently
        being served).
    nArrivals : list
        A list with two entries. The first slot is the total number of arrivals,
        while the second slot is the number of arrivals from the outside world.
    active : bool (the default is False)
        A variable that specifies whether the queue accepts arrivals from outside
        the network (queues always accept arrivals from inside the network).
    active_cap : int (the default is ``numpy.infty``)
        The maximum number of arrivals the queue will accept from outside the network.
    collect_data : bool (the default is False)
        A bool that defines whether the queue keeps :class:`~Agent` data
    data : dict
        Keeps track of each :class:`~Agent`'s arrival times, service start time, and
        departure times of all arrivals, and how many other agents were in the queue
        upon arrival. The keys are the :class:`~Agent`'s unique ``issn``,
        and the values is a list of lists. Each time an agent arrives at the queue it
        appends a list of that agent's arrival times, service start time, and departure
        times to end of the list.
    colors : dict
        A dictionary of the colors used when drawing the graph. The possible colors are
        ``edge_loop_color``: The default color of the edge if the edge is a loop.
        ``edge_color``: The normal color a non-loop edge.
        ``vertex_fill_color``: The normal fill color for a vertex. This colors the target
        vertex in the graph.
        ``vertex_pen_color``: The color of the pen for the of the target vertex.
        The defaults are:


    Notes
    -----
    Some defaults:

        >>> default_colors = {'edge_loop_color'     : [0, 0, 0, 0],
        ...                   'edge_color'   : [0.9, 0.9, 0.9, 0.5],
        ...                   'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
        ...                   'vertex_pen_color'    : [0.0, 0.5, 1.0, 1.0]}
    """
    def __init__(self, nServers=1, edge=(0,0,0), eType=0, arrival_f=lambda t: t + exponential(1),
                    service_f=lambda t: t + exponential(0.9), AgentClass=Agent, collect_data=False,
                    active_cap=infty, deactive_t=infty, colors=None, **kwargs) :

        self.edge         = edge
        self.eType        = eType
        self.nServers     = nServers
        self.nDepartures  = 0
        self.nSystem      = 0
        self.nArrivals    = [0, 0]
        self.active       = False
        self.data         = {}                # agent issn : [arrival t, service start t, departure t]

        self.arrival_f    = arrival_f
        self.service_f    = service_f
        self.AgentClass   = AgentClass
        self.collect_data = collect_data
        self.active_cap   = active_cap
        self.deactive_t   = deactive_t

        inftyAgent        = InftyAgent()
        self._arrivals    = [inftyAgent]
        self._departures  = [inftyAgent]
        self._queue       = collections.deque()
        self._nTotal      = 0
        self._current_t   = 0
        self._time        = infty
        self._next_ct     = 0
        self._black_cap   = 5                # Used to help color edges and vertices.

        default_colors   = {'edge_loop_color'   : [0, 0, 0, 0],
                            'edge_color'        : [0.9, 0.9, 0.9, 0.5],
                            'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
                            'vertex_pen_color'  : [0.0, 0.5, 1.0, 1.0]}

        if colors is not None :
            self.colors = colors
            for col in set(default_colors.keys()) - set(self.colors.keys()) :
                self.colors[col] = default_colors[col]
        else :
            self.colors = default_colors


    def __repr__(self) :
        tmp = "QueueServer: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self._queue), self.nArrivals, self.nDepartures, np.round(self._time, 3))
        return tmp

    def __lt__(a, b) :
        return a._time < b._time

    def __gt__(a, b) :
        return a._time > b._time

    def __eq__(a, b) :
        return a._time == b._time

    def __le__(a, b) :
        return a._time <= b._time

    def __ge__(a, b) :
        return a._time >= b._time


    def at_capacity(self) :
        """Returns whether the queue is at capacity or not.

        Returns
        -------
        out : bool
            Always returns False, since the ``QueueServer`` class has 
            infinite capacity.
        """
        return False


    def set_active(self) :
        """Changes the ``active`` attribute to True. The queue now has arrivals 
        comming from the outside world.
        """
        self.active = True
        self._add_arrival()


    def set_inactive(self) :
        """Changes the ``active`` attribute to False."""
        self.active = False


    def set_nServers(self, n) :
        """Change the number of servers in the queue to ``n``, where ``n``
        is positive integer. If ``n`` is not a positive integer than an
        Exception is raised.
        """
        if not isinstance(n, int) or n <= 0 :
            raise Exception("nServers must be positive, tried to set to %s.\n%s" % (n, str(self)) )
        else :
            self.nServers = n


    def nQueued(self) :
        """Returns the number of agents in the queue."""
        return len(self._queue)


    def _add_arrival(self, *args) :
        if len(args) > 0 :
            self._nTotal += 1
            heappush(self._arrivals, args[0])
        else : 
            if self._current_t >= self._next_ct :
                self._next_ct = self.arrival_f(self._current_t)

                if self._next_ct >= self.deactive_t :
                    self.active = False
                    return

                new_agent = self.AgentClass( (self.edge[2], self.nArrivals[1]) )
                new_agent.set_arrival(self._next_ct)
                heappush(self._arrivals, new_agent)

                self._nTotal      += 1
                self.nArrivals[1] += 1

                if self.nArrivals[1] >= self.active_cap :
                    self.active = False

        if self._arrivals[0]._time < self._departures[0]._time :
            self._time = self._arrivals[0]._time


    def _add_departure(self, agent, t) :
        self.nSystem       += 1
        self.nArrivals[0]  += 1

        if self.nSystem <= self.nServers :
            agent.set_departure(self.service_f(t))
            heappush(self._departures, agent)
        else :
            self._queue.append(agent)

        if self._arrivals[0]._time >= self._departures[0]._time :
            self._time = self._departures[0]._time


    def delay_service(self) :
        """Adds an extra service time to the next departing agents service time."""
        if len(self._departures) > 1 :
            agent = heappop(self._departures)
            agent.set_departure(self.service_f(agent._time))
            heappush(self._departures, agent)

            if self._arrivals[0]._time < self._departures[0]._time :
                self._time = self._arrivals[0]._time
            else :
                self._time = self._departures[0]._time


    def next_event_type(self) :
        """Returns an integer representing whether the next event is an arrival
        (returns 1), a departure (returns 2) or nothing (returns 0).
        """
        if self._arrivals[0]._time < self._departures[0]._time :
            return 1
        elif self._departures[0]._time < self._arrivals[0]._time :
            return 2
        elif self._departures[0]._time < infty :
            return 2
        else :
            return 0


    def next_event(self) :
        """Simulates the queue forward one event. 

        If there is a departure then an agent is returned.
        """
        if self._arrivals[0]._time < self._departures[0]._time :
            arrival = heappop(self._arrivals)
            self._current_t = arrival._time

            if self.active :
                self._add_arrival()

            self.nSystem       += 1
            self.nArrivals[0]  += 1

            if self.collect_data :
                if arrival.issn not in self.data :
                    self.data[arrival.issn] = [[arrival._time, 0, 0, len(self._queue)]]
                else :
                    self.data[arrival.issn].append([arrival._time, 0, 0, len(self._queue)])

            if self.nSystem <= self.nServers :
                if self.collect_data :
                    self.data[arrival.issn][-1][1] = arrival._time

                arrival.set_departure(self.service_f(arrival._time))
                heappush(self._departures, arrival)
            else :
                self._queue.append(arrival)

            if self._arrivals[0]._time < self._departures[0]._time :
                self._time = self._arrivals[0]._time
            else :
                self._time = self._departures[0]._time
                
        elif self._departures[0]._time < infty :
            new_depart        = heappop(self._departures)
            self._current_t   = new_depart._time
            self.nDepartures += 1
            self._nTotal     -= 1
            self.nSystem     -= 1

            if self.collect_data and new_depart.issn in self.data :
                self.data[new_depart.issn][-1][2] = self._current_t

            if len(self._queue) > 0 :
                agent = self._queue.popleft()
                if self.collect_data and agent.issn in self.data :
                    self.data[agent.issn][-1][1] = self._current_t

                agent.set_departure(self.service_f(self._current_t))
                heappush(self._departures, agent)

            new_depart.queue_action(self, 'departure')

            if self._arrivals[0]._time < self._departures[0]._time :
                self._time = self._arrivals[0]._time
            else :
                self._time = self._departures[0]._time

            return new_depart


    def current_color(self, which=0) :
        """Returns a color for the queue.

        Parameters
        ----------
        which : int (optional, the default is 0)
            Specifies the type of color to return.

        Returns
        -------
        color : list
            Returns a RGBA color. The color is represented as a list with 4 entries
            where each entry can be any floating point entry between 0 and 1.

            * If ``which`` is 1 the it returns the color of edges that are self loops. 
            * If ``which`` is 2 then it returns the color of the vertex pen color 
              (defined as color/vertex_color in :func:`~graph_tool.draw.graph_draw>`).
            * If ``which`` is anything else, then it returns the color a shade of the 
              edge that is proportional to the number of agents in the queue.
        """
        if which == 1 :
            color = self.colors['edge_loop_color']
  
        elif which == 2 :
            color = self.colors['vertex_pen_color']

        else :
            nSy = self.nSystem
            cap = self.nServers
            tmp = 0.9 - min(nSy / self._black_cap * cap, 0.9)

            if self.edge[0] == self.edge[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_fill_color'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_color'] ]
                color[3] = 0.7 - tmp / 1.8

        return color


    def clear(self) :
        """Clears out the queue. Removes all arrivals, departures, and queued agents from
        the ``QueueServer``, resets ``nArrivals``, ``nDepartures``, and ``nSystem`` to zero,
        and clears any stored ``data``.
        """
        self.data        = {}
        self.nArrivals   = [0, 0]
        self.nDepartures = 0
        self.nSystem     = 0
        self._nTotal     = 0
        self._current_t  = 0
        self._time       = infty
        self._next_ct    = 0
        self._queue      = collections.deque()
        inftyAgent       = InftyAgent()
        self._arrivals   = [inftyAgent]
        self._departures = [inftyAgent]


    def copy(self) :
        """Returns a deep copy of self."""
        return copy.deepcopy(self)


    def __deepcopy__(self, memo) :
        new_server              = self.__class__()
        new_server.edge         = copy.copy(self.edge)
        new_server.eType        = copy.copy(self.eType)
        new_server.nServers     = copy.copy(self.nServers)
        new_server.active       = copy.copy(self.active)
        new_server.active_cap   = copy.copy(self.active_cap)
        new_server.deactive_t   = copy.copy(self.deactive_t)
        new_server.collect_data = copy.copy(self.collect_data)
        new_server.nDepartures  = copy.copy(self.nDepartures)
        new_server.nSystem      = copy.copy(self.nSystem)
        new_server._nTotal      = copy.copy(self._nTotal)
        new_server._current_t   = copy.copy(self._current_t)
        new_server._time        = copy.copy(self._time)
        new_server._next_ct     = copy.copy(self._next_ct)
        new_server.data         = copy.deepcopy(self.data)
        new_server.nArrivals    = copy.deepcopy(self.nArrivals)
        new_server.colors       = copy.deepcopy(self.colors)
        new_server._queue       = copy.deepcopy(self._queue, memo)
        new_server._arrivals    = copy.deepcopy(self._arrivals, memo)
        new_server._departures  = copy.deepcopy(self._departures, memo)
        new_server.arrival_f    = self.arrival_f
        new_server.service_f    = self.service_f
        new_server.AgentClass   = self.AgentClass
        return new_server



class LossQueue(QueueServer) :
    """A finite capacity queue. Child of the :class:`~QueueServer` class.

    If the buffer is some finite value, then agents that arrive to the queue 
    are turned around and sent out of the queue. Essentially becomes a 
    ``QueueServer`` if the buffer/capacity is set to ``np.infty``.

    Parameters
    ----------
    qbuffer : int (optional, the default is 0)
        Specifies the length of the buffer (i.e. specifies how long the size
        of the queue can be).
    **kwargs :
        Parameters to pass to :class:`~QueueServer`.

    Attributes
    ----------
    nBlocked : int
        The number of times arriving agents have been blocked because the
        server was full.
    buffer : int
        Specifies the length of the buffer (i.e. specifies how long the 
        size of the queue can be).
    """

    def __init__(self, qbuffer=0, **kwargs) :
        default_colors  = { 'edge_loop_color'     : [0, 0, 0, 0],
                            'edge_color'   : [0.7, 0.7, 0.7, 0.5],
                            'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
                            'vertex_pen_color'    : [0.133, 0.545, 0.133, 1.0]} 

        if 'colors' in kwargs :
            for col in set(default_colors.keys()) - set(kwargs['colors'].keys()) :
                kwargs['colors'][col] = default_colors[col]
        else :
            kwargs['colors'] = default_colors

        QueueServer.__init__(self, **kwargs)

        self.nBlocked     = 0
        self.buffer       = qbuffer
        self._black_cap   = 1


    def __repr__(self) :
        tmp = "LossQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self._queue), self.nArrivals, self.nDepartures, np.round(self._time, 3))
        return tmp


    def at_capacity(self) :
        """Returns whether the queue is at capacity or not.

        Returns
        -------
        out : bool
            Returns whether the number of agents in the system is greater than 
            or equal to ``nServers + buffer``.
        """
        return self.nSystem >= self.nServers + self.buffer


    def next_event(self) :
        """Does something different.
        """
        if self._arrivals[0]._time < self._departures[0]._time :
            if self.nSystem < self.nServers + self.buffer :
                QueueServer.next_event(self)
            else :
                self.nBlocked      += 1
                self.nArrivals[0]  += 1
                self.nSystem       += 1

                new_agent = heappop(self._arrivals)
                new_agent.add_loss(self.edge)

                self._current_t = new_agent._time
                if self.active :
                    self._add_arrival()

                heappush(self._departures, new_agent)

                if self._arrivals[0]._time < self._departures[0]._time :
                    self._time = self._arrivals[0]._time
                else :
                    self._time = self._departures[0]._time

        elif self._departures[0]._time < self._arrivals[0]._time :
            return QueueServer.next_event(self)


    def clear(self) :
        QueueServer.clear(self)
        self.nBlocked  = 0


    def __deepcopy__(self, memo) :
        new_server          = QueueServer.__deepcopy__(self, memo)
        new_server.nBlocked = copy.copy(self.nBlocked)
        new_server.buffer   = copy.copy(self.buffer)
        return new_server



class NullQueue(QueueServer) :
    """A terminal queue.

    A queue that is used but the ``QueueNetwork`` class to represent 
    agents leaving the network.
    """
    def __init__(self, *args, **kwargs) :

        default_colors  = { 'edge_loop_color'     : [0, 0, 0, 0.5],
                            'edge_color'   : [0.7, 0.3, 0.3, 0.7],
                            'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
                            'vertex_pen_color'    : [0.0, 0.0, 0.0, 1.0]} 

        if 'colors' in kwargs :
            for col in set(default_colors.keys()) - set(kwargs['colors'].keys()) :
                kwargs['colors'][col] = default_colors[col]
        else :
            kwargs['colors'] = default_colors
        QueueServer.__init__(self, **kwargs)

    def __repr__(self) :
        return "NullQueue: %s." %  (self.edge[2])

    def initialize(self, *args, **kwargs) :
        pass

    def set_nServers(self, *args, **kwargs) :
        pass

    def nQueued(self) :
        return 0

    def _add_arrival(self, *args, **kwargs) :
        pass

    def _add_departure(self, *args, **kwargs) :
        pass

    def delay_service(self) :
        pass

    def next_event_type(self) :
        return 0

    def next_event(self) :
        pass

    def current_color(self, which='') :
        if which == 'loop' :
            color = self.colors['edge_loop_color']
        elif which == 'pen' :
            color = self.colors['vertex_pen_color']
        else :
            if self.edge[0] == self.edge[1] :
                color = self.colors['vertex_fill_color']
            else :
                color = self.colors['edge_color']
        return color

    def clear(self) :
        pass

    def __deepcopy__(self, memo) :
        return QueueServer.__deepcopy__(self, memo)
