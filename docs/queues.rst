Queues and agents
=================

.. automodule:: queueing_tool.queues
    :undoc-members:

Queues
------

    .. autoclass:: QueueServer(nServers=1, edge=(0,0,0), arrival_f=lambda t: t + np.random.exponential(1), service_f=lambda t: t + np.random.exponential(0.9), AgentClass=Agent, keep_data=False, active_cap=np.infty, deactive_t=np.infty, **kwargs)
        :members:
    .. autoclass:: LossQueue
        :members:
    .. autoclass:: InfoQueue
        :members:
    .. autoclass:: ResourceQueue(nServers=10, AgentClass=ResourceAgent, qbuffer=0, **kwargs)
        :members:
    .. autoclass:: NullQueue

Agents
------

    .. autoclass:: Agent
        :members:
    .. autoclass:: GreedyAgent
        :show-inheritance:
        :members:
    .. autoclass:: InfoAgent
        :show-inheritance:
        :members:
    .. autoclass:: ResourceAgent
        :show-inheritance:
        :members:

Queueing Functions
------------------

    .. autofunction:: poisson_random_measure
