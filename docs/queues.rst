Queues and agents
=================

.. automodule:: queueing_tool.queues

Queues
------

   .. autoclass:: QueueServer(nServers=1, edge=(0,0,0), eType=1, arrival_f=lambda t: t + np.random.exponential(1), service_f=lambda t: t + np.random.exponential(0.9), AgentClass=Agent, keep_data=False, active_cap=np.infty, deactive_t=np.infty, **kwargs)
      :members:
   .. autoclass:: LossQueue
      :show-inheritance:
      :members:
   .. autoclass:: InfoQueue(net_size=1, AgentClass=InfoAgent, qbuffer=np.infty, **kwargs)
      :show-inheritance:
      :members:
   .. autoclass:: ResourceQueue(nServers=10, AgentClass=ResourceAgent, qbuffer=0, **kwargs)
      :show-inheritance:
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