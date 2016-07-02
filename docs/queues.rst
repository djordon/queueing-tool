Queues and agents
=================

.. automodule:: queueing_tool.queues

Queues
------

   .. autoclass:: QueueServer(num_servers=1, edge=(0,0,0,1), arrival_f=lambda t: t + np.random.exponential(1), service_f=lambda t: t + np.random.exponential(0.9), AgentFactory=Agent, keep_data=False, active_cap=np.infty, deactive_t=np.infty, coloring_sensitivity=2, **kwargs)
      :members:
   .. autoclass:: LossQueue
      :show-inheritance:
      :members:
   .. autoclass:: InfoQueue(net_size=1, AgentFactory=InfoAgent, qbuffer=np.infty, **kwargs)
      :show-inheritance:
      :members:
   .. autoclass:: ResourceQueue(num_servers=10, AgentFactory=ResourceAgent, qbuffer=0, **kwargs)
      :show-inheritance:
      :members:
   .. autoclass:: NullQueue
      :show-inheritance:

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
