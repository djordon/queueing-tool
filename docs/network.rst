Queueing networks
=================

.. automodule:: queueing_tool.network

   .. autoclass:: QueueNetwork

Simulation methods
------------------

      .. automethod:: QueueNetwork.initialize
      .. automethod:: QueueNetwork.set_routing_probs
      .. automethod:: QueueNetwork.simulate

Data methods
------------

      .. automethod:: QueueNetwork.collect_data
      .. automethod:: QueueNetwork.data_agents
      .. automethod:: QueueNetwork.data_queues
      .. automethod:: QueueNetwork.stop_collecting_data

Graph drawing methods
---------------------

      .. automethod:: QueueNetwork.animate
      .. automethod:: QueueNetwork.draw
      .. automethod:: QueueNetwork.show_active
      .. automethod:: QueueNetwork.show_type

Generic methods
---------------

      .. automethod:: QueueNetwork.clear
      .. automethod:: QueueNetwork.clear_data
      .. automethod:: QueueNetwork.copy
      .. automethod:: QueueNetwork.next_event_description
      .. automethod:: QueueNetwork.reset_colors
