Queueing networks
=================

.. automodule:: queueing_tool.network

   .. autoclass:: QueueNetwork

Simulation methods
------------------

      .. automethod:: QueueNetwork.initialize
      .. automethod:: QueueNetwork.set_transitions
      .. automethod:: QueueNetwork.simulate
      .. automethod:: QueueNetwork.transitions

Data methods
------------

      .. automethod:: QueueNetwork.clear_data
      .. automethod:: QueueNetwork.get_agent_data
      .. automethod:: QueueNetwork.get_queue_data
      .. automethod:: QueueNetwork.start_collecting_data
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
      .. automethod:: QueueNetwork.copy
      .. automethod:: QueueNetwork.next_event_description
      .. automethod:: QueueNetwork.reset_colors
