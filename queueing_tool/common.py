import collections


EdgeID = collections.namedtuple(
    typename='EdgeID',
    field_names=['source', 'target', 'edge_index', 'edge_index']
)

AgentID = collections.namedtuple(
    typename='AgentID',
    field_names=['edge_index', 'agent_qid']
)
