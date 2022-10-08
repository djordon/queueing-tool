
from setuptools import setup, Extension


ext_modules = [
    Extension(
        name='queueing_tool.network.priority_queue',
        sources=['queueing_tool/network/priority_queue.c'],
    ),
    Extension(
        name='queueing_tool.queues.choice',
        sources=['queueing_tool/queues/choice.c'],
    ),
]

setup(ext_modules=ext_modules)
