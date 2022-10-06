import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


python_version = sys.version_info[:2]
if python_version < (2, 7) or (3, 0) <= python_version < (3, 3):
    raise RuntimeError('Python version 2.7 or >= 3.3 required.')

cmdclass = {'build_ext': build_ext}

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

with open('README.rst', 'r') as a_file:
    long_description = a_file.read()

with open('VERSION', 'r') as a_file:
    version = a_file.read().strip()

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics'
]

extras_require = {
    'plotting': ["matplotlib>=1.5.1"],
}

install_requires = ['networkx>=1.9', 'numpy>=1.9']

keywords = [
    'queueing theory',
    'queueing',
    'simulation',
    'queueing simulator',
    'queueing network simulation',
    'networks',
    'queueing simulation'
]

packages = [
    'queueing_tool',
    'queueing_tool.graph',
    'queueing_tool.network',
    'queueing_tool.queues'
]

tests_require = [
    'pytest>=3.0.2',
    'pytest-cov>=2.3.1',
    'pytest-sugar>=0.7.1',
]

if python_version[0] == 2:
    tests_require.append('mock')

setup(
    author='Daniel Jordon',
    author_email='dan.jordon@gmail.com',
    cmdclass=cmdclass,
    description='Queueing network simulator',
    ext_modules=ext_modules,
    classifiers=classifiers,
    install_requires=install_requires,
    keywords=keywords,
    long_description=long_description,
    license='MIT',
    name='queueing-tool',
    packages=packages,
    tests_require=tests_require,
    extras_require=extras_require,
    test_suite='nose.collector',
    url='https://github.com/djordon/queueing-tool',
    version=version
)
