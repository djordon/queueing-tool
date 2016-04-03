import sys
import numpy

from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
    use_cython = True
except ImportError:
    use_cython = False


_version = sys.version_info[:2]
if _version < (2, 7) or (3, 0) <=  _version < (3, 3):
    raise RuntimeError('Python version 2.7 or >= 3.3 required.')

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules.extend([
        Extension(
            'queueing_tool.network.priority_queue',
            ['queueing_tool/network/priority_queue.pyx'],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            'queueing_tool.queues.choice',
            ['queueing_tool/queues/choice.pyx'],
            include_dirs=[numpy.get_include()]
        )
    ])
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules.extend([
        Extension(
            'queueing_tool.network.priority_queue',
            ['queueing_tool/network/priority_queue.c'],
            include_dirs=[numpy.get_include()]
        ),
        Extension(
            'queueing_tool.queues.choice',
            ['queueing_tool/queues/choice.c'],
            include_dirs=[numpy.get_include()]
        )
    ])


with open('README.rst', 'r') as a_file:
    long_description = a_file.read()

with open('VERSION', 'r') as a_file:
    version = a_file.read().strip()


classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics'
]

install_requires = ['networkx>=1.11', 'numpy>=1.10']

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

tests_requires = {'test': 'nose>=1.3.7'}

if _version[0] == 2:
    tests_requires.append('mock')

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
    extra_requires=tests_requires,
    test_suite='nose.collector',
    url='https://github.com/djordon/queueing-tool',
    version=version
)
