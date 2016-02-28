import sys
import numpy

try :
    from setuptools import setup, Extension
except ImportError :
    from distutils.core import setup, Extension

try :
    from Cython.Distutils import build_ext
except ImportError :
    use_cython = False
else:
    use_cython = True


if sys.version_info[:2] < (2, 7) or (3, 0) <=  sys.version_info[:2] < (3, 2):
    raise RuntimeError('Python version 2.7 or >= 3.2 required.')

cmdclass    = {}
ext_modules = []

if use_cython:
    ext_modules.extend([
        Extension('queueing_tool.network.sorting', ['queueing_tool/network/sorting.pyx'], include_dirs=[numpy.get_include()]),
        Extension('queueing_tool.queues.choice', ['queueing_tool/queues/choice.pyx'], include_dirs=[numpy.get_include()])
    ])
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules.extend([
        Extension('queueing_tool.network.sorting', ['queueing_tool/network/sorting.c'], include_dirs=[numpy.get_include()]),
        Extension('queueing_tool.queues.choice', ['queueing_tool/queues/choice.c'], include_dirs=[numpy.get_include()])
    ])


with open('README', 'r') as a_file :
    long_description = a_file.read()

with open('VERSION', 'r') as a_file :
    version = a_file.read()[:-1]


setup(
    name='queueing_tool',
    version=version,
    description='Queueing network simulator',
    long_description=long_description,
    license='MIT',
    author='Daniel Jordon',
    author_email='dan@danjordon.com',
    url='https://github.com/djordon/queueing-tool',
    packages=['queueing_tool',
              'queueing_tool.network',
              'queueing_tool.queues',
              'queueing_tool.graph', 
              'queueing_tool.generation'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    keywords=['queueing theory', 'queueing', 'simulation', 'queueing simulator',
              'queueing network simulation', 'networks', 'queueing simulation'],
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics']
)
