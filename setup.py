from distutils.extension import Extension
import sys

try :
    from setuptools import setup
except ImportError :
    from distutils.core import setup

try :
    from Cython.Distutils import build_ext
except ImportError :
    use_cython = False
else:
    use_cython = True



if sys.version_info[0:2] < (3, 2):
    raise RuntimeError('Python version 3.2+ required.')

cmdclass    = {}
ext_modules = []

if use_cython:
    ext_modules.append(
        Extension('queueing_tool.network.sorting', ['queueing_tool/network/sorting.pyx'])
    )
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules.extend(
        Extension('queueing_tool.network.sorting', ['queueing_tool/network/sorting.c'])
    )


with open('README', 'r') as a_file :
    long_description = a_file.read()

with open('VERSION', 'r') as a_file :
    version = a_file.read()


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
              'queueing_tool.generation'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    keywords=['queueing', 'networks', 'simulation', 'network simulation'],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics']
)
