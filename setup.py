from distutils.core import setup
from distutils.extension import Extension
import sys

# With the help of this stackoverflow.com answer:
#   http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code

if sys.version_info[0:2] < (3, 2):
    raise RuntimeError('Python version 3.2+ required.')

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

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
    description = a_file.read()

with open('VERSION', 'r') as a_file :
    version = a_file.read()

setup(
    name='queueing_tool',
    version=version,
    description='Queueing network simulator',
    long_description=description,
    license='MIT',
    author='Daniel Jordon',
    author_email='dan.jordon@gmail.com',
    url='https://github.com/djordon/queueing-tool',
    packages=['queueing_tool', 
              'queueing_tool.network', 
              'queueing_tool.queues', 
              'queueing_tool.generation'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Researchers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering :: Mathematics']
)
