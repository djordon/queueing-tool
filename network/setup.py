from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Sorting functions',
    ext_modules = cythonize("sorting.pyx"),
)

