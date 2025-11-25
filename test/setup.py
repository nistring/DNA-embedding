from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "floyd_warshall",
        ["floyd_warshall.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "_hdist",
        ["_hdist.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
