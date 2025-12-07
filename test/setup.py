from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "_hdist",
        ["_hdist.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
