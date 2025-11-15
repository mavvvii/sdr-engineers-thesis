"""Setup script for C++ extension modules (fastfft).

This module is a thin setuptools wrapper used to build the optional
`fastfft` extension that links against FFTW3.
"""

import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "fastfft",
        ["fastfft.cpp"],
        include_dirs=[pybind11.get_include()],
        libraries=["fftw3"],
        language="c++",
    )
]

setup(name="fastfft", version="0.1", ext_modules=ext_modules)
