from setuptools import setup, Extension
import pybind11

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
