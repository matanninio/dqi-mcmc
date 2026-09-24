# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IBM Corporation

"""Setup script for building Cython extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Only build the optimal fast cache implementation
extensions = [
    Extension(
        "dqi_mcmc.api._elementary_poly_fast",
        ["dqi_mcmc/api/_elementary_poly_fast.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        annotate=True,
    ),
)
