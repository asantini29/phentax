# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
phentax: JAX implementation of IMRPhenomT(HM) gravitational waveform model.

This package provides differentiable, JIT-compiled gravitational waveform
generation for compact binary coalescences using JAX.
"""

from importlib.metadata import PackageNotFoundError, version

# Data structures
from . import core, utils, waveform

# from .utils.config import configure_jax

# # Configure JAX for float64 by default
# configure_jax()


__copyright__ = "2025, Alessandro Santini"
__author__ = "Alessandro Santini"
__email__ = "alessandro.santini@aei.mpg.de"

try:
    __version__ = version("phentax")
except PackageNotFoundError:
    __version__ = "unknown"
