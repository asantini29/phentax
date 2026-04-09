# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
phentax: JAX implementation of IMRPhenomT(HM) gravitational waveform model.

This package provides differentiable, JIT-compiled gravitational waveform
generation for compact binary coalescences using JAX.
"""

# from .utils.config import configure_jax

# # Configure JAX for float64 by default
# configure_jax()

# Data structures
from . import core, utils, waveform

__version__ = "0.1.1b3"
__copyright__ = "2025, Alessandro Santini"
__author__ = "Alessandro Santini"
