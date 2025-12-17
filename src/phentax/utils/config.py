# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
JAX configuration for phentax.

This module configures JAX for gravitational waveform computations,
enabling float64 precision by default and providing platform controls.

.. autosummary::
    configure_jax
    setup_logging
"""

import logging
import sys

import jax


def configure_jax(enable_x64: bool = True, platform: str | None = None) -> None:
    """
    Configure JAX for phentax.

    Parameters
    ----------
    enable_x64 : bool, default True
        Enable 64-bit floating point precision. Required for waveform accuracy.
    platform : str or None, default None
        Force JAX to use a specific platform ('cpu', 'gpu', 'tpu').
        If None, JAX auto-selects.
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    if platform is not None:
        jax.config.update("jax_platform_name", platform)


# Set up logging
# Set level
def setup_logging(name: str = "phentax", level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the phentax package.
    Parameters
    ----------
    name : str, default "phentax"
        Name of the logger.
    level : str, default "INFO"
        Logging level as a string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Only configure if this logger hasn't been configured yet
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger
