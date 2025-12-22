# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Spin-weighted spherical harmonics
=================================
Jax implementation of -2 spin-weighted spherical harmonics.
Enables jitted, vmapped and differentiable computation of :math:`_{-2}Y_{\\ell,m}`.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)


@jax.jit
def spin_weighted_spherical_harmonic(
    theta: float | Array,
    phi: float | Array,
    ell: int | Array = 2,
    emm: int | Array = 2,
) -> Array:
    """
    Compute spin-weighted spherical harmonic :math:`_{-2}Y_{\\ell,m}(\\theta, \phi)`.

    Optimized for JAX using explicit half-angle formulas to avoid branching.
    Valid for s=-2 and specific modes used in gravitational waves.

    Parameters
    ----------
    theta : float | Array
        Polar angle (inclination) in radians.
    phi : float | Array
        Azimuthal angle in radians.
    ell : int | Array, default 2
        Orbital angular momentum quantum number.
    emm : int | Array, default 2
        Azimuthal quantum number.

    Returns
    -------
    jnp.ndarray
        Value of the spin-weighted spherical harmonic.
    """
    # Constants
    sqrt_5_64pi = jnp.sqrt(5.0 / (64.0 * jnp.pi))
    sqrt_5_16pi = jnp.sqrt(5.0 / (16.0 * jnp.pi))
    sqrt_15_32pi = jnp.sqrt(15.0 / (32.0 * jnp.pi))
    sqrt_21_128pi = jnp.sqrt(21.0 / (128.0 * jnp.pi))
    sqrt_7_64pi = jnp.sqrt(7.0 / (64.0 * jnp.pi))
    sqrt_35_128pi = jnp.sqrt(35.0 / (128.0 * jnp.pi))
    sqrt_7_128pi = jnp.sqrt(7.0 / (128.0 * jnp.pi))
    sqrt_330_1024pi = jnp.sqrt(330.0 / (1024.0 * jnp.pi))

    # Precompute trigonometric functions
    c = jnp.cos(theta)
    c_2 = jnp.cos(theta / 2)
    s_2 = jnp.sin(theta / 2)
    s = jnp.sin(theta)

    # Define branches as lambdas to delay execution (only the selected one runs)
    def _zero():
        return jnp.zeros_like(theta)

    # Mode definitions
    def _22():
        return sqrt_5_64pi * (1 + c) ** 2

    def _2m2():
        return sqrt_5_64pi * (1 - c) ** 2

    def _21():
        return sqrt_5_16pi * s * (1 + c)

    def _2m1():
        return sqrt_5_16pi * s * (1 - c)

    def _20():
        return sqrt_15_32pi * s**2

    def _33():
        return -sqrt_21_128pi * (1 + c) ** 2 * s

    def _3m3():
        return sqrt_21_128pi * (1 - c) ** 2 * s

    def _32():
        return sqrt_7_64pi * (2 + 3 * c) * (1 + c) * s

    def _3m2():
        return sqrt_7_64pi * (2 - 3 * c) * (1 - c) * s

    def _31():
        return sqrt_35_128pi * (1 + c) * (1 - 3 * c + 4 * c**2) * s / (1 + c + 1e-30)

    def _3m1():
        return -sqrt_35_128pi * (1 - c) * (1 + 3 * c + 4 * c**2) * s / (1 - c + 1e-30)

    def _44():
        return (
            3.0 * jnp.sqrt(7.0 / jnp.pi) * jnp.power(c_2, 6.0) * jnp.power(s_2, 2.0)
        )  # 3 * sqrt_7_128pi * (1 + c) ** 2 * s**2

    def _4m4():
        return 3.0 * jnp.sqrt(7.0 / jnp.pi) * jnp.power(s_2, 6.0) * jnp.power(c_2, 2.0)

    def _55():
        return -jnp.sqrt(330.0 / jnp.pi) * jnp.power(c_2, 7.0) * jnp.power(s_2, 3.0)

    def _5m5():
        return jnp.sqrt(330.0 / jnp.pi) * jnp.power(c_2, 3.0) * jnp.power(s_2, 7.0)

    # Construct the switch table
    # Index = 10*ell + emm + 55
    # Max index needed is 10*5 + 5 + 55 = 110
    branches = [_zero] * 111

    branches[77] = _22  # 20 + 2 + 55
    branches[73] = _2m2  # 20 - 2 + 55
    branches[76] = _21  # 20 + 1 + 55
    branches[74] = _2m1  # 20 - 1 + 55
    branches[75] = _20  # 20 + 0 + 55

    branches[88] = _33  # 30 + 3 + 55
    branches[82] = _3m3  # 30 - 3 + 55
    branches[87] = _32  # 30 + 2 + 55
    branches[83] = _3m2  # 30 - 2 + 55
    branches[86] = _31  # 30 + 1 + 55
    branches[84] = _3m1  # 30 - 1 + 55

    branches[99] = _44  # 40 + 4 + 55
    branches[91] = _4m4  # 40 - 4 + 55

    branches[110] = _55  # 50 + 5 + 55
    branches[100] = _5m5  # 50 - 5 + 55

    # Calculate index
    idx = 10 * ell + emm + 55

    # Use lax.switch for efficient branching (computes only the active branch)
    # We clip the index to be safe, ensuring out-of-bounds maps to 0 (index 0 is _zero)
    idx = jnp.clip(idx, 0, 110).astype(int)

    angular = jax.lax.switch(idx, branches)

    return angular * jnp.exp(1j * emm * phi)


def spin_weighted_spherical_harmonic_all_modes(
    theta: float | Array,
    phi: float | Array,
    ells: Array,
    emms: Array,
) -> jnp.ndarray:
    """
    Vectorized computation of multiple spin-weighted spherical harmonics.

    Parameters
    ----------
    theta : float | Array
        Polar angle (inclination) in radians.
    phi : float | Array
        Azimuthal angle in radians.
    ells : Array
        Array of orbital angular momentum quantum numbers.
    emms : Array
        Array of azimuthal quantum numbers.

    Returns
    -------
    jnp.ndarray
        Array of spin-weighted spherical harmonics for each (ell, emm).
    """
    return jax.vmap(
        spin_weighted_spherical_harmonic, in_axes=(None, None, 0, 0), out_axes=1
    )(theta, phi, ells, emms)
