# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Utility
================================

Utility functions for phentax.

Contains helper functions for mass ratios, spins, unit conversions,
and spin-weighted spherical harmonics, all implemented in pure JAX.

.. autosummary::
    :toctree: _autosummary

    m1ofeta
    m2ofeta
    qofeta
    eta_from_q
    check_equal_bhs
    chi_eff
    sTotR
    hz_to_mf
    mf_to_hz
    second_to_mass
    mass_to_second
    amp_nrto_si
    mode_to_lm
    mode_to_int
"""


import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .constants import C_SI, MPC_TO_M, MTSUN_SI

# =============================================================================
# Mass ratio utilities
# =============================================================================


@jax.jit
def m1ofeta(eta: float | Array, total_mass: float | Array = 1.0) -> float | Array:
    """
    Compute primary mass from symmetric mass ratio.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio in [0, 0.25].
    total_mass : float, default 1.0
        Total mass (m1 + m2). If 1.0, returns dimensionless mass fraction.

    Returns
    -------
    float
        Primary (larger) mass.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    return 0.5 * total_mass * (1.0 + delta)


@jax.jit
def m2ofeta(eta: float | Array, total_mass: float | Array = 1.0) -> float | Array:
    """
    Compute secondary mass from symmetric mass ratio.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio in [0, 0.25].
    total_mass : float | Array, default 1.0
        Total mass (m1 + m2). If 1.0, returns dimensionless mass fraction.

    Returns
    -------
    float | Array
        Secondary (smaller) mass.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    return 0.5 * total_mass * (1.0 - delta)


@jax.jit
def qofeta(eta: float | Array) -> float | Array:
    """
    Compute mass ratio q = m1/m2 >= 1 from symmetric mass ratio.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio in [0, 0.25].

    Returns
    -------
    float | Array
        Mass ratio q >= 1.
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    return m1 / m2


@jax.jit
def eta_from_q(q: float | Array) -> float | Array:
    """
    Compute symmetric mass ratio from mass ratio q = m1/m2 >= 1.

    Parameters
    ----------
    q : float | Array
        Mass ratio q >= 1.

    Returns
    -------
    float | Array
        Symmetric mass ratio eta.
    """
    return q / (1.0 + q) ** 2


@jax.jit
def check_equal_bhs(
    m1: float | Array,
    m2: float | Array,
    s1z: float | Array,
    s2z: float | Array,
) -> bool | Array:
    """
    Check if the binary black hole system is equal-mass and equal-spin.

    Parameters
    ----------
    m1, m2 : float | Array
        Masses of the two black holes.
    s1z, s2z : float | Array
        Dimensionless spin z-components of the two black holes.

    Returns
    -------
    bool | Array
        True if equal-mass and equal-spin, False otherwise.
    """
    mass_equal = jnp.isclose(m1, m2)
    spin_equal = jnp.isclose(s1z, s2z)
    return jnp.logical_and(mass_equal, spin_equal)


# =============================================================================
# Spin utilities
# =============================================================================


@jax.jit
def chi_eff(
    eta: float | Array,
    s1z: float | Array,
    s2z: float | Array,
) -> float | Array:
    """
    Compute effective spin parameter.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.
    s1z : float | Array
        Dimensionless spin of primary along orbital angular momentum.
    s2z : float | Array
        Dimensionless spin of secondary along orbital angular momentum.

    Returns
    -------
    float | Array
        Effective spin chi_eff = (m1*s1z + m2*s2z) / (m1 + m2).
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    return (m1 * s1z + m2 * s2z) / (m1 + m2)


@jax.jit
def sTotR(
    eta: float | Array,
    s1z: float | Array,
    s2z: float | Array,
) -> float | Array:
    """
    Compute reduced total spin parameter S = (m1^2*s1z + m2^2*s2z) / (m1^2 + m2^2).

    This is the spin combination used in PhenomT fits.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.
    s1z : float | Array
        Dimensionless spin of primary along orbital angular momentum.
    s2z : float | Array
        Dimensionless spin of secondary along orbital angular momentum.

    Returns
    -------
    float | Array
        Reduced total spin S.
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    m1_sq = m1 * m1
    m2_sq = m2 * m2
    return (m1_sq * s1z + m2_sq * s2z) / (m1_sq + m2_sq)


# =============================================================================
# Unit conversions
# =============================================================================


@jax.jit
def hz_to_mass(f_hz: float | Array, total_mass: float | Array) -> float | Array:
    """
    Convert frequency from Hz to dimensionless units (Mf).

    Parameters
    ----------
    f_hz : float | Array
        Frequency in Hz.
    total_mass : float | Array
        Total mass in solar masses.

    Returns
    -------
    float | Array
        Dimensionless frequency Mf.
    """
    return f_hz * total_mass * MTSUN_SI


@jax.jit
def mass_to_hz(mf: float | Array, total_mass: float | Array) -> float | Array:
    """
    Convert frequency from dimensionless units (Mf) to Hz.

    Parameters
    ----------
    mf : float | Array
        Dimensionless frequency Mf.
    total_mass : float | Array
        Total mass in solar masses.

    Returns
    -------
    float | Array
        Frequency in Hz.
    """
    return mf / (total_mass * MTSUN_SI)


@jax.jit
def second_to_mass(t_sec: float | Array, total_mass: float | Array) -> float | Array:
    """
    Convert time from seconds to dimensionless units (t/M).

    Parameters
    ----------
    t_sec : float | Array
        Time in seconds.
    total_mass : float | Array
        Total mass in solar masses.

    Returns
    -------
    float | Array
        Dimensionless time t/M.
    """
    return t_sec / (total_mass * MTSUN_SI)


@jax.jit
def mass_to_second(t_m: float | Array, total_mass: float | Array) -> float | Array:
    """
    Convert time from dimensionless units (t/M) to seconds.

    Parameters
    ----------
    t_m : float | Array
        Dimensionless time t/M.
    total_mass : float | Array
        Total mass in solar masses.

    Returns
    -------
    float | Array
        Time in seconds.
    """
    return t_m * total_mass * MTSUN_SI


@jax.jit
def amp_nr_to_si(
    h_nr: jnp.ndarray,
    distance_mpc: float | Array,
    total_mass: float | Array,
) -> jnp.ndarray:
    """
    Convert strain amplitude from NR units to SI units.

    Parameters
    ----------
    h_nr : array
        Strain in NR units (dimensionless, at unit distance).
    distance_mpc : float | Array
        Luminosity distance in Megaparsecs.
    total_mass : float | Array
        Total mass in solar masses.

    Returns
    -------
    array
        Strain in SI units.
    """
    # h_SI = h_NR * (G * M / c^2) / D = h_NR * M_in_meters / D_in_meters
    mass_in_seconds = total_mass * MTSUN_SI
    mass_in_meters = mass_in_seconds * C_SI
    distance_in_meters = distance_mpc * MPC_TO_M
    return h_nr * mass_in_meters / distance_in_meters


# =============================================================================
# Mode utilities
# =============================================================================


@jax.jit
def mode_to_lm(mode: Array) -> tuple[Array, Array]:
    """
    Convert integer mode key to (ell, m) mode indices.
    Assumes positive m modes.

    Parameters
    ----------
    mode : int | Array
        Integer key: 10*ell + |m| for positive m modes.

    Returns
    -------
    tuple of Array
        (ell, m) mode indices.
    """
    ell = mode // 10
    m = mode % 10

    return ell, m


def mode_to_int(ell: int, emm: int) -> int:
    """
    Convert (ell, m) mode indices to integer key.

    Parameters
    ----------
    ell : int
        Orbital angular momentum quantum number.
    emm : int
        Azimuthal quantum number.

    Returns
    -------
    int
        Integer key: 10*ell + |m| for positive m modes.
    """
    return 10 * ell + abs(emm)
