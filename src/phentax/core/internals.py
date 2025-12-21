# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Internals
============================
Internal data structures and coefficient computation for IMRPhenomT(HM).

Credits for the original implementations: Cecilio García Quirós

.. autosummary::
   :toctree: _autosummary

   WaveformParams
   compute_waveform_params
   compute_time_grid
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..utils.constants import MRSUN_SI, MTSUN_SI, PC_SI
from ..utils.utility import hz_to_mass, m1ofeta, m2ofeta, mass_to_hz, second_to_mass
from . import fits


# =============================================================================
# Data structures (pytree-compatible)
# =============================================================================
class WaveformParams(NamedTuple):
    """
    Physical parameters and derived quantities.
    These are intermediate quantities used throughout the waveform computation.

    Note that while the waveform interface assumes `m1` and `m2` are in solar masses,
    here `m1` and `m2` are the dimensionless masses to match the :class:`phenomxpy` interface.

    Parameters
    ----------
    m1 : float | Array
        Primary mass (dimensionless).
    m2 : float | Array
        Secondary mass (dimensionless).
    M : float | Array
        Total mass (dimensionless).
    mass1 : float | Array
        Primary mass (M_sun).
    mass2 : float | Array
        Secondary mass (M_sun).
    total_mass : float | Array
        Total mass (M_sun).
    eta : float | Array
        Symmetric mass ratio.
    delta : float | Array
        Mass difference ratio (m1-m2)/M.
    chi_eff : float | Array
        Effective spin.
    chi1 : float | Array
        Spin 1 z-component.
    chi2 : float | Array
        Spin 2 z-component.
    distance : float | Array
        Luminosity distance (Mpc).
    inclination : float | Array
        Inclination angle (radians).
    phi_ref : float | Array
        Reference phase (radians).
    psi : float | Array
        Polarization angle (radians).
    Mf : float | Array
        Final mass (dimensionless).
    af : float | Array
        Final spin (dimensionless).
    M_sec : float | Array
        Total mass in seconds.
    amp_factor : float | Array
        Amplitude prefactor.
    Mf_min : float | Array
        Minimum frequency (dimensionless).
    Mf_ref : float | Array
        Reference frequency (dimensionless).
    Mdelta_t : float | Array
        Delta t in dimensionless units.
    Mt_min : float | Array
        Minimum time in dimensionless units.
    Mt_ref : float | Array
        Reference time in dimensionless units.
    delta_t : float | Array
        Time step in seconds.
    t_min : float | Array
        Start time (seconds), default None.
    t_ref : float | Array
        Reference time (seconds), default None.
    t_low : float | Array
        if 0, use fits for bisection edge. else use value.
    """

    m1: float | Array  # Primary mass (dimensionless)
    m2: float | Array  # Secondary mass (dimensionless)
    M: float | Array  # Total mass (dimensionless)
    mass1: float | Array  # Primary mass (M_sun)
    mass2: float | Array  # Secondary mass (M_sun)
    total_mass: float | Array  # Total mass (M_sun)
    eta: float | Array  # Symmetric mass ratio
    delta: float | Array  # Mass difference ratio (m1-m2)/M
    chi_eff: float | Array  # Effective spin
    chi1: float | Array  # Spin 1 z-component
    chi2: float | Array  # Spin 2 z-component
    distance: float | Array  # Luminosity distance (Mpc)
    inclination: float | Array  # Inclination angle (radians)
    phi_ref: float | Array  # Reference phase (radians).
    psi: float | Array  # Polarization angle (radians).
    Mf: float | Array  # Final mass (dimensionless)
    af: float | Array  # Final spin (dimensionless)
    M_sec: float | Array  # Total mass in seconds
    amp_factor: float | Array  # Amplitude prefactor
    Mf_min: float | Array  # Minimum frequency (dimensionless)
    Mf_ref: float | Array  # Reference frequency (dimensionless)
    Mdelta_t: float | Array  # Delta t in dimensionless units
    Mt_min: float | Array  # Minimum time in dimensionless units
    Mt_ref: float | Array  # Reference time in dimensionless units
    delta_t: float | Array  # Time step in seconds
    t_min: float | Array  # Start time (seconds), default None
    t_ref: float | Array  # Reference time (seconds), default None
    t_low: float | Array  # if 0, use fits for bisection edge. else use value.
    atol: float | Array
    rtol: float | Array
    Mt_end: float | Array = 500  # end the waveform 500M after the peak
    length: int | Array = 10000  # number of time steps in the waveform generation


# =============================================================================
# Parameter validation and derived quantities
# =============================================================================


@jax.jit
def _compute_waveform_params(
    m1: float | Array,  # Primary mass (M_sun)
    m2: float | Array,  # Secondary mass (M_sun)
    s1z: float | Array,  # Primary spin z-component (dimensionless)
    s2z: float | Array,  # Secondary spin z-component (dimensionless)
    distance: float | Array,  # Luminosity distance (Mpc)
    inclination: float | Array,  # Inclination angle (radians)
    phi_ref: float | Array,  # Reference phase (radians)
    psi: float | Array,  # Polarization angle (radians)
    f_ref: float | Array,  # Reference frequency (Hz), or 0 for peak
    f_min: float | Array,  # Minimum frequency (Hz)
    delta_t: float | Array = 5.0,
    t_min: float | Array = jnp.nan,
    t_ref: float | Array = jnp.nan,
    t_low: float | Array = 0.0,
    atol: float | Array = 1e-12,
    rtol: float | Array = 1e-12,
) -> WaveformParams:
    """
    Compute derived quantities from physical parameters.

    Parameters
    ----------
    params : WaveformParams
        Input physical parameters.

    Returns
    -------
    DerivedParams
        Derived quantities for waveform computation.
    """
    # compute dimensionless masses

    # Ensure m1 >= m2
    mass1 = jnp.maximum(m1, m2)
    mass2 = jnp.minimum(m1, m2)

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    delta = (mass1 - mass2) / total_mass

    # dimensionless masses
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    M = m1 + m2

    # Effective spin
    chi_eff = (mass1 * s1z + mass2 * s2z) / total_mass
    # Final state
    Mf = fits.final_mass_2017(eta, s1z, s2z)
    af = fits.final_spin_2017(eta, s1z, s2z)
    # Mf = Mf_frac * total_mass

    # Total mass in seconds
    M_sec = total_mass * MTSUN_SI

    Mf_min = hz_to_mass(f_min, total_mass)
    Mf_ref = hz_to_mass(f_ref, total_mass)

    Mdelta_t = second_to_mass(delta_t, total_mass)

    Mt_ref = second_to_mass(t_ref, total_mass) if t_ref is not None else jnp.nan
    Mt_min = second_to_mass(t_min, total_mass) if t_min is not None else jnp.nan

    # Amplitude prefactor: M / D
    # Convert distance from Mpc to meters
    D_m = distance * 1e6 * PC_SI
    amp_factor = total_mass * MRSUN_SI / D_m

    return WaveformParams(
        m1=m1,
        m2=m2,
        M=M,
        mass1=mass1,
        mass2=mass2,
        total_mass=total_mass,
        eta=eta,
        delta=delta,
        chi_eff=chi_eff,
        chi1=s1z,
        chi2=s2z,
        distance=distance,
        inclination=inclination,
        phi_ref=phi_ref,
        psi=psi,
        Mf=Mf,
        af=af,
        Mf_min=Mf_min,
        Mf_ref=Mf_ref,
        Mdelta_t=Mdelta_t,
        Mt_min=Mt_min,
        Mt_ref=Mt_ref,
        M_sec=M_sec,
        amp_factor=amp_factor,
        delta_t=delta_t,
        t_min=t_min,
        t_ref=t_ref,
        t_low=t_low,
        atol=atol,
        rtol=rtol,
    )


def compute_waveform_params(
    m1: float | Array,  # Primary mass (M_sun)
    m2: float | Array,  # Secondary mass (M_sun)
    s1z: float | Array,  # Primary spin z-component (dimensionless)
    s2z: float | Array,  # Secondary spin z-component (dimensionless)
    distance: float | Array,  # Luminosity distance (Mpc)
    inclination: float | Array,  # Inclination angle (radians)
    phi_ref: float | Array,  # Reference phase (radians)
    psi: float | Array,  # Polarization angle (radians)
    f_ref: float | Array,  # Reference frequency (Hz), or 0 for peak
    f_min: float | Array,  # Minimum frequency (Hz)
    delta_t: float | Array = 5.0,
    t_min: float | Array = jnp.nan,
    t_ref: float | Array = jnp.nan,
    t_low: float | Array = 0.0,
    atol: float | Array = 1e-12,
    rtol: float | Array = 1e-12,
) -> WaveformParams:
    """
    Wrapper to compute derived waveform parameters.

    Parameters
    ----------
    m1 : float | Array
        Primary mass (M_sun).
    m2 : float | Array
        Secondary mass (M_sun).
    s1z : float | Array
        Primary spin z-component (dimensionless).
    s2z : float | Array
        Secondary spin z-component (dimensionless).
    distance : float | Array
        Luminosity distance (Mpc).
    inclination : float | Array
        Inclination angle (radians).
    phi_ref : float | Array
        Reference phase (radians).
    psi : float | Array
        Polarization angle (radians).
    f_ref : float | Array
        Reference frequency (Hz), or 0 for peak.
    f_min : float | Array
        Minimum frequency (Hz).
    delta_t : float | Array, default 5.0
        Time step (seconds).
    t_min : float | Array, default jnp.nan
        Start time (seconds).
    t_ref : float | Array, default jnp.nan
        Reference time (seconds).
    t_low : float | Array, default 0.0
        if 0, use fits for bisection edge. else use value.
    atol : float | Array, optional
        Absolute Bisection tolerance
    rtol : float | Array, optional
        Relative bisection tolerance
    Returns
    -------
    WaveformParams
        Derived waveform parameters.
    """

    if isinstance(m1, float):
        return _compute_waveform_params(
            m1,
            m2,
            s1z,
            s2z,
            distance,
            inclination,
            phi_ref,
            psi,
            f_ref,
            f_min,
            delta_t,
            t_min,
            t_ref,
            t_low,
            atol,
            rtol,
        )
    else:
        return jax.vmap(
            _compute_waveform_params,
            in_axes=(
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            out_axes=0,
        )(
            m1,
            m2,
            s1z,
            s2z,
            distance,
            inclination,
            phi_ref,
            psi,
            f_ref,
            f_min,
            delta_t,
            t_min,
            t_ref,
            t_low,
            atol,
            rtol,
        )


@jax.jit
def compute_wf_length_params(
    params: WaveformParams,
) -> WaveformParams:
    """
    Compute waveform length parameters in dimensionless units.

    Parameters
    ----------
    params : WaveformParams
        Input physical parameters.

    Returns
    -------
    WaveformParams
        Updated waveform parameters with length parameters.
    """

    length_negative = (jnp.ceil(-params.Mt_min / params.Mdelta_t)).astype(int)
    length_positive = (jnp.ceil(params.Mt_end / params.Mdelta_t)).astype(int)
    total_length = length_negative + length_positive + 1  # +1 for the zero time
    Mt_min = -length_negative * params.Mdelta_t

    return params._replace(Mt_min=Mt_min, length=total_length)
