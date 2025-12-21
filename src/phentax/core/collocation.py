# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Collocation
============================
Collocation point methods for pseudo-PN coefficient computation.

This module implements the linear system solving to compute pseudo-PN coefficients
that augment the Post-Newtonian expansion to match numerical relativity fits.

Credits for the original implementations: Cecilio García Quirós

.. autosummary::
   :toctree: _autosummary
    OmegaPseudoPNCoeffs
    AmpPseudoPNCoeffs
    compute_omega_collocation_points
    compute_omega_pseudo_pn_coeffs
    compute_amp_collocation_points
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from . import fits

jax.config.update("jax_enable_x64", True)


class OmegaPseudoPNCoeffs(NamedTuple):
    """Pseudo-PN coefficients for omega inspiral ansatz (6 coefficients)."""

    c1: float | Array
    c2: float | Array
    c3: float | Array
    c4: float | Array
    c5: float | Array
    c6: float | Array


class AmpPseudoPNCoeffs(NamedTuple):
    """Pseudo-PN coefficients for amplitude inspiral ansatz (3 coefficients)."""

    c1: float | Array
    c2: float | Array
    c3: float | Array


# Theta collocation points for omega (dimensionless frequency parameter)
THETA_COLLOCATION_POINTS = jnp.array([0.33, 0.45, 0.55, 0.65, 0.75, 0.82])

# Time collocation points for amplitude (in units of M)
TIME_COLLOCATION_POINTS = jnp.array([-2000.0, -250.0, -150.0])


@jax.jit
def pn_ansatz_omega(
    theta: float | Array,
    omega_pn_coeffs: jnp.ndarray,
) -> Array:
    """
    TaylorT3 PN omega ansatz.

    Evaluates Eq. 6b from arXiv:2012.11923.

    Parameters
    ----------
    theta : float | Array
        Dimensionless frequency parameter theta = (-eta * t / 5)^(-1/8).
    omega_pn_coeffs : jnp.ndarray
        Array of 6 PN coefficients [omega1PN, omega1halfPN, omega2PN, omega2halfPN, omega3PN, omega3halfPN].

    Returns
    -------
    float | Array
        TaylorT3 omega value.
    """
    omega1PN = omega_pn_coeffs[0]
    omega1halfPN = omega_pn_coeffs[1]
    omega2PN = omega_pn_coeffs[2]
    omega2halfPN = omega_pn_coeffs[3]
    omega3PN = omega_pn_coeffs[4]
    omega3halfPN = omega_pn_coeffs[5]

    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    theta5 = theta3 * theta2
    theta6 = theta3 * theta3
    theta7 = theta4 * theta3

    logterm = 107.0 * jnp.log(theta) / 280.0
    fac = theta3 / 4.0

    return fac * (
        1.0
        + omega1PN * theta2
        + omega1halfPN * theta3
        + omega2PN * theta4
        + omega2halfPN * theta5
        + omega3PN * theta6
        + logterm * theta6
        + omega3halfPN * theta7
    )


def compute_omega_collocation_points(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    omega_pn_coeffs: jnp.ndarray,
) -> tuple[jnp.ndarray, float | Array, float | Array]:
    """
    Compute omega values at collocation points.

    The first point uses the PN ansatz at thetaini,
    the remaining 5 use inspiral_freq_cp fits.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.
    chi1, chi2 : float | Array
        Dimensionless spin z-components.
    omega_pn_coeffs : jnp.ndarray
        Array of 6 PN coefficients.

    Returns
    -------
    tuple
        (omega_values, tt0, tEarly) - omega values at 6 collocation points,
        t0 from fit, and tEarly value.
    """
    omega_values = jnp.zeros(6)

    # Get t0 from fit
    tt0 = fits.inspiral_t0_22(eta, chi1, chi2)

    # Compute tEarly from first theta collocation point
    theta0 = THETA_COLLOCATION_POINTS[0]
    tEarly = -5.0 / (eta * jnp.power(theta0, 8))

    # Compute thetaini for first point
    theta_ini = jnp.power(eta * (tt0 - tEarly) / 5.0, -0.125)

    # First point: use PN ansatz at thetaini
    omega_values = omega_values.at[0].set(pn_ansatz_omega(theta_ini, omega_pn_coeffs))

    # Remaining 5 points: use fits
    for i in range(1, 6):
        omega_fit = fits.inspiral_freq_cp(eta, chi1, chi2, i)
        omega_values = omega_values.at[i].set(omega_fit)

    return omega_values, tt0, tEarly


def compute_omega_pseudo_pn_coeffs(
    omega_pn_coeffs: jnp.ndarray,
    omega_collocation_values: jnp.ndarray,
) -> OmegaPseudoPNCoeffs:
    """
    Compute pseudo-PN coefficients for omega inspiral ansatz.

    Solves the linear system from Eq. 15 in arXiv:2012.11923:
        B[i] = 4/(theta[i]^3) * (omega_fit[i] - pn_omega[i])
        A[i,j] = theta[i]^(8+j)
        A * c = B

    Parameters
    ----------
    omega_pn_coeffs : jnp.ndarray
        Array of 6 PN coefficients.
    omega_collocation_values : jnp.ndarray
        Omega values at 6 collocation points (from fits or PN ansatz).

    Returns
    -------
    OmegaPseudoPNCoeffs
        The 6 pseudo-PN coefficients.
    """
    n_points = len(THETA_COLLOCATION_POINTS)
    theta = THETA_COLLOCATION_POINTS

    # Evaluate PN ansatz omega at the theta collocation points
    pn_omega = jnp.array(
        [pn_ansatz_omega(theta[i], omega_pn_coeffs) for i in range(n_points)]
    )

    # Construct RHS vector B
    # B = 4 / theta^3 * (omega_fit - pn_omega)
    B = 4.0 / (theta * theta * theta) * (omega_collocation_values - pn_omega)

    # Construct matrix A
    # A[i,j] = theta[i]^(8+j) where j goes from 0 to 5
    matrix = jnp.zeros((n_points, n_points))
    theta_power = jnp.power(theta, 8)

    for j in range(n_points):
        matrix = matrix.at[:, j].set(theta_power)
        theta_power = theta_power * theta

    # Solve linear system: A * c = B
    solution = jnp.linalg.solve(matrix, B)

    return OmegaPseudoPNCoeffs(
        c1=solution[0],
        c2=solution[1],
        c3=solution[2],
        c4=solution[3],
        c5=solution[4],
        c6=solution[5],
    )


def compute_amp_collocation_points(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    mode: int,
) -> jnp.ndarray:
    """
    Compute amplitude values at collocation points using fits.

    Uses inspiral_amp_cp fits at three time points: [-2000, -250, -150]M.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.
    chi1, chi2 : float | Array
        Dimensionless spin z-components.
    mode : int
        Mode identifier (22, 21, 33, 44, 55).

    Returns
    -------
    jnp.ndarray
        Amplitude values at the 3 collocation points.
    """
    amp_values = jnp.zeros(3)

    # Use fits for all 3 collocation points
    for i in range(3):
        amp_fit = fits.inspiral_amp_cp(eta, chi1, chi2, mode, i + 1)
        amp_values = amp_values.at[i].set(amp_fit)

    return amp_values
