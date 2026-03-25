# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests comparing pseudo-PN coefficients between phentax and phenomxpy.

Verifies that the collocation point evaluation and linear system solving
produce identical results to phenomxpy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

# Import phenomxpy for reference
from phenomxpy.phenomt.internals import pPhase, pWF, pWFHM

# Import phentax modules
from phentax.core import collocation, pn_coeffs


def assert_close(actual, expected, rtol=1e-10, atol=1e-12, name="value"):
    """Assert that actual is close to expected within tolerances."""
    abs_diff = abs(float(actual) - float(expected))
    rel_diff = abs_diff / abs(float(expected)) if float(expected) != 0 else abs_diff
    if not (abs_diff < atol or rel_diff < rtol):
        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Abs diff: {abs_diff}\n"
            f"  Rel diff: {rel_diff}"
        )


class TestPNAnsatzOmega:
    """Test that the PN ansatz omega matches phenomxpy."""

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
        ],
    )
    def test_pn_ansatz_omega(self, eta, s1z, s2z):
        """Test that pn_ansatz_omega matches phenomxpy.pPhase.pn_ansatz_omega."""
        # Create phenomxpy pWF and pPhase
        pwf = pWFHM(eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0)
        pphase = pPhase(pwf)

        # Get omega PN coefficients from phentax
        delta = jnp.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        omega_pn = pn_coeffs.compute_omega_pn_coeffs(eta, s1z, s2z, delta, m1, m2)
        omega_pn_array = jnp.array(
            [
                omega_pn.omega1PN,
                omega_pn.omega1halfPN,
                omega_pn.omega2PN,
                omega_pn.omega2halfPN,
                omega_pn.omega3PN,
                omega_pn.omega3halfPN,
            ]
        )

        # Test at several theta values
        theta_values = [0.33, 0.45, 0.55, 0.65, 0.75, 0.82]
        for theta in theta_values:
            expected = pphase.pn_ansatz_omega(theta)
            actual = collocation.pn_ansatz_omega(theta, omega_pn_array)
            assert_close(actual, expected, name=f"pn_ansatz_omega(theta={theta})")


class TestOmegaCollocationPoints:
    """Test omega collocation point values match phenomxpy."""

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
            (0.15, 0.8, 0.0),
        ],
    )
    def test_omega_collocation_points(self, eta, s1z, s2z):
        """Test that omega collocation point values match phenomxpy."""
        # Create phenomxpy objects
        pwf = pWFHM(eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0)
        pphase = pPhase(pwf)

        # Get omega PN coefficients
        delta = jnp.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        omega_pn = pn_coeffs.compute_omega_pn_coeffs(eta, s1z, s2z, delta, m1, m2)
        omega_pn_array = jnp.array(
            [
                omega_pn.omega1PN,
                omega_pn.omega1halfPN,
                omega_pn.omega2PN,
                omega_pn.omega2halfPN,
                omega_pn.omega3PN,
                omega_pn.omega3halfPN,
            ]
        )

        # Compute collocation points with phentax
        omega_values, tt0, tEarly = collocation.compute_omega_collocation_points(
            eta, s1z, s2z, omega_pn_array
        )

        # Compare tt0
        assert_close(tt0, pphase.tt0, name="tt0")

        # Compare tEarly
        assert_close(tEarly, pphase.tEarly, name="tEarly")

        # Compare all 6 omega collocation point values
        for i in range(6):
            expected = pphase.inspiral_collocation_points[i, 1]
            assert_close(
                omega_values[i],
                expected,
                name=f"omega_cp[{i}] (eta={eta}, s1z={s1z}, s2z={s2z})",
            )


class TestOmegaPseudoPNCoeffs:
    """Test omega pseudo-PN coefficients against phenomxpy."""

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.25, 0.5, -0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
            (0.15, 0.8, 0.0),
            (0.1, 0.5, 0.3),
        ],
    )
    def test_omega_pseudo_pn_coeffs(self, eta, s1z, s2z):
        """Test that omega pseudo-PN coefficients match phenomxpy."""
        # Create phenomxpy objects
        pwf = pWFHM(eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0)
        pphase = pPhase(pwf)

        # Get expected coefficients from phenomxpy
        expected_c1 = pphase.omegaInspC1
        expected_c2 = pphase.omegaInspC2
        expected_c3 = pphase.omegaInspC3
        expected_c4 = pphase.omegaInspC4
        expected_c5 = pphase.omegaInspC5
        expected_c6 = pphase.omegaInspC6

        # Compute using phentax
        delta = jnp.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)

        # Get PN coefficients
        omega_pn = pn_coeffs.compute_omega_pn_coeffs(eta, s1z, s2z, delta, m1, m2)
        omega_pn_array = jnp.array(
            [
                omega_pn.omega1PN,
                omega_pn.omega1halfPN,
                omega_pn.omega2PN,
                omega_pn.omega2halfPN,
                omega_pn.omega3PN,
                omega_pn.omega3halfPN,
            ]
        )

        # Get collocation point values
        omega_values, tt0, tEarly = collocation.compute_omega_collocation_points(
            eta, s1z, s2z, omega_pn_array
        )

        # Compute pseudo-PN coefficients
        pseudo_pn = collocation.compute_omega_pseudo_pn_coeffs(
            omega_pn_array, omega_values
        )

        # Compare coefficients
        assert_close(pseudo_pn.c1, expected_c1, name=f"omegaInspC1")
        assert_close(pseudo_pn.c2, expected_c2, name=f"omegaInspC2")
        assert_close(pseudo_pn.c3, expected_c3, name=f"omegaInspC3")
        assert_close(pseudo_pn.c4, expected_c4, name=f"omegaInspC4")
        assert_close(pseudo_pn.c5, expected_c5, name=f"omegaInspC5")
        assert_close(pseudo_pn.c6, expected_c6, name=f"omegaInspC6")


class TestAmpCollocationPoints:
    """Test amplitude collocation point values match phenomxpy."""

    @pytest.mark.parametrize(
        "mode,eta,s1z,s2z",
        [
            (22, 0.25, 0.0, 0.0),
            (22, 0.25, 0.5, 0.5),
            (22, 0.24, 0.3, 0.1),
            (21, 0.25, 0.0, 0.0),
            (21, 0.24, 0.3, 0.1),
            (33, 0.25, 0.0, 0.0),
            (33, 0.20, 0.6, 0.2),
            (44, 0.25, 0.0, 0.0),
            (55, 0.25, 0.0, 0.0),
        ],
    )
    def test_amp_collocation_points(self, mode, eta, s1z, s2z):
        """Test that amplitude collocation point values match phenomxpy."""
        from phenomxpy.phenomt.internals import pAmp

        # Create phenomxpy objects
        ell = mode // 10
        m = mode % 10

        if mode == 22:
            pwf = pWFHM(
                eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0, mode=[ell, m]
            )
            pphase = pPhase(pwf)
            pamp = pAmp(pwf, pphase)
        else:
            # Higher modes need the 22 phase
            pwf_22 = pWFHM(
                eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0, mode=[2, 2]
            )
            pphase_22 = pPhase(pwf_22)
            pwf = pWFHM(
                eta=eta, total_mass=100, s1=s1z, s2=s2z, f_min=20.0, mode=[ell, m]
            )
            pphase = pPhase(pwf, pPhase22=pphase_22)
            pamp = pAmp(pwf, pphase)

        # Compute collocation points with phentax
        amp_values = collocation.compute_amp_collocation_points(eta, s1z, s2z, mode)

        # Compare all 3 amplitude collocation point values
        for i in range(3):
            expected = pamp.inspiral_collocation_points[i, 1]
            assert_close(amp_values[i], expected, name=f"amp_cp[{i}] mode {mode}")
