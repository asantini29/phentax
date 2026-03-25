# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests comparing PN coefficients between phentax and phenomxpy.

These tests verify that the Post-Newtonian coefficients for omega (TaylorT3)
and amplitude match between the two implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

# Import phenomxpy
from phenomxpy.phenomt import internals as pxpy_internals

# Import phentax
from phentax.core import pn_coeffs as ptax_pn

# =============================================================================
# Helper functions
# =============================================================================


def assert_close(phentax_val, phenomxpy_val, name, rtol=1e-12, atol=1e-14):
    """Assert that phentax and phenomxpy values match within tolerance."""
    if np.isscalar(phenomxpy_val):
        err = abs(float(phentax_val) - float(phenomxpy_val))
        denom = abs(float(phenomxpy_val)) + 1e-30
        rel_err = err / denom
    else:
        err = np.max(np.abs(np.array(phentax_val) - np.array(phenomxpy_val)))
        denom = np.max(np.abs(phenomxpy_val)) + 1e-30
        rel_err = err / denom

    assert rel_err < rtol or err < atol, (
        f"{name}: phentax={phentax_val}, phenomxpy={phenomxpy_val}, "
        f"rel_err={rel_err:.2e}, abs_err={err:.2e}"
    )


# Test parameters
TEST_PARAMS = [
    # (eta, s1z, s2z)
    (0.25, 0.0, 0.0),
    (0.25, 0.5, 0.5),
    (0.25, 0.5, -0.5),
    (0.25, 0.9, 0.9),
    (0.222222, 0.0, 0.0),
    (0.222222, 0.3, 0.2),
    (0.16, 0.0, 0.0),
    (0.16, 0.6, 0.4),
    (0.0826446, 0.0, 0.0),
    (0.0826446, 0.8, 0.1),
]


# =============================================================================
# Test: Omega PN Coefficients (TaylorT3)
# =============================================================================


class TestOmegaPNCoeffs:
    """Compare omega PN coefficients for TaylorT3."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_omega_pn_coefficients(self, eta, s1z, s2z):
        """Test all omega PN coefficients match phenomxpy."""
        # Compute derived quantities
        delta = np.sqrt(1.0 - 4.0 * eta)
        total_mass = 100
        # m1, m2 as fractions of total mass
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)

        # Create phenomxpy pWFHM object to get PN coefficients
        pWF = pxpy_internals.pWFHM(
            mode=[2, 2],
            total_mass=total_mass,
            eta=eta,
            s1=s1z,
            s2=s2z,
            f_min=2.0,
            tlow_fit=False,
        )
        pPhase = pxpy_internals.pPhase(pWF)

        # Get phentax coefficients
        ptax_coeffs = ptax_pn.compute_omega_pn_coeffs(eta, s1z, s2z, delta, m1, m2)

        # Compare each coefficient
        assert_close(ptax_coeffs.omega1PN, pPhase.omega1PN, f"omega1PN(eta={eta})")
        assert_close(
            ptax_coeffs.omega1halfPN, pPhase.omega1halfPN, f"omega1halfPN(eta={eta})"
        )
        assert_close(ptax_coeffs.omega2PN, pPhase.omega2PN, f"omega2PN(eta={eta})")
        assert_close(
            ptax_coeffs.omega2halfPN, pPhase.omega2halfPN, f"omega2halfPN(eta={eta})"
        )
        assert_close(ptax_coeffs.omega3PN, pPhase.omega3PN, f"omega3PN(eta={eta})")
        assert_close(
            ptax_coeffs.omega3halfPN, pPhase.omega3halfPN, f"omega3halfPN(eta={eta})"
        )


# =============================================================================
# Test: Amplitude PN Coefficients (mode 22)
# =============================================================================


class TestAmpPNCoeffs22:
    """Compare amplitude PN coefficients for the 22 mode."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_amp_pn_coefficients_22(self, eta, s1z, s2z):
        """Test all amplitude PN coefficients for 22 mode match phenomxpy."""
        # Compute derived quantities
        delta = np.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)

        total_mass = 100

        # Create phenomxpy pWFHM and pAmp objects
        pWF = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pPhase = pxpy_internals.pPhase(pWF)
        pAmp = pxpy_internals.pAmp(pWF, pPhase)

        # Get phentax coefficients
        ptax_coeffs = ptax_pn.compute_amp_pn_coeffs_22(eta, s1z, s2z, delta, m1, m2)

        # Compare each coefficient
        assert_close(ptax_coeffs.ampN, pAmp.ampN, f"ampN_22(eta={eta})")
        assert_close(
            ptax_coeffs.amp0halfPNreal,
            pAmp.amp0halfPNreal,
            f"amp0halfPNreal_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp1PNreal, pAmp.amp1PNreal, f"amp1PNreal_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp1halfPNreal,
            pAmp.amp1halfPNreal,
            f"amp1halfPNreal_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp2PNreal, pAmp.amp2PNreal, f"amp2PNreal_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp2halfPNreal,
            pAmp.amp2halfPNreal,
            f"amp2halfPNreal_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp3PNreal, pAmp.amp3PNreal, f"amp3PNreal_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp3halfPNreal,
            pAmp.amp3halfPNreal,
            f"amp3halfPNreal_22(eta={eta})",
        )
        assert_close(ptax_coeffs.amplog, pAmp.amplog, f"amplog_22(eta={eta})")

        # Imaginary parts
        assert_close(
            ptax_coeffs.amp0halfPNimag,
            pAmp.amp0halfPNimag,
            f"amp0halfPNimag_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp1PNimag, pAmp.amp1PNimag, f"amp1PNimag_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp1halfPNimag,
            pAmp.amp1halfPNimag,
            f"amp1halfPNimag_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp2PNimag, pAmp.amp2PNimag, f"amp2PNimag_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp2halfPNimag,
            pAmp.amp2halfPNimag,
            f"amp2halfPNimag_22(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp3PNimag, pAmp.amp3PNimag, f"amp3PNimag_22(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp3halfPNimag,
            pAmp.amp3halfPNimag,
            f"amp3halfPNimag_22(eta={eta})",
        )

        # Prefactor
        assert_close(ptax_coeffs.fac0, pAmp.fac0, f"fac0_22(eta={eta})")


# =============================================================================
# Test: Amplitude PN Coefficients (higher modes)
# =============================================================================


class TestAmpPNCoeffsHigherModes:
    """Compare amplitude PN coefficients for higher modes."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS[:5])  # Fewer params to speed up
    def test_amp_pn_coefficients_21(self, eta, s1z, s2z):
        """Test amplitude PN coefficients for 21 mode."""
        delta = np.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)

        total_mass = 100
        # Create phenomxpy objects
        pWF22 = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pPhase22 = pxpy_internals.pPhase(pWF22)

        pWF21 = pxpy_internals.pWFHM(
            mode=[2, 1], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pAmp21 = pxpy_internals.pAmp(pWF21, pPhase22)

        ptax_coeffs = ptax_pn.compute_amp_pn_coeffs_21(eta, s1z, s2z, delta, m1, m2)

        assert_close(ptax_coeffs.ampN, pAmp21.ampN, f"ampN_21(eta={eta})")
        assert_close(
            ptax_coeffs.amp0halfPNreal,
            pAmp21.amp0halfPNreal,
            f"amp0halfPNreal_21(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp1PNreal, pAmp21.amp1PNreal, f"amp1PNreal_21(eta={eta})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS[:5])
    def test_amp_pn_coefficients_33(self, eta, s1z, s2z):
        """Test amplitude PN coefficients for 33 mode."""
        delta = np.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        total_mass = 100

        pWF22 = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pPhase22 = pxpy_internals.pPhase(pWF22)

        pWF33 = pxpy_internals.pWFHM(
            mode=[3, 3], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pAmp33 = pxpy_internals.pAmp(pWF33, pPhase22)

        ptax_coeffs = ptax_pn.compute_amp_pn_coeffs_33(eta, s1z, s2z, delta, m1, m2)

        assert_close(ptax_coeffs.ampN, pAmp33.ampN, f"ampN_33(eta={eta})")
        assert_close(
            ptax_coeffs.amp0halfPNreal,
            pAmp33.amp0halfPNreal,
            f"amp0halfPNreal_33(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp1halfPNreal,
            pAmp33.amp1halfPNreal,
            f"amp1halfPNreal_33(eta={eta})",
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS[:5])
    def test_amp_pn_coefficients_44(self, eta, s1z, s2z):
        """Test amplitude PN coefficients for 44 mode."""
        delta = np.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)

        total_mass = 100

        pWF22 = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pPhase22 = pxpy_internals.pPhase(pWF22)

        pWF44 = pxpy_internals.pWFHM(
            mode=[4, 4], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pAmp44 = pxpy_internals.pAmp(pWF44, pPhase22)

        ptax_coeffs = ptax_pn.compute_amp_pn_coeffs_44(eta, s1z, s2z, delta, m1, m2)

        assert_close(ptax_coeffs.ampN, pAmp44.ampN, f"ampN_44(eta={eta})")
        assert_close(
            ptax_coeffs.amp1PNreal, pAmp44.amp1PNreal, f"amp1PNreal_44(eta={eta})"
        )
        assert_close(
            ptax_coeffs.amp2PNreal, pAmp44.amp2PNreal, f"amp2PNreal_44(eta={eta})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS[:5])
    def test_amp_pn_coefficients_55(self, eta, s1z, s2z):
        """Test amplitude PN coefficients for 55 mode."""
        delta = np.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        total_mass = 100

        pWF22 = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pPhase22 = pxpy_internals.pPhase(pWF22)

        pWF55 = pxpy_internals.pWFHM(
            mode=[5, 5], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pAmp55 = pxpy_internals.pAmp(pWF55, pPhase22)

        ptax_coeffs = ptax_pn.compute_amp_pn_coeffs_55(eta, s1z, s2z, delta, m1, m2)

        assert_close(ptax_coeffs.ampN, pAmp55.ampN, f"ampN_55(eta={eta})")
        assert_close(
            ptax_coeffs.amp1halfPNreal,
            pAmp55.amp1halfPNreal,
            f"amp1halfPNreal_55(eta={eta})",
        )
        assert_close(
            ptax_coeffs.amp2halfPNreal,
            pAmp55.amp2halfPNreal,
            f"amp2halfPNreal_55(eta={eta})",
        )


# =============================================================================
# Test: Powers of 5
# =============================================================================


class TestPowersOf5:
    """Test the powers of 5 array used in phase computation."""

    def test_powers_of_5(self):
        """Test powers of 5 array matches phenomxpy."""
        # Create phenomxpy pPhase to get powers of 5
        pWF = pxpy_internals.pWFHM(
            mode=[2, 2], total_mass=100, eta=0.25, s1=0.0, s2=0.0, f_min=20.0
        )
        pPhase = pxpy_internals.pPhase(pWF)

        for i, (ptax, pxpy) in enumerate(zip(ptax_pn.POWERS_OF_5, pPhase.powers_of_5)):
            assert_close(float(ptax), float(pxpy), f"powers_of_5[{i}]")
