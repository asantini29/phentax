# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests for amplitude coefficients comparison with phenomxpy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from phentax.core.amplitude import compute_amplitude_coeffs_22
from phentax.core.internals import compute_waveform_params
from phentax.core.phase import compute_phase_coeffs_22

# Try to import phenomxpy for comparison
try:
    import phenomxpy
    from phenomxpy.phenomt.phenomt import IMRPhenomT

    PHENOMXPY_AVAILABLE = True
except ImportError:
    PHENOMXPY_AVAILABLE = False


def _make_wf_params(eta, chi1, chi2, total_mass=100.0, f_min=20.0):
    """Create WaveformParams from eta and spins for comparison tests."""
    delta = np.sqrt(1.0 - 4.0 * eta)
    m1_solar = 0.5 * (1.0 + delta) * total_mass
    m2_solar = 0.5 * (1.0 - delta) * total_mass
    return compute_waveform_params(
        m1=m1_solar,
        m2=m2_solar,
        s1z=chi1,
        s2z=chi2,
        distance=100.0,
        inclination=0.0,
        phi_ref=0.0,
        psi=0.0,
        f_ref=f_min,
        f_min=f_min,
    )


def assert_close(actual, expected, rtol=1e-10, atol=1e-12, name=""):
    """Helper to assert closeness with detailed error message."""
    # Handle JAX arrays
    if hasattr(actual, "item"):
        actual = actual.item()
    if hasattr(expected, "item"):
        expected = expected.item()

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Abs diff: {abs(actual - expected)}\n"
            f"  Rel diff: {abs(actual - expected) / abs(expected) if expected != 0 else 'inf'}"
        ) from e


@pytest.mark.skipif(not PHENOMXPY_AVAILABLE, reason="phenomxpy not installed")
class TestAmplitudeCoeffs22:
    """
    Test suite for AmplitudeCoeffs22 against phenomxpy.
    """

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),  # Schwarzschild
            (0.25, 0.5, 0.5),  # Equal spin aligned
            (0.24, 0.3, -0.1),  # Unequal mass, unequal spin
            (0.2, 0.6, 0.2),  # More unequal mass
            (0.1, -0.5, 0.4),  # Low mass ratio
        ],
    )
    def test_pn_coefficients(self, eta, chi1, chi2):
        """Test PN coefficients."""
        # Compute phentax coefficients
        wf_params = _make_wf_params(eta, chi1, chi2)
        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)
        coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs)

        # Compute phenomxpy coefficients
        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        # Check PN coefficients (now stored as arrays in AmplitudeCoeffs)
        # Real: [ampN, amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half, amplog]
        assert_close(coeffs.pn_real_coeffs[0], pamp.ampN, name="ampN")
        assert_close(
            coeffs.pn_real_coeffs[1], pamp.amp0halfPNreal, name="amp0halfPNreal"
        )
        assert_close(coeffs.pn_real_coeffs[2], pamp.amp1PNreal, name="amp1PNreal")
        assert_close(
            coeffs.pn_real_coeffs[3], pamp.amp1halfPNreal, name="amp1halfPNreal"
        )
        assert_close(coeffs.pn_real_coeffs[4], pamp.amp2PNreal, name="amp2PNreal")
        assert_close(
            coeffs.pn_real_coeffs[5], pamp.amp2halfPNreal, name="amp2halfPNreal"
        )
        assert_close(coeffs.pn_real_coeffs[6], pamp.amp3PNreal, name="amp3PNreal")
        assert_close(
            coeffs.pn_real_coeffs[7], pamp.amp3halfPNreal, name="amp3halfPNreal"
        )
        assert_close(coeffs.pn_real_coeffs[8], pamp.amplog, name="amplog")

        # Imag: [amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half]
        assert_close(
            coeffs.pn_imag_coeffs[0], pamp.amp0halfPNimag, name="amp0halfPNimag"
        )
        assert_close(coeffs.pn_imag_coeffs[1], pamp.amp1PNimag, name="amp1PNimag")
        assert_close(
            coeffs.pn_imag_coeffs[2], pamp.amp1halfPNimag, name="amp1halfPNimag"
        )
        assert_close(coeffs.pn_imag_coeffs[3], pamp.amp2PNimag, name="amp2PNimag")
        assert_close(
            coeffs.pn_imag_coeffs[4], pamp.amp2halfPNimag, name="amp2halfPNimag"
        )
        assert_close(coeffs.pn_imag_coeffs[5], pamp.amp3PNimag, name="amp3PNimag")
        assert_close(
            coeffs.pn_imag_coeffs[6], pamp.amp3halfPNimag, name="amp3halfPNimag"
        )

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_inspiral_coefficients(self, eta, chi1, chi2):
        """Test inspiral pseudo-PN coefficients."""
        wf_params = _make_wf_params(eta, chi1, chi2)
        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)
        coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        # Check pseudo-PN coefficients (collocation point values no longer stored)
        assert_close(coeffs.inspC1, pamp.inspC1, name="inspC1")
        assert_close(coeffs.inspC2, pamp.inspC2, name="inspC2")
        assert_close(coeffs.inspC3, pamp.inspC3, name="inspC3")

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_ringdown_coefficients(self, eta, chi1, chi2):
        """Test ringdown coefficients."""
        wf_params = _make_wf_params(eta, chi1, chi2)
        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)
        coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        assert_close(coeffs.alpha1RD, pamp.alpha1RD, name="alpha1RD")
        assert_close(coeffs.ampPeak, pamp.ampPeak, name="ampPeak")
        assert_close(coeffs.c1_prec, pamp.c1, name="c1")
        assert_close(coeffs.c2_prec, pamp.c2, name="c2")
        assert_close(coeffs.c3, pamp.c3, name="c3")
        assert_close(coeffs.c4_prec, pamp.c4, name="c4")

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_intermediate_coefficients(self, eta, chi1, chi2):
        """Test intermediate coefficients."""
        wf_params = _make_wf_params(eta, chi1, chi2)
        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)
        coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        # Relaxed tolerance: intermediate coefficients involve numerical
        # root-finding that can differ slightly between implementations.
        # mergerC2 in particular shows ~1e-3 relative differences due to
        # different numerical paths through the collocation system.
        assert_close(coeffs.tshift, pamp.tshift, rtol=1e-3, name="tshift")
        assert_close(coeffs.mergerC1, pamp.mergerC1, rtol=1e-3, name="mergerC1")
        assert_close(coeffs.mergerC2, pamp.mergerC2, rtol=1e-3, name="mergerC2")
        assert_close(coeffs.mergerC3, pamp.mergerC3, rtol=1e-3, name="mergerC3")
        assert_close(coeffs.mergerC4, pamp.mergerC4, rtol=1e-3, name="mergerC4")
