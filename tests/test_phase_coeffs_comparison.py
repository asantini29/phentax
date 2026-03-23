# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests comparing PhaseCoeffs22 between phentax and phenomxpy.

Verifies that all phase/omega coefficients match exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

# Import phenomxpy for reference
from phenomxpy.phenomt.internals import pPhase, pWF, pWFHM

# Import phentax
from phentax.core.internals import compute_waveform_params
from phentax.core.phase import compute_phase_coeffs_22

total_mass = 100


def _make_wf_params(eta, s1z, s2z, total_mass=100.0, f_min=20.0):
    """Create WaveformParams from eta and spins for comparison tests."""
    import numpy as np

    delta = np.sqrt(1.0 - 4.0 * eta)
    m1_solar = 0.5 * (1.0 + delta) * total_mass
    m2_solar = 0.5 * (1.0 - delta) * total_mass
    return compute_waveform_params(
        m1=m1_solar,
        m2=m2_solar,
        s1z=s1z,
        s2z=s2z,
        distance=100.0,
        inclination=0.0,
        phi_ref=0.0,
        psi=0.0,
        f_ref=f_min,
        f_min=f_min,
    )


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


class TestPhaseCoeffs22:
    """Test PhaseCoeffs22 against phenomxpy pPhase."""

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.25, 0.5, -0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
            (0.15, 0.8, 0.0),
        ],
    )
    def test_pn_coefficients(self, eta, s1z, s2z):
        """Test PN coefficients match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.omega1PN, pphase.omega1PN, name="omega1PN")
        assert_close(coeffs.omega1halfPN, pphase.omega1halfPN, name="omega1halfPN")
        assert_close(coeffs.omega2PN, pphase.omega2PN, name="omega2PN")
        assert_close(coeffs.omega2halfPN, pphase.omega2halfPN, name="omega2halfPN")
        assert_close(coeffs.omega3PN, pphase.omega3PN, name="omega3PN")
        assert_close(coeffs.omega3halfPN, pphase.omega3halfPN, name="omega3halfPN")

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
    def test_pseudo_pn_coefficients(self, eta, s1z, s2z):
        """Test pseudo-PN coefficients match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.omegaInspC1, pphase.omegaInspC1, name="omegaInspC1")
        assert_close(coeffs.omegaInspC2, pphase.omegaInspC2, name="omegaInspC2")
        assert_close(coeffs.omegaInspC3, pphase.omegaInspC3, name="omegaInspC3")
        assert_close(coeffs.omegaInspC4, pphase.omegaInspC4, name="omegaInspC4")
        assert_close(coeffs.omegaInspC5, pphase.omegaInspC5, name="omegaInspC5")
        assert_close(coeffs.omegaInspC6, pphase.omegaInspC6, name="omegaInspC6")

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
        ],
    )
    def test_ringdown_coefficients(self, eta, s1z, s2z):
        """Test ringdown coefficients match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.omegaRING, pphase.omegaRING, name="omegaRING")
        assert_close(coeffs.alpha1RD, pphase.alpha1RD, name="alpha1RD")
        assert_close(coeffs.omegaPeak, pphase.omegaPeak, name="omegaPeak")
        assert_close(coeffs.c1, pphase.c1, name="c1")
        assert_close(coeffs.c2, pphase.c2, name="c2")
        assert_close(coeffs.c3, pphase.c3, name="c3")

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
            (0.20, 0.6, 0.2),
        ],
    )
    def test_intermediate_coefficients(self, eta, s1z, s2z):
        """Test intermediate region coefficients match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.omegaMergerC1, pphase.omegaMergerC1, name="omegaMergerC1")
        assert_close(coeffs.omegaMergerC2, pphase.omegaMergerC2, name="omegaMergerC2")
        assert_close(coeffs.omegaMergerC3, pphase.omegaMergerC3, name="omegaMergerC3")

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
        ],
    )
    def test_times_and_cuts(self, eta, s1z, s2z):
        """Test timing quantities match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.inspiral_cut, pphase.inspiral_cut, name="inspiral_cut")
        assert_close(coeffs.tt0, pphase.tt0, name="tt0")
        assert_close(coeffs.tEarly, pphase.tEarly, name="tEarly")

    @pytest.mark.parametrize(
        "eta,s1z,s2z",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, 0.1),
        ],
    )
    def test_phase_offsets(self, eta, s1z, s2z):
        """Test phase continuity offsets match."""
        pwf = pWFHM(
            mode=[2, 2], total_mass=total_mass, eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        pphase = pPhase(pwf)

        wf_params = _make_wf_params(eta, s1z, s2z)
        _, coeffs = compute_phase_coeffs_22(wf_params)

        assert_close(coeffs.phOffInsp, pphase.phOffInsp, name="phOffInsp")
        assert_close(coeffs.phOffMerger, pphase.phOffMerger, name="phOffMerger")
        assert_close(coeffs.phOffRD, pphase.phOffRD, name="phOffRD")
