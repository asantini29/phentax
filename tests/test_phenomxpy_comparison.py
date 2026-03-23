# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Comparison tests between phentax (JAX) and phenomxpy (NumPy/Numba).

These tests verify that phentax produces numerically equivalent results
to the reference phenomxpy implementation within specified tolerances.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

# Import phenomxpy (reference implementation)
from phenomxpy.phenomt import fits as pxpy_fits
from phenomxpy.phenomt import internals as pxpy_internals

# Import phentax (JAX implementation)
from phentax.core import fits as ptax_fits
from phentax.utils import utility as ptax_utils

# =============================================================================
# Test parameters - various mass ratios and spins
# =============================================================================

# Test points: (eta, s1z, s2z)
TEST_PARAMS = [
    # Equal mass, non-spinning
    (0.25, 0.0, 0.0),
    # Equal mass, aligned spins
    (0.25, 0.5, 0.5),
    # Equal mass, anti-aligned spins
    (0.25, 0.5, -0.5),
    # Equal mass, high spins
    (0.25, 0.9, 0.9),
    # q=2 (eta~0.222), non-spinning
    (0.222222, 0.0, 0.0),
    # q=2, spinning
    (0.222222, 0.3, 0.2),
    # q=4 (eta=0.16), non-spinning
    (0.16, 0.0, 0.0),
    # q=4, spinning
    (0.16, 0.6, 0.4),
    # q=10 (eta~0.0826), non-spinning
    (0.0826446, 0.0, 0.0),
    # q=10, spinning
    (0.0826446, 0.8, 0.1),
]

# Final spin values for ringdown tests
FINAL_SPIN_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]

# Tolerance for numerical comparisons
RTOL = 1e-10  # Relative tolerance
ATOL = 1e-14  # Absolute tolerance (for values near zero)


# =============================================================================
# Helper functions
# =============================================================================


def assert_close(phentax_val, phenomxpy_val, name: str, rtol=RTOL, atol=ATOL):
    """Assert that two values are numerically close."""
    phentax_np = np.asarray(phentax_val)
    phenomxpy_np = np.asarray(phenomxpy_val)

    if not np.allclose(phentax_np, phenomxpy_np, rtol=rtol, atol=atol):
        rel_err = np.abs(phentax_np - phenomxpy_np) / (np.abs(phenomxpy_np) + 1e-30)
        pytest.fail(
            f"{name} mismatch:\n"
            f"  phentax:   {phentax_np}\n"
            f"  phenomxpy: {phenomxpy_np}\n"
            f"  rel_error: {rel_err}"
        )


# =============================================================================
# Test: Utility functions
# =============================================================================


class TestUtilsComparison:
    """Compare utility functions between implementations."""

    @pytest.mark.parametrize("eta", [0.25, 0.222, 0.16, 0.1, 0.05])
    def test_m1ofeta(self, eta):
        """Compare m1ofeta."""
        phentax = float(ptax_utils.m1ofeta(eta))
        phenomxpy = pxpy_fits.m1ofeta(eta)
        assert_close(phentax, phenomxpy, f"m1ofeta(eta={eta})")

    @pytest.mark.parametrize("eta", [0.25, 0.222, 0.16, 0.1, 0.05])
    def test_m2ofeta(self, eta):
        """Compare m2ofeta."""
        phentax = float(ptax_utils.m2ofeta(eta))
        phenomxpy = pxpy_fits.m2ofeta(eta)
        assert_close(phentax, phenomxpy, f"m2ofeta(eta={eta})")

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_sTotR(self, eta, s1z, s2z):
        """Compare sTotR spin combination."""
        phentax = float(ptax_utils.sTotR(eta, s1z, s2z))
        phenomxpy = pxpy_fits.sTotR(eta, s1z, s2z)
        assert_close(phentax, phenomxpy, f"sTotR(eta={eta}, s1z={s1z}, s2z={s2z})")


# =============================================================================
# Test: Final state fits
# =============================================================================


class TestFinalStateFits:
    """Compare final mass and spin fits."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_final_mass_2017(self, eta, s1z, s2z):
        """Compare final mass fit."""
        phentax = float(ptax_fits.final_mass_2017(eta, s1z, s2z))
        phenomxpy = pxpy_fits.IMRPhenomX_FinalMass2017(eta, s1z, s2z)
        assert_close(
            phentax, phenomxpy, f"FinalMass2017(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_final_spin_2017(self, eta, s1z, s2z):
        """Compare final spin fit."""
        phentax = float(ptax_fits.final_spin_2017(eta, s1z, s2z))
        phenomxpy = pxpy_fits.IMRPhenomX_FinalSpin2017(eta, s1z, s2z)
        assert_close(
            phentax, phenomxpy, f"FinalSpin2017(eta={eta}, s1z={s1z}, s2z={s2z})"
        )


# =============================================================================
# Test: QNM frequency fits
# =============================================================================


class TestQNMFits:
    """Compare quasi-normal mode frequency fits."""

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fring_22(self, af):
        """Compare fring_22 (ringdown frequency)."""
        phentax = float(ptax_fits.fring_22(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fringfit(af, 22)
        assert_close(phentax, phenomxpy, f"fring_22(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_22(self, af):
        """Compare fdamp_22 (damping frequency)."""
        phentax = float(ptax_fits.fdamp_22(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampfit(af, 22)
        assert_close(phentax, phenomxpy, f"fdamp_22(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_n2_22(self, af):
        """Compare fdamp_n2_22 (2nd overtone damping)."""
        phentax = float(ptax_fits.fdamp_n2_22(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampn2fit(af, 22)
        assert_close(phentax, phenomxpy, f"fdamp_n2_22(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fring_21(self, af):
        """Compare fring_21."""
        phentax = float(ptax_fits.fring_21(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fringfit(af, 21)
        assert_close(phentax, phenomxpy, f"fring_21(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_21(self, af):
        """Compare fdamp_21."""
        phentax = float(ptax_fits.fdamp_21(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampfit(af, 21)
        assert_close(phentax, phenomxpy, f"fdamp_21(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fring_33(self, af):
        """Compare fring_33."""
        phentax = float(ptax_fits.fring_33(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fringfit(af, 33)
        assert_close(phentax, phenomxpy, f"fring_33(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_33(self, af):
        """Compare fdamp_33."""
        phentax = float(ptax_fits.fdamp_33(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampfit(af, 33)
        assert_close(phentax, phenomxpy, f"fdamp_33(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fring_44(self, af):
        """Compare fring_44."""
        phentax = float(ptax_fits.fring_44(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fringfit(af, 44)
        assert_close(phentax, phenomxpy, f"fring_44(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_44(self, af):
        """Compare fdamp_44."""
        phentax = float(ptax_fits.fdamp_44(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampfit(af, 44)
        assert_close(phentax, phenomxpy, f"fdamp_44(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fring_55(self, af):
        """Compare fring_55."""
        phentax = float(ptax_fits.fring_55(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fringfit(af, 55)
        assert_close(phentax, phenomxpy, f"fring_55(af={af})")

    @pytest.mark.parametrize("af", FINAL_SPIN_VALUES)
    def test_fdamp_55(self, af):
        """Compare fdamp_55."""
        phentax = float(ptax_fits.fdamp_55(af))
        phenomxpy = pxpy_fits.IMRPhenomT_fdampfit(af, 55)
        assert_close(phentax, phenomxpy, f"fdamp_55(af={af})")


# =============================================================================
# Test: Intermediate fits
# =============================================================================


class TestIntermediateFits:
    """Compare intermediate region calibration fits.

    Tests the unified intermediate_freq_cp1 and intermediate_amp_cp1 functions
    against phenomxpy's IMRPhenomT_Intermediate_Freq_CP1 and
    IMRPhenomT_Intermediate_Amp_CP1 for all supported modes.
    """

    @pytest.mark.parametrize("mode", [22, 21, 33, 44, 55])
    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_intermediate_freq_cp1(self, eta, s1z, s2z, mode):
        """Compare intermediate frequency CP1 for all modes."""
        pWF = pxpy_internals.pWFHM(
            mode=[mode // 10, mode % 10], eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        phenomxpy = pxpy_fits.IMRPhenomT_Intermediate_Freq_CP1(pWF)
        phentax = float(ptax_fits.intermediate_freq_cp1(eta, s1z, s2z, mode))
        assert_close(phentax, phenomxpy, f"intermediate_freq_cp1 mode {mode}")

    @pytest.mark.parametrize("mode", [22, 21, 33, 44, 55])
    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_intermediate_amp_cp1(self, eta, s1z, s2z, mode):
        """Compare intermediate amplitude CP1 for all modes."""
        pWF = pxpy_internals.pWFHM(
            mode=[mode // 10, mode % 10], eta=eta, s1=s1z, s2=s2z, f_min=20.0
        )
        phenomxpy = pxpy_fits.IMRPhenomT_Intermediate_Amp_CP1(pWF)
        phentax = float(ptax_fits.intermediate_amp_cp1(eta, s1z, s2z, mode))
        assert_close(phentax, phenomxpy, f"intermediate_amp_cp1 mode {mode}")


# =============================================================================
# Test: Peak fits (using pWFHM objects)
# =============================================================================


class TestPeakFits:
    """Compare peak frequency and amplitude fits."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_freq_22(self, eta, s1z, s2z):
        """Compare peak frequency for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakFrequency(pWF)
        phentax = float(ptax_fits.peak_freq_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_freq_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_freq_21(self, eta, s1z, s2z):
        """Compare peak frequency for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakFrequency(pWF)
        phentax = float(ptax_fits.peak_freq_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_freq_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_freq_33(self, eta, s1z, s2z):
        """Compare peak frequency for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakFrequency(pWF)
        phentax = float(ptax_fits.peak_freq_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_freq_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_freq_44(self, eta, s1z, s2z):
        """Compare peak frequency for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakFrequency(pWF)
        phentax = float(ptax_fits.peak_freq_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_freq_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_freq_55(self, eta, s1z, s2z):
        """Compare peak frequency for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakFrequency(pWF)
        phentax = float(ptax_fits.peak_freq_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_freq_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_amp_22(self, eta, s1z, s2z):
        """Compare peak amplitude for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakAmp(pWF)
        phentax = float(ptax_fits.peak_amp_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_amp_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_amp_21(self, eta, s1z, s2z):
        """Compare peak amplitude for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakAmp(pWF)
        phentax = float(ptax_fits.peak_amp_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_amp_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_amp_33(self, eta, s1z, s2z):
        """Compare peak amplitude for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakAmp(pWF)
        phentax = float(ptax_fits.peak_amp_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_amp_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_amp_44(self, eta, s1z, s2z):
        """Compare peak amplitude for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakAmp(pWF)
        phentax = float(ptax_fits.peak_amp_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_amp_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_peak_amp_55(self, eta, s1z, s2z):
        """Compare peak amplitude for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_PeakAmp(pWF)
        phentax = float(ptax_fits.peak_amp_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"peak_amp_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )


# =============================================================================
# Test: Ringdown fits
# =============================================================================


class TestRingdownFits:
    """Compare ringdown calibration fits."""

    # --- Ringdown Frequency D2 ---
    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d2_22(self, eta, s1z, s2z):
        """Compare ringdown frequency D2 for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D2(pWF)
        phentax = float(ptax_fits.rd_freq_d2_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d2_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d2_21(self, eta, s1z, s2z):
        """Compare ringdown frequency D2 for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D2(pWF)
        phentax = float(ptax_fits.rd_freq_d2_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d2_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d2_33(self, eta, s1z, s2z):
        """Compare ringdown frequency D2 for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D2(pWF)
        phentax = float(ptax_fits.rd_freq_d2_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d2_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d2_44(self, eta, s1z, s2z):
        """Compare ringdown frequency D2 for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D2(pWF)
        phentax = float(ptax_fits.rd_freq_d2_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d2_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d2_55(self, eta, s1z, s2z):
        """Compare ringdown frequency D2 for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D2(pWF)
        phentax = float(ptax_fits.rd_freq_d2_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d2_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    # --- Ringdown Frequency D3 ---
    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d3_22(self, eta, s1z, s2z):
        """Compare ringdown frequency D3 for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D3(pWF)
        phentax = float(ptax_fits.rd_freq_d3_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d3_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d3_21(self, eta, s1z, s2z):
        """Compare ringdown frequency D3 for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D3(pWF)
        phentax = float(ptax_fits.rd_freq_d3_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d3_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d3_33(self, eta, s1z, s2z):
        """Compare ringdown frequency D3 for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D3(pWF)
        phentax = float(ptax_fits.rd_freq_d3_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d3_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d3_44(self, eta, s1z, s2z):
        """Compare ringdown frequency D3 for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D3(pWF)
        phentax = float(ptax_fits.rd_freq_d3_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d3_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_freq_d3_55(self, eta, s1z, s2z):
        """Compare ringdown frequency D3 for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_RD_Freq_D3(pWF)
        phentax = float(ptax_fits.rd_freq_d3_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_freq_d3_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    # --- Ringdown Amplitude C3 ---
    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_amp_c3_22(self, eta, s1z, s2z):
        """Compare ringdown amplitude C3 for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Ringdown_Amp_C3(pWF)
        phentax = float(ptax_fits.rd_amp_c3_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_amp_c3_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_amp_c3_21(self, eta, s1z, s2z):
        """Compare ringdown amplitude C3 for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Ringdown_Amp_C3(pWF)
        phentax = float(ptax_fits.rd_amp_c3_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_amp_c3_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_amp_c3_33(self, eta, s1z, s2z):
        """Compare ringdown amplitude C3 for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Ringdown_Amp_C3(pWF)
        phentax = float(ptax_fits.rd_amp_c3_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_amp_c3_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_amp_c3_44(self, eta, s1z, s2z):
        """Compare ringdown amplitude C3 for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Ringdown_Amp_C3(pWF)
        phentax = float(ptax_fits.rd_amp_c3_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_amp_c3_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_rd_amp_c3_55(self, eta, s1z, s2z):
        """Compare ringdown amplitude C3 for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Ringdown_Amp_C3(pWF)
        phentax = float(ptax_fits.rd_amp_c3_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"rd_amp_c3_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )


# =============================================================================
# Test: Time shift
# =============================================================================


class TestTimeShift:
    """Compare time shift fits."""

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_tshift_22(self, eta, s1z, s2z):
        """Compare time shift for 22 mode (should always be 0)."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_tshift(pWF)
        phentax = float(ptax_fits.tshift_22(eta, s1z, s2z))
        assert_close(phentax, phenomxpy, f"tshift_22(eta={eta}, s1z={s1z}, s2z={s2z})")

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_tshift_21(self, eta, s1z, s2z):
        """Compare time shift for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_tshift(pWF)
        phentax = float(ptax_fits.tshift_21(eta, s1z, s2z))
        assert_close(phentax, phenomxpy, f"tshift_21(eta={eta}, s1z={s1z}, s2z={s2z})")

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_tshift_33(self, eta, s1z, s2z):
        """Compare time shift for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_tshift(pWF)
        phentax = float(ptax_fits.tshift_33(eta, s1z, s2z))
        assert_close(phentax, phenomxpy, f"tshift_33(eta={eta}, s1z={s1z}, s2z={s2z})")

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_tshift_44(self, eta, s1z, s2z):
        """Compare time shift for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_tshift(pWF)
        phentax = float(ptax_fits.tshift_44(eta, s1z, s2z))
        assert_close(phentax, phenomxpy, f"tshift_44(eta={eta}, s1z={s1z}, s2z={s2z})")

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_tshift_55(self, eta, s1z, s2z):
        """Compare time shift for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_tshift(pWF)
        phentax = float(ptax_fits.tshift_55(eta, s1z, s2z))
        assert_close(phentax, phenomxpy, f"tshift_55(eta={eta}, s1z={s1z}, s2z={s2z})")


# =============================================================================
# Test: Inspiral TaylorT3 t0 fits (mode-independent)
# =============================================================================


class TestInspiralT0Fits:
    """Compare inspiral TaylorT3 t0 fits.

    Note: inspiral_t0 is mode-independent in IMRPhenomT. All modes return
    the same value for the same (eta, s1z, s2z) parameters.
    """

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_inspiral_t0_22(self, eta, s1z, s2z):
        """Compare inspiral t0 for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_TaylorT3_t0(pWF)
        phentax = float(ptax_fits.inspiral_t0_22(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"inspiral_t0_22(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_inspiral_t0_21(self, eta, s1z, s2z):
        """Compare inspiral t0 for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_TaylorT3_t0(pWF)
        phentax = float(ptax_fits.inspiral_t0_21(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"inspiral_t0_21(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_inspiral_t0_33(self, eta, s1z, s2z):
        """Compare inspiral t0 for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_TaylorT3_t0(pWF)
        phentax = float(ptax_fits.inspiral_t0_33(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"inspiral_t0_33(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_inspiral_t0_44(self, eta, s1z, s2z):
        """Compare inspiral t0 for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_TaylorT3_t0(pWF)
        phentax = float(ptax_fits.inspiral_t0_44(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"inspiral_t0_44(eta={eta}, s1z={s1z}, s2z={s2z})"
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    def test_inspiral_t0_55(self, eta, s1z, s2z):
        """Compare inspiral t0 for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_TaylorT3_t0(pWF)
        phentax = float(ptax_fits.inspiral_t0_55(eta, s1z, s2z))
        assert_close(
            phentax, phenomxpy, f"inspiral_t0_55(eta={eta}, s1z={s1z}, s2z={s2z})"
        )


# =============================================================================
# Test: Inspiral frequency collocation points (mode-independent)
# =============================================================================


class TestInspiralFreqCP:
    """Compare inspiral frequency collocation points.

    Note: Inspiral frequency CPs are mode-independent in IMRPhenomT.
    Tests all 5 collocation points (idx=1-5) for each parameter set.
    """

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3, 4, 5])
    def test_inspiral_freq_cp(self, eta, s1z, s2z, idx):
        """Compare inspiral frequency collocation point."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Freq_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_freq_cp(eta, s1z, s2z, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_freq_cp(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )


# =============================================================================
# Test: Inspiral amplitude collocation points (mode-dependent)
# =============================================================================


class TestInspiralAmpCP:
    """Compare inspiral amplitude collocation points.

    Note: Inspiral amplitude CPs are mode-dependent.
    Tests all 3 collocation points (idx=1-3) for each mode and parameter set.
    """

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_inspiral_amp_cp_22(self, eta, s1z, s2z, idx):
        """Compare inspiral amplitude collocation point for 22 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 2], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Amp_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_amp_cp(eta, s1z, s2z, 22, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_amp_cp_22(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_inspiral_amp_cp_21(self, eta, s1z, s2z, idx):
        """Compare inspiral amplitude collocation point for 21 mode."""
        pWF = pxpy_internals.pWFHM(mode=[2, 1], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Amp_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_amp_cp(eta, s1z, s2z, 21, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_amp_cp_21(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_inspiral_amp_cp_33(self, eta, s1z, s2z, idx):
        """Compare inspiral amplitude collocation point for 33 mode."""
        pWF = pxpy_internals.pWFHM(mode=[3, 3], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Amp_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_amp_cp(eta, s1z, s2z, 33, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_amp_cp_33(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_inspiral_amp_cp_44(self, eta, s1z, s2z, idx):
        """Compare inspiral amplitude collocation point for 44 mode."""
        pWF = pxpy_internals.pWFHM(mode=[4, 4], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Amp_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_amp_cp(eta, s1z, s2z, 44, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_amp_cp_44(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )

    @pytest.mark.parametrize("eta,s1z,s2z", TEST_PARAMS)
    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_inspiral_amp_cp_55(self, eta, s1z, s2z, idx):
        """Compare inspiral amplitude collocation point for 55 mode."""
        pWF = pxpy_internals.pWFHM(mode=[5, 5], eta=eta, s1=s1z, s2=s2z, f_min=20.0)
        phenomxpy = pxpy_fits.IMRPhenomT_Inspiral_Amp_CP(pWF, idx)
        phentax = float(ptax_fits.inspiral_amp_cp(eta, s1z, s2z, 55, idx))
        assert_close(
            phentax,
            phenomxpy,
            f"inspiral_amp_cp_55(eta={eta}, s1z={s1z}, s2z={s2z}, idx={idx})",
        )


# =============================================================================
# Summary test that runs all comparisons and reports
# =============================================================================


class TestSummary:
    """Summary statistics for all comparison tests."""

    def test_summary_stats(self):
        """Run a subset of comparisons and report statistics."""
        results = []

        for eta, s1z, s2z in TEST_PARAMS[:3]:  # Use first 3 param sets
            # Final state
            phentax_mf = float(ptax_fits.final_mass_2017(eta, s1z, s2z))
            phenomxpy_mf = pxpy_fits.IMRPhenomX_FinalMass2017(eta, s1z, s2z)
            results.append(
                (
                    "FinalMass",
                    abs(phentax_mf - phenomxpy_mf) / abs(phenomxpy_mf + 1e-30),
                )
            )

            phentax_af = float(ptax_fits.final_spin_2017(eta, s1z, s2z))
            phenomxpy_af = pxpy_fits.IMRPhenomX_FinalSpin2017(eta, s1z, s2z)
            results.append(
                (
                    "FinalSpin",
                    abs(phentax_af - phenomxpy_af) / abs(phenomxpy_af + 1e-30),
                )
            )

        # Report
        max_err = max(r[1] for r in results)
        mean_err = sum(r[1] for r in results) / len(results)

        print(f"\nComparison summary:")
        print(f"  Max relative error: {max_err:.2e}")
        print(f"  Mean relative error: {mean_err:.2e}")

        assert max_err < 1e-10, f"Max error {max_err} exceeds tolerance"
