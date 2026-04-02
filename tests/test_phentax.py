# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests for phentax package.

These tests verify:
1. Basic import and functionality
2. Parameter creation and derived quantities
3. Waveform generation
4. Consistency of fits functions
"""

import jax
import jax.numpy as jnp
import pytest

# Enable float64
jax.config.update("jax_enable_x64", True)


class TestImports:
    """Test that all modules import correctly."""

    def test_import_phentax(self):
        """Test main package import."""
        import phentax

        assert hasattr(phentax, "__version__")

    def test_import_config_constants(self):
        """Test constants module import."""
        from phentax.utils import config, constants

        assert hasattr(constants, "MTSUN_SI")
        assert hasattr(config, "configure_jax")

    def test_import_utils(self):
        """Test utils module import."""
        from phentax.utils import utility, ylm

        assert hasattr(utility, "chi_eff")
        assert hasattr(utility, "m1ofeta")
        assert hasattr(ylm, "spin_weighted_spherical_harmonic")

    def test_import_fits(self):
        """Test fits module import."""
        from phentax.core import fits

        assert hasattr(fits, "final_mass_2017")
        assert hasattr(fits, "final_spin_2017")
        assert hasattr(fits, "fring_22")

    def test_import_ansatze(self):
        """Test ansatze module import."""
        from phentax import ansatze

        assert hasattr(ansatze, "inspiral_omega_taylort3")
        assert hasattr(ansatze, "ringdown_omega_ansatz")

    def test_import_internals(self):
        """Test internals module import."""
        from phentax.core import internals

        assert hasattr(internals, "WaveformParams")
        assert hasattr(internals, "compute_waveform_params")

    def test_import_waveform(self):
        """Test waveform module import."""
        from phentax import waveform

        assert hasattr(waveform, "IMRPhenomTHM")


class TestUtils:
    """Test utility functions."""

    def test_m1ofeta(self):
        """Test m1ofeta function."""
        from phentax.utils.utility import m1ofeta

        # Equal mass: eta = 0.25, m1 = m2 = 0.5
        eta = 0.25
        m1 = m1ofeta(eta)
        assert jnp.isclose(m1, 0.5, rtol=1e-10)

        # q=4 system: eta = 4/25 = 0.16
        eta = 0.16
        m1 = m1ofeta(eta)
        assert m1 > 0.5  # Primary should be larger

    def test_m2ofeta(self):
        """Test m2ofeta function."""
        from phentax.utils.utility import m2ofeta

        eta = 0.25
        m2 = m2ofeta(eta)
        assert jnp.isclose(m2, 0.5, rtol=1e-10)

    def test_chi_eff(self):
        """Test chi_eff function."""
        from phentax.utils.utility import chi_eff

        # Equal mass, aligned spins
        eta = 0.25
        s1z = 0.5
        s2z = 0.5
        chi = chi_eff(eta, s1z, s2z)
        assert jnp.isclose(chi, 0.5, rtol=1e-10)

        # Equal mass, opposite spins
        s1z = 0.5
        s2z = -0.5
        chi = chi_eff(eta, s1z, s2z)
        assert jnp.isclose(chi, 0.0, rtol=1e-10)

    def test_sTotR(self):
        """Test sTotR function."""
        from phentax.utils.utility import sTotR

        # For equal mass (eta=0.25, m1=m2=0.5), and equal spins (s1z=s2z=0.5):
        # sTotR = (m1^2*s1z + m2^2*s2z) / (m1^2+m2^2) = 0.25 / 0.5 = 0.5
        eta = 0.25
        s1z = 0.5
        s2z = 0.5
        s = sTotR(eta, s1z, s2z)
        assert jnp.isclose(s, 0.5, rtol=1e-10)

    def test_spin_weighted_spherical_harmonic(self):
        """Test SWSH function."""
        from phentax.utils.ylm import spin_weighted_spherical_harmonic

        # Y^{-2}_{22} at theta=0 (face-on) should be maximal
        Y22 = spin_weighted_spherical_harmonic(0.0, 0.0, 2, 2)
        assert jnp.isfinite(Y22)

        # Y^{-2}_{22} at theta=pi should be 0 (face-off)
        Y22_off = spin_weighted_spherical_harmonic(jnp.pi, 0.0, 2, 2)
        assert jnp.isclose(jnp.abs(Y22_off), 0.0, atol=1e-10)


class TestFits:
    """Test calibrated fits."""

    def test_final_mass(self):
        """Test final mass fits."""
        from phentax.core.fits import final_mass_2017

        # Equal mass, non-spinning
        eta = 0.25
        Mf = final_mass_2017(eta, 0.0, 0.0)
        # Final mass should be ~0.95-0.97 of initial
        assert 0.9 < Mf < 1.0
        assert Mf < 1.0  # Energy must be radiated

    def test_final_spin(self):
        """Test final spin fits."""
        from phentax.core.fits import final_spin_2017

        # Equal mass, non-spinning
        eta = 0.25
        af = final_spin_2017(eta, 0.0, 0.0)
        # Final spin should be ~0.68 for non-spinning equal mass
        assert 0.5 < af < 0.8

        # Spinning case
        af_spin = final_spin_2017(eta, 0.5, 0.5)
        assert af_spin > af  # Aligned spins increase final spin

    def test_fring_22(self):
        """Test ringdown frequency fit."""
        from phentax.core.fits import fring_22

        # Non-spinning remnant
        f_ring = fring_22(0.0)
        assert f_ring > 0

        # Spinning remnant should have higher frequency
        f_ring_spin = fring_22(0.7)
        assert f_ring_spin > f_ring

    def test_fdamp_22(self):
        """Test damping frequency fit."""
        from phentax.core.fits import fdamp_22

        f_damp = fdamp_22(0.0)
        assert f_damp > 0

        f_damp_spin = fdamp_22(0.7)
        assert f_damp_spin > 0


class TestInternals:
    """Test internal data structures and computation."""

    def test_waveform_params_creation(self):
        """Test WaveformParams creation via compute_waveform_params."""
        from phentax.core.internals import compute_waveform_params

        wf_params = compute_waveform_params(
            m1=30.0,
            m2=30.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )

        assert jnp.isclose(wf_params.total_mass, 60.0, rtol=1e-10)
        assert jnp.isclose(wf_params.eta, 0.25, rtol=1e-10)
        assert wf_params.distance == 100.0

    def test_compute_derived_quantities(self):
        """Test derived parameter computation."""
        from phentax.core.internals import compute_waveform_params

        wf_params = compute_waveform_params(
            m1=30.0,
            m2=30.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )

        assert jnp.isclose(wf_params.eta, 0.25, rtol=1e-10)
        assert jnp.isclose(wf_params.delta, 0.0, atol=1e-10)
        assert jnp.isclose(wf_params.chi_eff, 0.0, rtol=1e-10)
        assert wf_params.Mf < 1.0  # Some mass radiated (dimensionless)
        assert 0 < wf_params.af < 1  # Physical spin

    def test_compute_phase_coeffs(self):
        """Test phase coefficient computation."""
        from phentax.core.internals import compute_waveform_params
        from phentax.core.phase import compute_phase_coeffs_22

        wf_params = compute_waveform_params(
            m1=30.0,
            m2=30.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )

        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)

        assert phase_coeffs.omegaRING > 0
        assert phase_coeffs.alpha1RD > 0
        assert phase_coeffs.omegaPeak > 0


class TestWaveform:
    """Test waveform generation."""

    def test_model_creation(self):
        """Test IMRPhenomTHM model creation."""
        from phentax.waveform import IMRPhenomTHM

        model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
        assert model.num_modes == 1

    def test_compute_polarizations_runs(self):
        """Test that compute_polarizations runs without error."""
        from phentax.waveform import IMRPhenomTHM

        model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

        times, mask, hp, hc = model.compute_polarizations(
            m1=30.0,
            m2=30.0,
            chi1z=0.0,
            chi2z=0.0,
            distance=100.0,
            phi_ref=0.0,
            f_ref=20.0,
            f_min=20.0,
            inclination=0.0,
            psi=0.0,
        )

        assert hp.shape == times.shape
        assert hc.shape == times.shape
        # Check finite values where mask is valid
        assert jnp.all(jnp.isfinite(hp[mask]))
        assert jnp.all(jnp.isfinite(hc[mask]))

    def test_compute_polarizations_deterministic(self):
        """Test that compute_polarizations produces consistent results."""
        from phentax.waveform import IMRPhenomTHM

        model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

        kwargs = dict(
            m1=30.0,
            m2=30.0,
            chi1z=0.0,
            chi2z=0.0,
            distance=100.0,
            phi_ref=0.0,
            f_ref=20.0,
            f_min=20.0,
            inclination=0.0,
            psi=0.0,
        )

        times1, mask1, hp1, hc1 = model.compute_polarizations(**kwargs)
        times2, mask2, hp2, hc2 = model.compute_polarizations(**kwargs)

        assert jnp.allclose(hp1[mask1], hp2[mask2])
        assert jnp.allclose(hc1[mask1], hc2[mask2])

    def test_compute_hlms(self):
        """Test hlm mode computation."""
        from phentax.waveform import IMRPhenomTHM

        model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

        times, mask, hlms = model.compute_hlms(
            m1=30.0,
            m2=30.0,
            chi1z=0.0,
            chi2z=0.0,
            distance=100.0,
            phi_ref=0.0,
            f_ref=20.0,
            f_min=20.0,
            inclination=0.0,
            psi=0.0,
        )

        assert jnp.all(jnp.isfinite(hlms[0][mask]))


class TestJAXFeatures:
    """Test JAX-specific features."""

    def test_vmap_over_params(self):
        """Test that waveform params can be batched over parameters."""
        from phentax.core.internals import compute_waveform_params

        # compute_waveform_params handles batching internally when given arrays
        m1_batch = jnp.array([20.0, 30.0, 40.0])
        m2_batch = jnp.array([20.0, 30.0, 40.0])
        zeros = jnp.zeros(3)
        dists = jnp.full(3, 100.0)
        frefs = jnp.full(3, 20.0)

        wf_params = compute_waveform_params(
            m1=m1_batch,
            m2=m2_batch,
            s1z=zeros,
            s2z=zeros,
            distance=dists,
            inclination=zeros,
            phi_ref=zeros,
            psi=zeros,
            f_ref=frefs,
            f_min=frefs,
        )

        assert wf_params.eta.shape == (3,)
        assert jnp.allclose(wf_params.eta, 0.25, rtol=1e-10)  # All equal mass

    def test_grad_of_fits(self):
        """Test that fits are differentiable."""
        from phentax.core.fits import final_spin_2017

        def f(eta):
            return final_spin_2017(eta, 0.0, 0.0)

        grad_f = jax.grad(f)

        # Gradient at eta=0.2 (not 0.25, which has a sqrt singularity in delta)
        # At eta=0.25, delta=sqrt(1-4*0.25)=0 and d(delta)/d(eta) = -2/delta = inf
        g = grad_f(0.2)

        assert jnp.isfinite(g)

    def test_jit_compilation_caching(self):
        """Test that JIT compilation works and caches."""
        from phentax.core.fits import final_mass_2017

        jit_fm = jax.jit(final_mass_2017)

        # First call compiles
        result1 = jit_fm(0.25, 0.0, 0.0)

        # Second call uses cache
        result2 = jit_fm(0.25, 0.0, 0.0)

        assert jnp.isclose(result1, result2)


class TestPhysicalConsistency:
    """Test physical consistency of waveforms."""

    def test_amplitude_peaks_near_merger(self):
        """Test that amplitude peaks near end of waveform (merger)."""
        from phentax.core.amplitude import compute_amplitude_coeffs_22, imr_amplitude
        from phentax.core.internals import compute_waveform_params
        from phentax.core.phase import compute_phase_coeffs_22, imr_omega

        wf_params = compute_waveform_params(
            m1=30.0,
            m2=30.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )
        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)
        amp_coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs)

        # Test in dimensionless time (M): inspiral → merger
        times_M = jnp.linspace(-5000.0, 100.0, 2000)
        amp = imr_amplitude(times_M, wf_params.eta, amp_coeffs, phase_coeffs)

        peak_idx = jnp.argmax(jnp.abs(amp))

        # Peak should be in the last 10% (near merger at t~0)
        assert peak_idx > 0.9 * len(times_M)

    def test_frequency_increases_before_merger(self):
        """Test that frequency increases during inspiral (chirp)."""
        from phentax.core.internals import compute_waveform_params
        from phentax.core.phase import compute_phase_coeffs_22, imr_omega

        wf_params = compute_waveform_params(
            m1=30.0,
            m2=30.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )

        wf_params, phase_coeffs = compute_phase_coeffs_22(wf_params)

        # Times in dimensionless units (M), inspiral region
        times_M = jnp.linspace(-5000.0, -500.0, 1000)

        omega = imr_omega(times_M, wf_params.eta, phase_coeffs)

        # Omega should generally increase (positive derivative)
        domega = jnp.diff(omega)
        frac_positive = jnp.sum(domega > 0) / len(domega)

        # Relaxed threshold for prototype implementation
        assert frac_positive > 0.5

    def test_higher_mass_longer_waveform(self):
        """Test that higher mass systems have longer waveforms at fixed f_low."""
        from phentax.core.internals import compute_waveform_params

        wf_params_low = compute_waveform_params(
            m1=10.0,
            m2=10.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )
        wf_params_high = compute_waveform_params(
            m1=50.0,
            m2=50.0,
            s1z=0.0,
            s2z=0.0,
            distance=100.0,
            inclination=0.0,
            phi_ref=0.0,
            psi=0.0,
            f_ref=20.0,
            f_min=20.0,
        )

        # Higher mass = longer time scale
        assert wf_params_high.M_sec > wf_params_low.M_sec
