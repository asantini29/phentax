"""Test for the usage of tref/tmin fref/fmin in the waveform generation"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from phentax.waveform import IMRPhenomTHM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

tlowfit = True
tol = 1e-12

waveform = IMRPhenomTHM(
    higher_modes="all",
    include_negative_modes=True,
    t_low_fit=tlowfit,
    coarse_grain=False,
    atol=tol,
    rtol=tol,
    T=None,
)

m1 = 5e6
m2 = 3e6
chi1 = 0.9
chi2 = 0.3
distance = 500.0
inclination = jnp.pi / 3.0
phi_ref = 0.0
psi = 0.0

dt = 20.0
f_min = 1e-4
f_ref = f_min
t_ref = -3600.0
t_min = t_ref


class TestMinRefQuantities:
    """Test that the minimum reference quantities are correctly handled."""

    def test_fmin_fref(self):
        """Generate the waveform using f_min and f_ref"""

        times, mask, h_plus, h_cross = waveform.compute_polarizations_at_once(
            m1=m1,
            m2=m2,
            chi1z=chi1,
            chi2z=chi2,
            distance=distance,
            phi_ref=phi_ref,
            inclination=inclination,
            psi=psi,
            f_ref=f_ref,
            f_min=f_min,
            delta_t=dt,
        )

    def test_tmin_tref(self):
        """Generate the waveform using t_min and t_ref"""

        times, mask, h_plus, h_cross = waveform.compute_polarizations_at_once(
            m1=m1,
            m2=m2,
            chi1z=chi1,
            chi2z=chi2,
            distance=distance,
            phi_ref=phi_ref,
            inclination=inclination,
            psi=psi,
            t_ref=t_ref,
            t_min=t_min,
            delta_t=dt,
        )

    def test_fmin_tref(self):
        """Generate the waveform using f_min and t_ref"""

        times, mask, h_plus, h_cross = waveform.compute_polarizations_at_once(
            m1=m1,
            m2=m2,
            chi1z=chi1,
            chi2z=chi2,
            distance=distance,
            phi_ref=phi_ref,
            inclination=inclination,
            psi=psi,
            f_min=f_min,
            t_ref=t_ref,
            delta_t=dt,
        )
        logger.debug(f"Generated waveform with f_min={f_min} and t_ref={t_ref}")

    def test_batched_fmin_fref(self):
        """Generate the waveform using f_min and f_ref in a batched way"""

        batch_size = 3
        m1s = jnp.array([m1] * batch_size)
        m2s = jnp.array([m2] * batch_size)
        chi1s = jnp.array([chi1] * batch_size)
        chi2s = jnp.array([chi2] * batch_size)
        distances = jnp.array([distance] * batch_size)
        phi_refs = jnp.array([phi_ref] * batch_size)
        inclinations = jnp.array([inclination] * batch_size)
        psis = jnp.array([psi] * batch_size)
        f_refs = jnp.array([f_ref] * batch_size)
        f_mins = jnp.array([f_min] * batch_size)

        times, mask, h_plus, h_cross = waveform.compute_polarizations_at_once(
            m1=m1s,
            m2=m2s,
            chi1z=chi1s,
            chi2z=chi2s,
            distance=distances,
            phi_ref=phi_refs,
            inclination=inclinations,
            psi=psis,
            f_ref=f_refs,
            f_min=f_mins,
            delta_t=dt,
        )

    def test_batched_tmin_tref(self):
        """Generate the waveform using t_min and t_ref in a batched way"""

        batch_size = 3
        m1s = jnp.array([m1] * batch_size)
        m2s = jnp.array([m2] * batch_size)
        chi1s = jnp.array([chi1] * batch_size)
        chi2s = jnp.array([chi2] * batch_size)
        distances = jnp.array([distance] * batch_size)
        phi_refs = jnp.array([phi_ref] * batch_size)
        inclinations = jnp.array([inclination] * batch_size)
        psis = jnp.array([psi] * batch_size)
        t_refs = jnp.array([t_ref] * batch_size)
        t_mins = jnp.array([t_min] * batch_size)
        f_refs = jnp.array([f_ref] * batch_size)
        f_mins = jnp.array([f_min] * batch_size)

        times, mask, h_plus, h_cross = waveform.compute_polarizations_at_once(
            m1=m1s,
            m2=m2s,
            chi1z=chi1s,
            chi2z=chi2s,
            distance=distances,
            phi_ref=phi_refs,
            inclination=inclinations,
            psi=psis,
            t_ref=t_refs,
            t_min=t_mins,
            delta_t=dt,
        )
