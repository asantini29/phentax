
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phentax.core.amplitude import imr_amplitude, imr_amplitude_dot, compute_amplitude_coeffs_22, compute_amplitude_coeffs_hm
from phentax.core.phase import compute_phase_coeffs_22
from phentax.core.internals import compute_waveform_params

# Use realistic waveform parameters (from test_amp_phase.py)
m1 = 50.0
m2 = 30.0
chi1 = 0.5
chi2 = 0.7
distance = 1.0
inclination = 1.0
phi_ref = 0.0
psi = 0.0
f_ref = 10.0
f_min = 20.0

# Compute waveform parameters
wf_params = compute_waveform_params(
    m1, m2, chi1, chi2, distance, inclination, phi_ref, psi, f_ref, f_min
)
# Compute phase and amplitude coefficients
wf_params, phase_coeffs_22 = compute_phase_coeffs_22(wf_params)
amp_coeffs_22 = compute_amplitude_coeffs_22(wf_params, phase_coeffs_22)

time = jnp.array(-200.0)
eta = wf_params.eta
eps = 1e-5

def finite_diff_amp_dot(time, eta, amp_coeffs, phase_coeffs_22, eps):
    amp_p = imr_amplitude(time + eps, eta, amp_coeffs, phase_coeffs_22)
    amp_m = imr_amplitude(time - eps, eta, amp_coeffs, phase_coeffs_22)
    return (amp_p - amp_m) / (2 * eps)

def test_imr_amplitude_dot_22():
    analytic = imr_amplitude_dot(time, eta, amp_coeffs_22, phase_coeffs_22)
    fd = finite_diff_amp_dot(time, eta, amp_coeffs_22, phase_coeffs_22, eps)
    print(f"Analytic derivative (22): {analytic}")
    print(f"Finite diff approx (22): {fd}")
    print(f"Difference (22): {jnp.abs(analytic - fd)}")
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)

@pytest.mark.parametrize("mode", [21, 33, 44, 55])
def test_imr_amplitude_dot_higher_modes(mode):
    amp_coeffs_hm = compute_amplitude_coeffs_hm(wf_params, phase_coeffs_22, mode=mode)
    analytic_hm = imr_amplitude_dot(time, eta, amp_coeffs_hm, phase_coeffs_22)
    fd_hm = finite_diff_amp_dot(time, eta, amp_coeffs_hm, phase_coeffs_22, eps)
    print(f"Analytic derivative ({mode}): {analytic_hm}")
    print(f"Finite diff approx ({mode}): {fd_hm}")
    print(f"Difference ({mode}): {jnp.abs(analytic_hm - fd_hm)}")
    np.testing.assert_allclose(analytic_hm, fd_hm, rtol=1e-4, atol=1e-6)