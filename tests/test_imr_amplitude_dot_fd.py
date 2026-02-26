import jax
import jax.numpy as jnp
import numpy as np


from phentax.core.amplitude import imr_amplitude, imr_amplitude_dot, compute_amplitude_coeffs_22
from phentax.core.phase import compute_phase_coeffs_22
from phentax.core.internals import compute_waveform_params

def main():

    # Use realistic waveform parameters (from test_amp_phase.py)
    m1 = 50.0
    m2 = 50.0
    chi1 = 0.0
    chi2 = 0.0
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
    amp_coeffs = compute_amplitude_coeffs_22(wf_params, phase_coeffs_22)

    # Use a time in the inspiral region
    time = jnp.array(-200.0)
    eta = wf_params.eta

    # Analytic derivative
    analytic = imr_amplitude_dot(time, eta, amp_coeffs, phase_coeffs_22)

    # Finite difference derivative
    eps = 1e-5
    amp_p = imr_amplitude(time + eps, eta, amp_coeffs, phase_coeffs_22)
    amp_m = imr_amplitude(time - eps, eta, amp_coeffs, phase_coeffs_22)
    fd = (amp_p - amp_m) / (2 * eps)

    print(f"Analytic derivative: {analytic}")
    print(f"Finite diff approx: {fd}")
    print(f"Difference: {jnp.abs(analytic - fd)}")

    # Assert close
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)
    print("Test passed!")

if __name__ == "__main__":
    main()
