import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from phenomxpy.phenomt.internals import pAmp, pPhase, pWF, pWFHM

from phentax.core.amplitude import (
    compute_amplitude_coeffs_22,
    compute_amplitude_coeffs_hm,
    imr_amplitude,
)
from phentax.core.internals import compute_waveform_params
from phentax.core.phase import (
    compute_phase_coeffs_22,
    compute_phase_coeffs_hm,
    imr_omega,
    imr_phase,
)
from phentax.utils.config import configure_jax
from phentax.utils.utility import check_equal_bhs

# Configure JAX
configure_jax(platform="cpu", enable_x64=True)

# Define test cases
# (m1, m2, chi1, chi2, case_name)
TEST_CASES = [
    (50.0, 50.0, 0.0, 0.0, "equal_mass_nonspinning"),
    (80.0, 20.0, 0.0, 0.0, "unequal_mass_nonspinning"),
    (50.0, 50.0, 0.5, 0.5, "equal_mass_aligned_spin"),
    (80.0, 20.0, 0.5, 0.2, "unequal_mass_aligned_spin"),
    (80.0, 20.0, 0.5, -0.3, "unequal_mass_antialigned_spin"),
]

MODES = [22, 21, 33, 44, 55]


@pytest.mark.parametrize("m1, m2, chi1, chi2, case_name", TEST_CASES)
def test_amp_phase_comparison(m1, m2, chi1, chi2, case_name):
    print(f"\nRunning test case: {case_name}")

    # Setup parameters
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    print(f"Total mass M: {M}, Symmetric mass ratio eta: {eta}")
    fmin = 20.0
    inclination = 1.0
    distance = 1.0
    fref = 10.0
    phiref = 0.0
    psi = 0.0

    dparams = compute_waveform_params(
        m1, m2, chi1, chi2, distance, inclination, phiref, psi, fref, fmin
    )
    print("Derived parameters computed.")
    print(f"Derived eta: {dparams.eta}, chi1: {dparams.chi1}, chi2: {dparams.chi2}")

    # PhenomXPy setup
    _pwf = pWF(
        eta=float(eta),
        s1=float(chi1),
        s2=float(chi2),
        total_mass=float(M),
        f_min=float(fmin),
        f_ref=float(fref),
        distance=float(distance),
        inclination=float(inclination),
    )

    # Compute coefficients
    tax_phase_coeffs = {}
    tax_amp_coeffs = {}
    xpy_phase_coeffs = {}
    xpy_amp_coeffs = {}

    # Mode 22 first (needed for others)
    time_start = time.time()
    dparams, phase_coeffs_22 = compute_phase_coeffs_22(dparams)
    amp_coeffs_22 = compute_amplitude_coeffs_22(dparams, phase_coeffs_22)
    time_end = time.time()
    print(
        f"Computed Phentax 22 mode coefficients in {time_end - time_start:.4f} seconds."
    )

    tax_phase_coeffs["22"] = phase_coeffs_22
    tax_amp_coeffs["22"] = amp_coeffs_22

    # PhenomXPy 22
    pwf22 = pWFHM(mode=[2, 2], pWF_input=_pwf)
    time_start = time.time()
    pphase22 = pPhase(pwf22)
    pamp22 = pAmp(pwf22, pphase22)
    time_end = time.time()
    print(
        f"Computed PhenomXPy 22 mode coefficients in {time_end - time_start:.4f} seconds."
    )

    xpy_phase_coeffs["22"] = pphase22
    xpy_amp_coeffs["22"] = pamp22

    # Other modes
    odd_m_modes = [21, 33, 55]
    is_equal = check_equal_bhs(m1, m2, chi1, chi2)
    for mode in MODES:
        if mode == 22:
            continue

        l, m = mode // 10, mode % 10

        # Phentax
        amp_coeffs = compute_amplitude_coeffs_hm(dparams, phase_coeffs_22, mode=mode)
        phase_coeffs = compute_phase_coeffs_hm(
            dparams,
            phase_coeffs_22,
            OmegaCutPNAMP=amp_coeffs.omegaCutPNAMP,
            PhiCutPNAMP=amp_coeffs.phiCutPNAMP,
            mode=mode,
        )

        tax_phase_coeffs[str(mode)] = phase_coeffs
        tax_amp_coeffs[str(mode)] = amp_coeffs

        # PhenomXPy
        try:
            pwf_mode = pWFHM(mode=[l, m], pWF_input=_pwf)
            pamp_mode = pAmp(pwf_mode, pphase22)
            pphase_mode = pPhase(
                pwf_mode, pphase22, pamp_mode.omegaCutPNAMP, pamp_mode.phiCutPNAMP
            )

            xpy_phase_coeffs[str(mode)] = pphase_mode
            xpy_amp_coeffs[str(mode)] = pamp_mode
        except Exception as e:
            print(f"PhenomXPy failed for mode {mode}: {e}")
            xpy_phase_coeffs[str(mode)] = None
            xpy_amp_coeffs[str(mode)] = None

    # Plotting
    times = jnp.linspace(-200, 500, 1000)
    times_np = np.asarray(times)

    fig, axs = plt.subplots(3, 1, figsize=(16, 21))

    # Compute 22 phase for reference
    phase22 = imr_phase(times, dparams.eta, tax_phase_coeffs["22"])

    for mode in MODES:
        if is_equal and mode in odd_m_modes:
            print(f"Skipping mode {mode} for equal-mass case.")
            continue
        mode_str = str(mode)

        # Phentax
        if mode == 22:
            _phase_tax = phase22
        else:
            _phase_tax = imr_phase(
                times, dparams.eta, tax_phase_coeffs[mode_str], phase_22=phase22
            )

        _amp_tax = imr_amplitude(
            times, dparams.eta, tax_amp_coeffs[mode_str], tax_phase_coeffs["22"]
        )

        _omega_tax = imr_omega(times, dparams.eta, tax_phase_coeffs[mode_str])

        # PhenomXPy
        if xpy_phase_coeffs[mode_str] is not None:
            time_start = time.time()
            _phase_xpy = xpy_phase_coeffs[mode_str].imr_phase(times_np)
            _amp_xpy = xpy_amp_coeffs[mode_str].imr_amplitude(times_np)
            _omega_xpy = xpy_phase_coeffs[mode_str].imr_omega(times_np)
            time_end = time.time()
            print(
                f"Computed PhenomXPy mode {mode} coefficients in {time_end - time_start:.4f} seconds."
            )
        else:
            _phase_xpy = np.zeros_like(times_np) * np.nan
            _amp_xpy = np.zeros_like(times_np) * np.nan

        # Plot Phase
        axs[0].plot(times, _phase_tax, label=f"TAX {mode}")
        axs[0].plot(times, _phase_xpy, "--", label=f"XPY {mode}")

        # Plot Omega
        axs[1].plot(times, _omega_tax, label=f"TAX {mode}")
        axs[1].plot(times, _omega_xpy, "--", label=f"XPY {mode}")

        # Plot Amplitude
        axs[2].plot(times, jnp.abs(_amp_tax), label=f"TAX {mode}")
        axs[2].plot(times, jnp.abs(_amp_xpy), "--", label=f"XPY {mode}")

        # breakpoint()
        amp_check = (
            np.isclose(jnp.real(_amp_tax), jnp.real(_amp_xpy)).all()
            and np.isclose(jnp.imag(_amp_tax), jnp.imag(_amp_xpy)).all()
        )

        phase_check = np.isclose(_phase_tax, _phase_xpy).all()

        assert amp_check, f"Amplitude mismatch for mode {mode} in case {case_name}"
        assert phase_check, f"Phase mismatch for mode {mode} in case {case_name}"

        print(f"Mode {mode} - Amplitude match: {amp_check}, Phase match: {phase_check}")

    axs[0].set_ylabel("Phase (radians)")
    axs[0].set_title(f"Phase Comparison - {case_name}")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_ylabel("Omega (rad/M)")
    axs[1].set_title(f"Omega Comparison - {case_name}")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_xlabel("Time (M)")
    axs[2].set_ylabel("|Amplitude|")
    axs[2].set_title(f"Amplitude Comparison - {case_name}")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    # Save plot
    plot_dir = Path("tests/plots")
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f"amp_phase_{case_name}.png"
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"Saved plot to {plot_path}")

    # breakpoint()


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__])
