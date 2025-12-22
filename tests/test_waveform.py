import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from phenomxpy.phenomt.internals import pWF
from phenomxpy.phenomt.phenomt import IMRPhenomTHM as xpy_thm

from phentax.waveform import IMRPhenomTHM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_CASES = [
    (50.0, 40.0, 20, 1.0 / 4096.0, "ligo_like"),
    (5e6, 1e6, 1e-4, 2.5, "lisa_like"),
]


@pytest.mark.parametrize("m1, m2, f_min, delta_t, case_name", TEST_CASES)
def test_waveform_comparison(m1, m2, f_min, delta_t, case_name):
    print(f"\nRunning test case: {case_name}")
    chi1 = 0.9
    chi2 = 0.3
    distance = 500.0
    inclination = jnp.pi / 3.0
    phi_ref = 0.0
    psi = 0.0
    f_ref = f_min
    # t_ref = 0.0

    tlowfit = True
    tol = 1e-12

    imr = IMRPhenomTHM(
        higher_modes="all",
        include_negative_modes=True,
        t_low_fit=tlowfit,
        coarse_grain=False,
        atol=tol,
        rtol=tol,
    )
    mode_array = None  # [[2,2], [2,1], [3,3], [4,4]]

    st = time.time()
    times, mask, h_plus, h_cross = imr.compute_polarizations_at_once(
        m1,
        m2,
        chi1,
        chi2,
        distance,
        phi_ref,
        f_ref,
        f_min,
        inclination,
        psi,
        delta_t=delta_t,
    )
    logger.info(f"PHENTAX polarizations computed in {time.time() - st} WARMUP seconds")

    st = time.time()
    times, mask, h_plus, h_cross = imr.compute_polarizations_at_once(
        m1,
        m2,
        chi1,
        chi2,
        distance,
        phi_ref,
        f_ref,
        f_min,
        inclination,
        psi,
        delta_t=delta_t,
    )
    logger.info(f"PHENTAX polarizations computed in {time.time() - st} seconds")

    st = time.time()
    pwf = pWF(
        eta=m1 * m2 / (m1 + m2) ** 2,
        s1=chi1,
        s2=chi2,
        f_min=f_min,
        f_ref=f_ref,
        total_mass=m1 + m2,
        distance=distance,
        inclination=inclination,
        polarization_angle=psi,
        delta_t=delta_t,
        phi_ref=phi_ref,
    )

    xpy_wave_gen = xpy_thm(mode_array=mode_array, pWF_input=pwf)
    logger.info(f"XPY waveform generator created in {time.time() - st} seconds")

    xpy_plus, xpy_cross, xpy_times = xpy_wave_gen.compute_polarizations()

    st = time.time()
    xpy_plus, xpy_cross, xpy_times = xpy_wave_gen.compute_polarizations()
    logger.info(f"XPY polarizations computed in {time.time() - st} seconds")

    # plt.figure(); plt.plot(times[mask], h_plus[mask].real); plt.plot(times[mask], h_cross[mask].real); plt.title("PHENTAX"); plt.xlabel("time [s]"); plt.show()
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(16, 12))
    axs[0, 0].plot(times[mask], h_plus[mask], label="PHENTAX")
    axs[0, 0].plot(xpy_times, xpy_plus, ls="--", label="PHENOMXPY")
    axs[0, 0].legend()
    axs[0, 0].set_title("H plus")

    axs[0, 1].plot(times[mask], h_cross[mask], label="PHENTAX")
    axs[0, 1].plot(xpy_times, xpy_cross, ls="--", label="PHENOMXPY")
    axs[0, 1].legend()
    axs[0, 1].set_title("H cross")

    from scipy.interpolate import CubicSpline as _CubicSpline

    _spline = _CubicSpline(xpy_times, xpy_plus)
    xpy_plus_interp = _spline(np.asarray(times[mask]))
    xpy_cross_interp = _spline(np.asarray(times[mask]))

    # difference plot
    axs[1, 0].plot(
        times[mask],
        jnp.abs(h_plus[mask] - xpy_plus_interp) / jnp.abs(h_plus[mask]),
    )
    axs[1, 0].set_title("Relative difference H plus")
    axs[1, 0].set_xlabel("time [s]")

    axs[1, 1].plot(
        times[mask],
        jnp.abs(h_cross[mask] - xpy_cross_interp) / jnp.abs(h_cross[mask]),
    )
    axs[1, 1].set_title("Relative difference H cross")
    axs[1, 1].set_xlabel("time [s]")

    plt.tight_layout()
    # find the location of the plots directory with respect to this file
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f"waveform_comparison_{case_name}.png")

    isclose = jnp.allclose(h_plus[mask], xpy_plus_interp, rtol=1e-5, atol=1e-5)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-5"

    isclose = jnp.allclose(h_plus[mask], xpy_plus_interp, rtol=1e-7, atol=1e-7)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-7"

    isclose = jnp.allclose(h_plus[mask], xpy_plus_interp, rtol=1e-12, atol=1e-12)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-12"

    print("==" * 10)

    isclose = jnp.allclose(h_cross[mask], xpy_cross_interp, rtol=1e-5, atol=1e-5)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-5"

    isclose = jnp.allclose(h_cross[mask], xpy_cross_interp, rtol=1e-7, atol=1e-7)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-7"

    isclose = jnp.allclose(h_cross[mask], xpy_cross_interp, rtol=1e-12, atol=1e-12)
    assert isclose, f"Waveform mismatch for case {case_name} at tolerance 1e-12"
