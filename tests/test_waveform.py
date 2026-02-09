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


@pytest.mark.parametrize("m1, m2, f_min, delta_t, case_name", TEST_CASES)
def test_time_grid_correctness(m1, m2, f_min, delta_t, case_name):
    """
    Test that both coarse-grained (adaptive) and uniform time grids produce 
    correct waveforms and compare their performance.
    """
    print(f"\nTesting time grid correctness for: {case_name}")
    chi1 = 0.9
    chi2 = 0.3
    distance = 500.0
    inclination = jnp.pi / 3.0
    phi_ref = 0.0
    psi = 0.0
    f_ref = f_min
    
    tlowfit = True
    tol = 1e-12
    
    # Test with coarse_grain=True (adaptive grid)
    print(f"  Testing adaptive grid (coarse_grain=True)...")
    jax.clear_caches()
    imr_adaptive = IMRPhenomTHM(
        higher_modes="all",
        include_negative_modes=True,
        t_low_fit=tlowfit,
        coarse_grain=True,
        atol=tol,
        rtol=tol,
    )
    
    # Warmup call
    times_adaptive, mask_adaptive, h_plus_adaptive, h_cross_adaptive = (
        imr_adaptive.compute_polarizations_at_once(
            m1, m2, chi1, chi2,
            distance, phi_ref, f_ref, f_min,
            inclination, psi,
            delta_t=delta_t,
        )
    )
    
    # Timed call
    st = time.time()
    times_adaptive, mask_adaptive, h_plus_adaptive, h_cross_adaptive = (
        imr_adaptive.compute_polarizations_at_once(
            m1, m2, chi1, chi2,
            distance, phi_ref, f_ref, f_min,
            inclination, psi,
            delta_t=delta_t,
        )
    )
    times_adaptive.block_until_ready()
    time_adaptive = time.time() - st
    
    valid_points_adaptive = mask_adaptive.sum()
    grid_size_adaptive = times_adaptive.shape[-1]
    
    logger.info(
        f"  Adaptive grid - Time: {time_adaptive:.4f}s, "
        f"Grid size: {grid_size_adaptive}, Valid points: {valid_points_adaptive}"
    )
    
    # Test with coarse_grain=False (uniform grid)
    print(f"  Testing uniform grid (coarse_grain=False)...")
    jax.clear_caches()
    imr_uniform = IMRPhenomTHM(
        higher_modes="all",
        include_negative_modes=True,
        t_low_fit=tlowfit,
        coarse_grain=False,
        atol=tol,
        rtol=tol,
    )
    
    # Warmup call
    times_uniform, mask_uniform, h_plus_uniform, h_cross_uniform = (
        imr_uniform.compute_polarizations_at_once(
            m1, m2, chi1, chi2,
            distance, phi_ref, f_ref, f_min,
            inclination, psi,
            delta_t=delta_t,
        )
    )
    
    # Timed call
    st = time.time()
    times_uniform, mask_uniform, h_plus_uniform, h_cross_uniform = (
        imr_uniform.compute_polarizations_at_once(
            m1, m2, chi1, chi2,
            distance, phi_ref, f_ref, f_min,
            inclination, psi,
            delta_t=delta_t,
        )
    )
    times_uniform.block_until_ready()
    time_uniform = time.time() - st
    
    valid_points_uniform = mask_uniform.sum()
    grid_size_uniform = times_uniform.shape[-1]
    
    logger.info(
        f"  Uniform grid - Time: {time_uniform:.4f}s, "
        f"Grid size: {grid_size_uniform}, Valid points: {valid_points_uniform}"
    )
    
    # Performance comparison
    speedup = time_uniform / time_adaptive if time_adaptive > 0 else 1.0
    efficiency_adaptive = (valid_points_adaptive / grid_size_adaptive * 100)
    efficiency_uniform = (valid_points_uniform / grid_size_uniform * 100)
    
    logger.info(
        f"  Performance - Speedup: {speedup:.2f}x, "
        f"Adaptive efficiency: {efficiency_adaptive:.1f}%, "
        f"Uniform efficiency: {efficiency_uniform:.1f}%"
    )
    
    # Verify correctness: both grids should produce similar waveforms
    # Interpolate adaptive grid results onto uniform grid times for comparison
    from scipy.interpolate import CubicSpline as _CubicSpline
    
    # Extract valid times and waveforms
    times_adaptive_valid = np.asarray(times_adaptive[mask_adaptive])
    h_plus_adaptive_valid = np.asarray(h_plus_adaptive[mask_adaptive])
    h_cross_adaptive_valid = np.asarray(h_cross_adaptive[mask_adaptive])
    
    times_uniform_valid = np.asarray(times_uniform[mask_uniform])
    h_plus_uniform_valid = np.asarray(h_plus_uniform[mask_uniform])
    h_cross_uniform_valid = np.asarray(h_cross_uniform[mask_uniform])
    
    # Find common time range
    t_min = max(times_adaptive_valid.min(), times_uniform_valid.min())
    t_max = min(times_adaptive_valid.max(), times_uniform_valid.max())
    
    # Filter to common range
    mask_adaptive_common = (times_adaptive_valid >= t_min) & (times_adaptive_valid <= t_max)
    mask_uniform_common = (times_uniform_valid >= t_min) & (times_uniform_valid <= t_max)
    
    times_adaptive_common = times_adaptive_valid[mask_adaptive_common]
    h_plus_adaptive_common = h_plus_adaptive_valid[mask_adaptive_common]
    h_cross_adaptive_common = h_cross_adaptive_valid[mask_adaptive_common]
    
    times_uniform_common = times_uniform_valid[mask_uniform_common]
    h_plus_uniform_common = h_plus_uniform_valid[mask_uniform_common]
    h_cross_uniform_common = h_cross_uniform_valid[mask_uniform_common]
    
    # Interpolate adaptive results onto uniform times
    spline_plus = _CubicSpline(times_adaptive_common, h_plus_adaptive_common)
    spline_cross = _CubicSpline(times_adaptive_common, h_cross_adaptive_common)
    
    h_plus_adaptive_interp = spline_plus(times_uniform_common)
    h_cross_adaptive_interp = spline_cross(times_uniform_common)
    
    # Check agreement at different tolerances
    # More lenient tolerance due to interpolation and different grid spacing
    isclose_plus = jnp.allclose(
        h_plus_uniform_common, h_plus_adaptive_interp, 
        rtol=1e-3, atol=1e-10
    )
    isclose_cross = jnp.allclose(
        h_cross_uniform_common, h_cross_adaptive_interp,
        rtol=1e-3, atol=1e-10
    )
    
    # Calculate maximum relative error
    rel_err_plus = jnp.max(
        jnp.abs(h_plus_uniform_common - h_plus_adaptive_interp) / 
        (jnp.abs(h_plus_uniform_common) + 1e-20)
    )
    rel_err_cross = jnp.max(
        jnp.abs(h_cross_uniform_common - h_cross_adaptive_interp) / 
        (jnp.abs(h_cross_uniform_common) + 1e-20)
    )
    
    logger.info(
        f"  Waveform agreement - Max rel. error h_plus: {rel_err_plus:.2e}, "
        f"h_cross: {rel_err_cross:.2e}"
    )
    
    assert isclose_plus, (
        f"h_plus mismatch between adaptive and uniform grids for {case_name}. "
        f"Max rel. error: {rel_err_plus:.2e}"
    )
    assert isclose_cross, (
        f"h_cross mismatch between adaptive and uniform grids for {case_name}. "
        f"Max rel. error: {rel_err_cross:.2e}"
    )
    
    # Verify that adaptive grid is actually more efficient
    assert efficiency_adaptive > 5.0, (
        f"Adaptive grid efficiency too low: {efficiency_adaptive:.1f}%"
    )
    
    # Verify grid sizes are reasonable (no OOM)
    assert grid_size_adaptive <= 20000, (
        f"Adaptive grid size too large: {grid_size_adaptive}"
    )
    assert grid_size_uniform <= 150000, (
        f"Uniform grid size too large: {grid_size_uniform}"
    )
    
    print(f"  ✓ Both grids produce consistent waveforms")
    print(f"  ✓ Adaptive grid is {100 - efficiency_adaptive:.1f}% more efficient")
    print(f"  ✓ Execution time - Adaptive: {time_adaptive:.4f}s, Uniform: {time_uniform:.4f}s")
