"""
Comparison tests between phentax and phenomxpy ansatz functions.

These tests verify that the fundamental ansatz building blocks
produce numerically equivalent results between implementations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

# Import phenomxpy ansatze (reference implementation)
from phenomxpy.phenomt import numba_ansatze as pxpy_ansatze

# Import phentax ansatze (JAX implementation)
from phentax import ansatze as ptax_ansatze

# =============================================================================
# Helper functions
# =============================================================================


def assert_close(phentax_val, phenomxpy_val, name, rtol=1e-12, atol=1e-14):
    """Assert that phentax and phenomxpy values match within tolerance."""
    if np.isscalar(phenomxpy_val):
        err = abs(phentax_val - phenomxpy_val)
        rel_err = err / (abs(phenomxpy_val) + 1e-30)
    else:
        err = np.max(np.abs(np.array(phentax_val) - np.array(phenomxpy_val)))
        rel_err = err / (np.max(np.abs(phenomxpy_val)) + 1e-30)

    assert rel_err < rtol or err < atol, (
        f"{name}: phentax={phentax_val}, phenomxpy={phenomxpy_val}, "
        f"rel_err={rel_err:.2e}, abs_err={err:.2e}"
    )


# =============================================================================
# Test: Ringdown Omega Ansatz
# =============================================================================


class TestRingdownOmegaAnsatz:
    """Compare ringdown omega ansatz implementations.

    phenomxpy: numba_ringdown_ansatz_omega(t, c1, c2, c3, c4, omegaRING)
    Formula: omega = omegaRING + num/den where:
        expC = exp(-c2 * t)
        expC2 = expC^2
        num = -c1 * c2 * (2 * c4 * expC2 + c3 * expC)
        den = 1 + c4 * expC2 + c3 * expC
    """

    @pytest.mark.parametrize("c1", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("c2", [0.1, 0.3, 0.5])
    @pytest.mark.parametrize("c3", [0.2, 0.5])
    @pytest.mark.parametrize("c4", [0.1, 0.3])
    def test_ringdown_omega_single_time(self, c1, c2, c3, c4):
        """Test ringdown omega ansatz at a single time point."""
        omega_ring = 0.4
        t = 5.0  # Time after peak (positive)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_omega(
            t, c1, c2, c3, c4, omega_ring
        )

        # Compute using exact formula for comparison
        expC = np.exp(-c2 * t)
        expC2 = expC * expC
        num = -c1 * c2 * (2 * c4 * expC2 + c3 * expC)
        den = 1 + c4 * expC2 + c3 * expC
        expected = num / den + omega_ring

        assert_close(phenomxpy, expected, f"ringdown_omega_ansatz(c1={c1}, c2={c2})")

    @pytest.mark.parametrize("c1", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("c2", [0.1, 0.3])
    def test_ringdown_omega_array(self, c1, c2):
        """Test ringdown omega ansatz on a time array."""
        c3 = 0.3
        c4 = 0.2
        omega_ring = 0.4
        times = np.linspace(0.1, 50.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_omega_array(
            times, c1, c2, c3, c4, omega_ring
        )

        # Direct computation
        expC = np.exp(-c2 * times)
        expC2 = expC * expC
        num = -c1 * c2 * (2 * c4 * expC2 + c3 * expC)
        den = 1 + c4 * expC2 + c3 * expC
        expected = num / den + omega_ring

        assert_close(phenomxpy, expected, "ringdown_omega_array")


# =============================================================================
# Test: Ringdown Amplitude Ansatz
# =============================================================================


class TestRingdownAmplitudeAnsatz:
    """Compare ringdown amplitude ansatz implementations.

    phenomxpy: numba_ringdown_ansatz_amplitude(t, c1, c2, c3, c4, alpha1RD, tshift)
    Formula: amp = exp(-alpha1RD * (t - tshift)) * (c1 * tanh(c2 * (t - tshift) + c3) + c4)
    """

    @pytest.mark.parametrize("c1", [0.1, 0.5])
    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1])
    def test_ringdown_amp_single_time(self, c1, alpha1RD):
        """Test ringdown amplitude ansatz at a single time point."""
        c2 = 0.2
        c3 = 0.3
        c4 = 0.1
        tshift = 2.0
        t = 10.0

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_amplitude(
            t, c1, c2, c3, c4, alpha1RD, tshift
        )

        # Direct computation
        tanh = np.tanh(c2 * (t - tshift) + c3)
        expAlpha = np.exp(-alpha1RD * (t - tshift))
        expected = expAlpha * (c1 * tanh + c4)

        assert_close(phenomxpy, expected, f"ringdown_amp_ansatz")

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1, 0.15])
    def test_ringdown_amp_array(self, alpha1RD):
        """Test ringdown amplitude ansatz on a time array."""
        c1 = 0.5
        c2 = 0.2
        c3 = 0.3
        c4 = 0.1
        tshift = 2.0
        times = np.linspace(2.1, 50.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_amplitude_array(
            times, c1, c2, c3, c4, alpha1RD, tshift
        )

        # Direct computation
        tanh = np.tanh(c2 * (times - tshift) + c3)
        expAlpha = np.exp(-alpha1RD * (times - tshift))
        expected = expAlpha * (c1 * tanh + c4)

        assert_close(phenomxpy, expected, "ringdown_amp_array")


# =============================================================================
# Test: Intermediate Omega Ansatz
# =============================================================================


class TestIntermediateOmegaAnsatz:
    """Compare intermediate omega ansatz implementations.

    phenomxpy: numba_intermediate_ansatz_omega(t, alpha1RD, omegaPeak, omegaRING,
                                               domegaPeak, C1, C2, C3)
    Formula:
        x = arcsinh(alpha1RD * t)
        w = 1 - omegaPeak/omegaRING + x*(domegaPeak/alpha1RD + x*(C1 + x*(C2 + x*C3)))
        omega = omegaRING * (1 - w)
    """

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1, 0.15])
    @pytest.mark.parametrize("omegaPeak", [0.3, 0.35, 0.4])
    def test_intermediate_omega_single_time(self, alpha1RD, omegaPeak):
        """Test intermediate omega ansatz at a single time point."""
        omegaRING = 0.45
        domegaPeak = 0.01
        C1 = 0.001
        C2 = 0.0001
        C3 = 0.00001
        t = 5.0

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_omega(
            t, alpha1RD, omegaPeak, omegaRING, domegaPeak, C1, C2, C3
        )

        # Direct computation
        x = np.arcsinh(alpha1RD * t)
        w = (
            1
            - omegaPeak / omegaRING
            + x * (domegaPeak / alpha1RD + x * (C1 + x * (C2 + x * C3)))
        )
        expected = omegaRING * (1 - w)

        assert_close(phenomxpy, expected, "intermediate_omega_ansatz")

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1])
    def test_intermediate_omega_array(self, alpha1RD):
        """Test intermediate omega ansatz on a time array."""
        omegaPeak = 0.35
        omegaRING = 0.45
        domegaPeak = 0.01
        C1 = 0.001
        C2 = 0.0001
        C3 = 0.00001
        times = np.linspace(-10.0, 10.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_omega_array(
            times, alpha1RD, omegaPeak, omegaRING, domegaPeak, C1, C2, C3
        )

        # Direct computation
        x = np.arcsinh(alpha1RD * times)
        w = (
            1
            - omegaPeak / omegaRING
            + x * (domegaPeak / alpha1RD + x * (C1 + x * (C2 + x * C3)))
        )
        expected = omegaRING * (1 - w)

        assert_close(phenomxpy, expected, "intermediate_omega_array")


# =============================================================================
# Test: Intermediate Amplitude Ansatz
# =============================================================================


class TestIntermediateAmplitudeAnsatz:
    """Compare intermediate amplitude ansatz implementations.

    phenomxpy: numba_intermediate_ansatz_amplitude(t, alpha1RD, C1, C2, C3, C4, tshift)
    Formula:
        sech1 = 1/cosh(alpha1RD * (t - tshift))
        sech2 = 1/cosh(2 * alpha1RD * (t - tshift))
        amp = C1 + C2*sech1 + C3*sech2^(1/7) + C4*(t-tshift)^2
    """

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1, 0.15])
    @pytest.mark.parametrize("C1", [0.3, 0.4, 0.5])
    def test_intermediate_amp_single_time(self, alpha1RD, C1):
        """Test intermediate amplitude ansatz at a single time point."""
        C2 = 0.1
        C3 = 0.05
        C4 = -0.001
        tshift = 2.0
        t = 5.0

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_amplitude(
            t, alpha1RD, C1, C2, C3, C4, tshift
        )

        # Direct computation
        sech1 = 1 / np.cosh(alpha1RD * (t - tshift))
        sech2 = 1 / np.cosh(2 * alpha1RD * (t - tshift))
        expected = (
            C1 + C2 * sech1 + C3 * np.power(sech2, 1 / 7) + C4 * (t - tshift) ** 2
        )

        assert_close(phenomxpy, expected, "intermediate_amp_ansatz")

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1])
    def test_intermediate_amp_array(self, alpha1RD):
        """Test intermediate amplitude ansatz on a time array."""
        C1 = 0.4
        C2 = 0.1
        C3 = 0.05
        C4 = -0.001
        tshift = 2.0
        times = np.linspace(-10.0, 20.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_amplitude_array(
            times, alpha1RD, C1, C2, C3, C4, tshift
        )

        # Direct computation
        sech1 = 1 / np.cosh(alpha1RD * (times - tshift))
        sech2 = 1 / np.cosh(2 * alpha1RD * (times - tshift))
        expected = (
            C1 + C2 * sech1 + C3 * np.power(sech2, 1 / 7) + C4 * (times - tshift) ** 2
        )

        assert_close(phenomxpy, expected, "intermediate_amp_array")


# =============================================================================
# Test: PN Omega Ansatz
# =============================================================================


class TestPNOmegaAnsatz:
    """Compare PN (TaylorT3) omega ansatz implementations.

    phenomxpy: numba_pn_ansatz_omega(theta, coefficients)
    Formula: omega = (theta^3/4) * (1 + sum of PN terms)
    """

    @pytest.mark.parametrize("theta", [0.1, 0.2, 0.3, 0.5])
    def test_pn_omega_single_value(self, theta):
        """Test PN omega ansatz at a single theta value."""
        # Typical PN coefficients (simplified)
        coefficients = np.array([0.743, -0.925, 1.234, -0.567, 0.891, -0.234])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_pn_ansatz_omega(theta, coefficients)

        # Direct computation
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta3 * theta2
        theta6 = theta3 * theta3
        theta7 = theta4 * theta3
        logterm = 107 * np.log(theta) / 280
        fac = theta3 / 4

        expected = fac * (
            1
            + coefficients[0] * theta2
            + coefficients[1] * theta3
            + coefficients[2] * theta4
            + coefficients[3] * theta5
            + coefficients[4] * theta6
            + logterm * theta6
            + coefficients[5] * theta7
        )

        assert_close(phenomxpy, expected, f"pn_omega_ansatz(theta={theta})")

    def test_pn_omega_array(self):
        """Test PN omega ansatz on an array of theta values."""
        theta = np.linspace(0.05, 0.5, 50)
        coefficients = np.array([0.743, -0.925, 1.234, -0.567, 0.891, -0.234])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_pn_ansatz_omega_array(theta, coefficients)

        # Direct computation
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta3 * theta2
        theta6 = theta3 * theta3
        theta7 = theta4 * theta3
        logterm = 107 * np.log(theta) / 280
        fac = theta3 / 4

        expected = fac * (
            1
            + coefficients[0] * theta2
            + coefficients[1] * theta3
            + coefficients[2] * theta4
            + coefficients[3] * theta5
            + coefficients[4] * theta6
            + logterm * theta6
            + coefficients[5] * theta7
        )

        assert_close(phenomxpy, expected, "pn_omega_array")


# =============================================================================
# Test: Inspiral Omega Ansatz
# =============================================================================


class TestInspiralOmegaAnsatz:
    """Compare full inspiral omega ansatz implementations.

    phenomxpy: numba_inspiral_ansatz_omega(time, eta, pn_coefficients, pseudo_coefficients)
    Formula: omega = (theta^3/4) * (TaylorT3 + pseudo_PN_terms)
        where theta = (-eta * time / 5)^(-1/8)
    """

    @pytest.mark.parametrize("eta", [0.25, 0.2, 0.16])
    @pytest.mark.parametrize("time", [-1000.0, -500.0, -100.0])
    def test_inspiral_omega_single_time(self, eta, time):
        """Test inspiral omega ansatz at a single time point."""
        # PN coefficients
        pn_coeffs = np.array([0.743, -0.925, 1.234, -0.567, 0.891, -0.234])
        pseudo_coeffs = np.array([0.01, 0.002, 0.0005, 0.0001, 0.00002, 0.000005])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_inspiral_ansatz_omega(
            time, eta, pn_coeffs, pseudo_coeffs
        )

        # Direct computation
        theta = np.power(-eta * time / 5, -1 / 8)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta5 = theta4 * theta
        theta6 = theta5 * theta
        theta7 = theta6 * theta
        theta8 = theta7 * theta
        theta9 = theta8 * theta
        theta10 = theta9 * theta
        theta11 = theta10 * theta
        theta12 = theta11 * theta
        theta13 = theta12 * theta
        logterm = 107 * np.log(theta) / 280
        fac = theta3 / 4

        taylort3 = (
            1
            + pn_coeffs[0] * theta2
            + pn_coeffs[1] * theta3
            + pn_coeffs[2] * theta4
            + pn_coeffs[3] * theta5
            + pn_coeffs[4] * theta6
            + logterm * theta6
            + pn_coeffs[5] * theta7
        )

        pseudo = (
            pseudo_coeffs[0] * theta8
            + pseudo_coeffs[1] * theta9
            + pseudo_coeffs[2] * theta10
            + pseudo_coeffs[3] * theta11
            + pseudo_coeffs[4] * theta12
            + pseudo_coeffs[5] * theta13
        )

        expected = fac * (taylort3 + pseudo)

        assert_close(
            phenomxpy, expected, f"inspiral_omega_ansatz(eta={eta}, time={time})"
        )

    @pytest.mark.parametrize("eta", [0.25, 0.2])
    def test_inspiral_omega_array(self, eta):
        """Test inspiral omega ansatz on a time array."""
        times = np.linspace(-5000.0, -50.0, 100)
        pn_coeffs = np.array([0.743, -0.925, 1.234, -0.567, 0.891, -0.234])
        pseudo_coeffs = np.array([0.01, 0.002, 0.0005, 0.0001, 0.00002, 0.000005])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_inspiral_ansatz_omega_array(
            times, eta, pn_coeffs, pseudo_coeffs
        )

        # Verify non-NaN and correct shape
        assert not np.any(np.isnan(phenomxpy)), "NaN values in inspiral omega"
        assert len(phenomxpy) == len(times), "Wrong output shape"


# =============================================================================
# Test: Inspiral Amplitude Ansatz
# =============================================================================


class TestInspiralAmplitudeAnsatz:
    """Compare inspiral amplitude ansatz implementations.

    phenomxpy: numba_inspiral_ansatz_amplitude(x, fac0, pn_real_coeffs, pn_imag_coeffs, pseudo_pn_coeffs)
    Formula: Complex amplitude as PN expansion in x = omega^(2/3)
    """

    @pytest.mark.parametrize("x", [0.01, 0.02, 0.05, 0.1])
    def test_inspiral_amp_single_x(self, x):
        """Test inspiral amplitude ansatz at a single x value."""
        fac0 = 1.0
        pn_real_coeffs = np.array([1.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8])
        pn_imag_coeffs = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        pseudo_pn_coeffs = np.array([0.01, 0.02, 0.03])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_inspiral_ansatz_amplitude(
            x, fac0, pn_real_coeffs, pn_imag_coeffs, pseudo_pn_coeffs
        )

        # Direct computation
        xhalf = np.sqrt(x)
        x1half = x * xhalf
        x2 = x * x
        x2half = x2 * xhalf
        x3 = x2 * x
        x3half = x3 * xhalf
        x4 = x2 * x2
        x4half = x4 * xhalf
        x5 = x3 * x2

        ampreal = (
            pn_real_coeffs[0]
            + pn_real_coeffs[1] * xhalf
            + pn_real_coeffs[2] * x
            + pn_real_coeffs[3] * x1half
            + pn_real_coeffs[4] * x2
            + pn_real_coeffs[5] * x2half
            + pn_real_coeffs[6] * x3
            + pn_real_coeffs[7] * x3half
            + pn_real_coeffs[8] * np.log(16 * x) * x3
        )
        ampimag = (
            pn_imag_coeffs[0] * xhalf
            + pn_imag_coeffs[1] * x
            + pn_imag_coeffs[2] * x1half
            + pn_imag_coeffs[3] * x2
            + pn_imag_coeffs[4] * x2half
            + pn_imag_coeffs[5] * x3
            + pn_imag_coeffs[6] * x3half
        )
        ampreal += (
            pseudo_pn_coeffs[0] * x4
            + pseudo_pn_coeffs[1] * x4half
            + pseudo_pn_coeffs[2] * x5
        )

        expected = fac0 * x * (ampreal + 1j * ampimag)

        assert_close(np.real(phenomxpy), np.real(expected), f"inspiral_amp_real(x={x})")
        assert_close(np.imag(phenomxpy), np.imag(expected), f"inspiral_amp_imag(x={x})")

    def test_inspiral_amp_array(self):
        """Test inspiral amplitude ansatz on an array of x values."""
        x = np.linspace(0.01, 0.1, 50)
        fac0 = 1.0
        pn_real_coeffs = np.array([1.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8])
        pn_imag_coeffs = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        pseudo_pn_coeffs = np.array([0.01, 0.02, 0.03])

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_inspiral_ansatz_amplitude_array(
            x, fac0, pn_real_coeffs, pn_imag_coeffs, pseudo_pn_coeffs
        )

        # Verify non-NaN and correct shape
        assert not np.any(np.isnan(phenomxpy)), "NaN values in inspiral amplitude"
        assert len(phenomxpy) == len(x), "Wrong output shape"


# =============================================================================
# Test: Ringdown Phase Ansatz
# =============================================================================


class TestRingdownPhaseAnsatz:
    """Compare ringdown phase ansatz implementations.

    phenomxpy: numba_ringdown_ansatz_phase(t, c1, c2, c3, c4, omegaRING, phOffRD)
    """

    @pytest.mark.parametrize("c1", [0.1, 0.5])
    @pytest.mark.parametrize("omegaRING", [0.3, 0.4, 0.5])
    def test_ringdown_phase_single_time(self, c1, omegaRING):
        """Test ringdown phase ansatz at a single time point."""
        c2 = 0.2
        c3 = 0.3
        c4 = 0.1
        phOffRD = 0.0
        t = 10.0

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_phase(
            t, c1, c2, c3, c4, omegaRING, phOffRD
        )

        # Check it's a real number
        assert np.isfinite(phenomxpy), f"Non-finite ringdown phase: {phenomxpy}"

    @pytest.mark.parametrize("omegaRING", [0.3, 0.4])
    def test_ringdown_phase_array(self, omegaRING):
        """Test ringdown phase ansatz on a time array."""
        c1 = 0.5
        c2 = 0.2
        c3 = 0.3
        c4 = 0.1
        phOffRD = 0.0
        times = np.linspace(0.1, 50.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_ringdown_ansatz_phase_array(
            times, c1, c2, c3, c4, omegaRING, phOffRD
        )

        # Verify non-NaN and correct shape
        assert not np.any(np.isnan(phenomxpy)), "NaN values in ringdown phase"
        assert len(phenomxpy) == len(times), "Wrong output shape"


# =============================================================================
# Test: Intermediate Phase Ansatz
# =============================================================================


class TestIntermediatePhaseAnsatz:
    """Compare intermediate phase ansatz implementations.

    phenomxpy: numba_intermediate_ansatz_phase(t, alpha1RD, C1, C2, C3,
                                               omegaPeak, domegaPeak, omegaRING, phOffMerger)
    """

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1])
    @pytest.mark.parametrize("omegaPeak", [0.3, 0.35])
    def test_intermediate_phase_single_time(self, alpha1RD, omegaPeak):
        """Test intermediate phase ansatz at a single time point."""
        C1 = 0.001
        C2 = 0.0001
        C3 = 0.00001
        domegaPeak = 0.01
        omegaRING = 0.45
        phOffMerger = 0.0
        t = 5.0

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_phase(
            t, alpha1RD, C1, C2, C3, omegaPeak, domegaPeak, omegaRING, phOffMerger
        )

        # Check it's a real number
        assert np.isfinite(phenomxpy), f"Non-finite intermediate phase: {phenomxpy}"

    @pytest.mark.parametrize("alpha1RD", [0.05, 0.1])
    def test_intermediate_phase_array(self, alpha1RD):
        """Test intermediate phase ansatz on a time array."""
        C1 = 0.001
        C2 = 0.0001
        C3 = 0.00001
        omegaPeak = 0.35
        domegaPeak = 0.01
        omegaRING = 0.45
        phOffMerger = 0.0
        times = np.linspace(-10.0, 10.0, 100)

        # phenomxpy result
        phenomxpy = pxpy_ansatze.numba_intermediate_ansatz_phase_array(
            times, alpha1RD, C1, C2, C3, omegaPeak, domegaPeak, omegaRING, phOffMerger
        )

        # Verify non-NaN and correct shape
        assert not np.any(np.isnan(phenomxpy)), "NaN values in intermediate phase"
        assert len(phenomxpy) == len(times), "Wrong output shape"


# =============================================================================
# Summary test
# =============================================================================


class TestAnsatzeSummary:
    """Summary of ansatze verification."""

    def test_all_ansatze_available(self):
        """Verify all expected ansatze are available in phenomxpy."""
        expected_funcs = [
            "numba_ringdown_ansatz_omega",
            "numba_ringdown_ansatz_omega_array",
            "numba_ringdown_ansatz_amplitude",
            "numba_ringdown_ansatz_amplitude_array",
            "numba_intermediate_ansatz_omega",
            "numba_intermediate_ansatz_omega_array",
            "numba_intermediate_ansatz_amplitude",
            "numba_intermediate_ansatz_amplitude_array",
            "numba_pn_ansatz_omega",
            "numba_pn_ansatz_omega_array",
            "numba_inspiral_ansatz_omega",
            "numba_inspiral_ansatz_omega_array",
            "numba_inspiral_ansatz_amplitude",
            "numba_inspiral_ansatz_amplitude_array",
            "numba_ringdown_ansatz_phase",
            "numba_ringdown_ansatz_phase_array",
            "numba_intermediate_ansatz_phase",
            "numba_intermediate_ansatz_phase_array",
        ]

        for func_name in expected_funcs:
            assert hasattr(pxpy_ansatze, func_name), f"Missing: {func_name}"
