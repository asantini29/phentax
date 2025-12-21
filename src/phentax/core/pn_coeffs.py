# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Post-Newtonian (PN) coefficients for IMRPhenomT(HM).

Contains the TaylorT3 omega PN coefficients and amplitude PN coefficients
for all supported modes. These are mode-dependent and spin-dependent.

References:
- TaylorT3 omega: Eq. A5 in arXiv:2012.11923 (IMRPhenomT paper)
- Amplitude PN: Eq. 9.4 Blanchet 2008, Eq. 43 Faye 2012, Eq. 4.17 Arun, Eq. 4.27 Buonanno
"""


import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class OmegaPNCoeffs(eqx.Module):
    """TaylorT3 omega PN coefficients (up to 3.5PN)."""

    omega1PN: float | Array
    omega1halfPN: float | Array
    omega2PN: float | Array
    omega2halfPN: float | Array
    omega3PN: float | Array
    omega3halfPN: float | Array


class AmpPNCoeffs(eqx.Module):
    """Amplitude PN coefficients for a specific mode."""

    # Real part coefficients
    ampN: float | Array  # Leading order
    amp0halfPNreal: float | Array
    amp1PNreal: float | Array
    amp1halfPNreal: float | Array
    amp2PNreal: float | Array
    amp2halfPNreal: float | Array
    amp3PNreal: float | Array
    amp3halfPNreal: float | Array
    amplog: float | Array  # Log term coefficient
    # Imaginary part coefficients
    amp0halfPNimag: float | Array
    amp1PNimag: float | Array
    amp1halfPNimag: float | Array
    amp2PNimag: float | Array
    amp2halfPNimag: float | Array
    amp3PNimag: float | Array
    amp3halfPNimag: float | Array
    # Prefactor
    fac0: float | Array


@jax.jit
def compute_omega_pn_coeffs(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> OmegaPNCoeffs:
    """
    Compute TaylorT3 omega PN coefficients.

    Eq. A5 in arXiv:2012.11923. Paper misses the term eta^3 * 235925/1769472 at 3PN order.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    chi1, chi2 : Array
        Dimensionless spin z-components.
    delta : Array
        Mass difference ratio (m1-m2)/M.
    m1, m2 : Array
        Component masses as fractions of total mass (m1+m2=1).

    Returns
    -------
    OmegaPNCoeffs
        PN coefficients for TaylorT3 omega.
    """
    eta2 = eta * eta
    eta3 = eta * eta2

    chi12 = chi1 * chi1
    chi22 = chi2 * chi2
    chi23 = chi2 * chi22

    omega1PN = 743 / 2688 + (11 * eta) / 32

    omega1halfPN = (-19 * (chi1 + chi2) * eta) / 80 + (
        -113 * (-2 * chi1 * m1 - 2 * chi2 * m2) - 96 * jnp.pi
    ) / 320

    omega2PN = (
        ((56975 + 61236 * chi12 - 119448 * chi1 * chi2 + 61236 * chi22) * eta) / 258048
        + (371 * eta2) / 2048
        + (1855099 - 3429216 * chi12 * m1 - 3429216 * chi22 * m2) / 14450688
    )

    omega2halfPN = (
        (-17 * (chi1 + chi2) * eta2) / 128
        + (-146597 * (-2 * chi1 * m1 - 2 * chi2 * m2) - 46374 * jnp.pi) / 129024
        + (
            eta
            * (
                -2 * (chi1 * (1213 - 63 * delta) + chi2 * (1213 + 63 * delta))
                + 117 * jnp.pi
            )
        )
        / 2304
    )
    omega3PN = (
        -720817631400877 / 288412611379200
        - (16928263 * chi12) / 137625600
        - (16928263 * chi22) / 137625600
        - (16928263 * chi12 * delta) / 137625600
        + (16928263 * chi22 * delta) / 137625600
        + (
            (-2318475 + 18767224 * chi12 - 54663952 * chi1 * chi2 + 18767224 * chi22)
            * eta2
        )
        / 137625600
        + (235925 * eta3) / 1769472
        + (107 * jnp.euler_gamma) / 280
        - (6127 * chi1 * jnp.pi) / 12800
        - (6127 * chi2 * jnp.pi) / 12800
        - (6127 * chi1 * delta * jnp.pi) / 12800
        + (6127 * chi2 * delta * jnp.pi) / 12800
        + (
            eta
            * (
                632550449425
                + 35200873512 * chi12
                - 28527282000 * chi1 * chi2
                + 9605339856 * chi12 * delta
                - 1512 * chi22 * (-23281001 + 6352738 * delta)
                + 34172264448 * (chi1 + chi2) * jnp.pi
                - 22912243200 * jnp.pi**2
            )
        )
        / 104044953600
        + (53 * jnp.pi**2) / 200
        + (107 * jnp.log(2)) / 280
    )
    omega3halfPN = (
        (-12029 * (chi1 + chi2) * eta3) / 92160
        + (
            eta2
            * (
                507654 * chi1 * chi22
                - 838782 * chi23
                + chi2 * (-840149 + 507654 * chi12 - 870576 * delta)
                + chi1 * (-840149 - 838782 * chi12 + 870576 * delta)
                + 1701228 * jnp.pi
            )
        )
        / 15482880
        + (
            eta
            * (
                -1134 * chi23 * (-206917 + 71931 * delta)
                + chi1
                * (
                    -1496368361
                    - 429508815 * delta
                    + 1134 * chi12 * (206917 + 71931 * delta)
                )
                - chi2 * (1496368361 - 429508815 * delta + 437064012 * chi12 * m1)
                - 437064012 * chi1 * chi22 * m2
                - 144
                * (488825 + 923076 * chi12 - 1782648 * chi1 * chi2 + 923076 * chi22)
                * jnp.pi
            )
        )
        / 185794560
        + (
            -2 * chi1 * (-6579635551 + 535759434 * chi12) * m1
            + 13159271102 * chi2 * m2
            - 1071518868 * chi23 * m2
            + (-565550067 + 930460608 * chi12 * m1 + 930460608 * chi22 * m2) * jnp.pi
        )
        / 1300561920
    )

    return OmegaPNCoeffs(
        omega1PN=omega1PN,
        omega1halfPN=omega1halfPN,
        omega2PN=omega2PN,
        omega2halfPN=omega2halfPN,
        omega3PN=omega3PN,
        omega3halfPN=omega3halfPN,
    )


@jax.jit
def compute_amp_pn_coeffs_22(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> AmpPNCoeffs:
    """
    Compute amplitude PN coefficients for the (2,2) mode.

    3PN non-spinning from Eq 9.4 Blanchet 2008.
    3.5PN non-spinning from Eq. 43 Faye 2012.
    1.5PN spinning from Eq 4.17 Arun.
    2PN spinning from Eq. 4.27 Buonanno.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    chi1, chi2 : Array
        Dimensionless spin z-components.
    delta : Array
        Mass difference ratio (m1-m2)/M.
    m1, m2 : Array
        Component masses as fractions of total mass (m1+m2=1).

    Returns
    -------
    AmpPNCoeffs
        Amplitude PN coefficients for the 22 mode.
    """
    eta2 = eta * eta
    eta3 = eta * eta2

    # Spin combinations
    chis = 0.5 * (chi1 + chi2)
    chia = 0.5 * (chi1 - chi2)
    S0 = m1 * chi1 + m2 * chi2

    # Euler's constant
    euler_gamma = 0.5772156649015329

    # Prefactor
    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    # PN coefficients for 22 mode
    ampN = 1.0
    amp0halfPNreal = 0.0
    amp0halfPNimag = 0.0
    amp1PNreal = -107.0 / 42.0 + (55.0 * eta) / 42.0
    amp1PNimag = 0.0
    amp1halfPNreal = (
        (-4.0 * chis) / 3.0
        - (4.0 * chia * delta) / 3.0
        + (4.0 * chis * eta) / 3.0
        + 2.0 * jnp.pi
    )
    amp1halfPNimag = 0.0
    amp2PNreal = (
        -2173.0 / 1512.0 - (1069.0 * eta) / 216.0 + (2047.0 * eta2) / 1512.0 + S0 * S0
    )
    amp2PNimag = 0.0
    amp2halfPNreal = (-107.0 * jnp.pi) / 21.0 + (34.0 * eta * jnp.pi) / 21.0
    amp2halfPNimag = -24.0 * eta
    amp3PNreal = (
        27027409.0 / 646800.0
        - (278185.0 * eta) / 33264.0
        - (20261.0 * eta2) / 2772.0
        + (114635.0 * eta3) / 99792.0
        - (856.0 * euler_gamma) / 105.0
        + (2.0 * jnp.pi * jnp.pi) / 3.0
        + (41.0 * eta * jnp.pi * jnp.pi) / 96.0
    )
    amp3PNimag = (428.0 * jnp.pi) / 105.0
    amp3halfPNreal = (
        (-2173.0 * jnp.pi) / 756.0
        - (2495.0 * eta * jnp.pi) / 378.0
        + (40.0 * eta2 * jnp.pi) / 27.0
    )
    amp3halfPNimag = (14333.0 * eta) / 162.0 - (4066.0 * eta2) / 945.0
    amplog = -428.0 / 105.0

    return AmpPNCoeffs(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp1PNreal=amp1PNreal,
        amp1halfPNreal=amp1halfPNreal,
        amp2PNreal=amp2PNreal,
        amp2halfPNreal=amp2halfPNreal,
        amp3PNreal=amp3PNreal,
        amp3halfPNreal=amp3halfPNreal,
        amplog=amplog,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNimag=amp1PNimag,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNimag=amp2PNimag,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNimag=amp3PNimag,
        amp3halfPNimag=amp3halfPNimag,
        fac0=fac0,
    )


@jax.jit
def compute_amp_pn_coeffs_21(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> AmpPNCoeffs:
    """Compute amplitude PN coefficients for the (2,1) mode."""
    # Spin combinations
    chia = 0.5 * (chi1 - chi2)
    chis = 0.5 * (chi1 + chi2)
    Sc = m1 * m1 * chi1 + m2 * m2 * chi2
    Sigmac = m2 * chi2 - m1 * chi1

    eta2 = eta * eta

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    ampN = 0.0
    amp0halfPNreal = delta / 3.0
    amp0halfPNimag = 0.0
    amp1PNreal = -0.5 * chia - (chis * delta) / 2.0
    amp1PNimag = 0.0
    amp1halfPNreal = (-17.0 * delta) / 84.0 + (5.0 * delta * eta) / 21.0
    amp1halfPNimag = 0.0
    amp2PNreal = (
        (delta * jnp.pi) / 3.0
        - (43.0 * delta * Sc) / 21.0
        - (79.0 * Sigmac) / 42.0
        + (139.0 * eta * Sigmac) / 42.0
    )
    amp2PNimag = -delta / 6.0 - (delta * jnp.log(16.0)) / 6.0
    amp2halfPNreal = (
        (-43.0 * delta) / 378.0
        - (509.0 * delta * eta) / 378.0
        + (79.0 * delta * eta2) / 504.0
    )
    amp2halfPNimag = 0.0
    amp3PNreal = (-17.0 * delta * jnp.pi) / 84.0 + (delta * eta * jnp.pi) / 14.0
    amp3PNimag = (
        (17.0 * delta) / 168.0
        - (353.0 * delta * eta) / 84.0
        + (17.0 * delta * jnp.log(16.0)) / 168.0
        - (delta * eta * jnp.log(4096.0)) / 84.0
    )
    amp3halfPNreal = 0.0
    amp3halfPNimag = 0.0
    amplog = 0.0

    return AmpPNCoeffs(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp1PNreal=amp1PNreal,
        amp1halfPNreal=amp1halfPNreal,
        amp2PNreal=amp2PNreal,
        amp2halfPNreal=amp2halfPNreal,
        amp3PNreal=amp3PNreal,
        amp3halfPNreal=amp3halfPNreal,
        amplog=amplog,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNimag=amp1PNimag,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNimag=amp2PNimag,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNimag=amp3PNimag,
        amp3halfPNimag=amp3halfPNimag,
        fac0=fac0,
    )


@jax.jit
def compute_amp_pn_coeffs_33(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> AmpPNCoeffs:
    """Compute amplitude PN coefficients for the (3,3) mode."""
    Sc = m1 * m1 * chi1 + m2 * m2 * chi2
    Sigmac = m2 * chi2 - m1 * chi1

    eta2 = eta * eta

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    # sqrt(15/14) and sqrt(105/2) etc
    sqrt_15_14 = jnp.sqrt(15.0 / 14.0)
    sqrt_105_2 = jnp.sqrt(105.0 / 2.0)
    sqrt_21_10 = jnp.sqrt(21.0 / 10.0)
    sqrt_3_70 = jnp.sqrt(3.0 / 70.0)

    ampN = 0.0
    amp0halfPNreal = (3.0 * sqrt_15_14 * delta) / 4.0
    amp0halfPNimag = 0.0
    amp1PNreal = 0.0
    amp1PNimag = 0.0
    amp1halfPNreal = -3.0 * sqrt_15_14 * delta + (3.0 * sqrt_15_14 * delta * eta) / 2.0
    amp1halfPNimag = 0.0
    amp2PNreal = (
        (9.0 * sqrt_15_14 * delta * jnp.pi) / 4.0
        - (3.0 * sqrt_105_2 * delta * Sc) / 8.0
        - (9.0 * sqrt_15_14 * Sigmac) / 8.0
        + (27.0 * sqrt_15_14 * eta * Sigmac) / 8.0
    )
    amp2PNimag = (-9.0 * sqrt_21_10 * delta) / 4.0 + (
        9.0 * sqrt_15_14 * delta * jnp.log(3.0 / 2.0)
    ) / 2.0
    amp2halfPNreal = (
        (369.0 * sqrt_3_70 * delta) / 88.0
        - (919.0 * sqrt_3_70 * delta * eta) / 22.0
        + (887.0 * sqrt_3_70 * delta * eta2) / 88.0
    )
    amp2halfPNimag = 0.0
    amp3PNreal = 0.0
    amp3PNimag = 0.0
    amp3halfPNreal = 0.0
    amp3halfPNimag = 0.0
    amplog = 0.0

    return AmpPNCoeffs(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp1PNreal=amp1PNreal,
        amp1halfPNreal=amp1halfPNreal,
        amp2PNreal=amp2PNreal,
        amp2halfPNreal=amp2halfPNreal,
        amp3PNreal=amp3PNreal,
        amp3halfPNreal=amp3halfPNreal,
        amplog=amplog,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNimag=amp1PNimag,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNimag=amp2PNimag,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNimag=amp3PNimag,
        amp3halfPNimag=amp3halfPNimag,
        fac0=fac0,
    )


@jax.jit
def compute_amp_pn_coeffs_44(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> AmpPNCoeffs:
    """Compute amplitude PN coefficients for the (4,4) mode."""
    eta2 = eta * eta
    eta3 = eta * eta2

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    sqrt_5_7 = jnp.sqrt(5.0 / 7.0)
    sqrt_35 = jnp.sqrt(35.0)
    sqrt_7_5 = jnp.sqrt(7.0 / 5.0)

    ampN = 0.0
    amp0halfPNreal = 0.0
    amp0halfPNimag = 0.0
    amp1PNreal = (8.0 * sqrt_5_7) / 9.0 - (8.0 * sqrt_5_7 * eta) / 3.0
    amp1PNimag = 0.0
    amp1halfPNreal = 0.0
    amp1halfPNimag = 0.0
    amp2PNreal = (
        -2372.0 / (99.0 * sqrt_35)
        + (5092.0 * sqrt_5_7 * eta) / 297.0
        - (100.0 * sqrt_35 * eta2) / 99.0
    )
    amp2PNimag = 0.0
    amp2halfPNreal = (32.0 * sqrt_5_7 * jnp.pi) / 9.0 - (
        32.0 * sqrt_5_7 * eta * jnp.pi
    ) / 3.0
    amp2halfPNimag = (
        (-16.0 * sqrt_7_5) / 3.0
        + (1193.0 * eta) / (9.0 * sqrt_35)
        + (64.0 * sqrt_5_7 * jnp.log(2.0)) / 9.0
        - (64.0 * sqrt_5_7 * eta * jnp.log(2.0)) / 3.0
    )
    amp3PNreal = (
        1068671.0 / (45045.0 * sqrt_35)
        - (1088119.0 * eta) / (6435.0 * sqrt_35)
        + (293758.0 * eta2) / (1053.0 * sqrt_35)
        - (226097.0 * eta3) / (3861.0 * sqrt_35)
    )
    amp3PNimag = 0.0
    amp3halfPNreal = 0.0
    amp3halfPNimag = 0.0
    amplog = 0.0

    return AmpPNCoeffs(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp1PNreal=amp1PNreal,
        amp1halfPNreal=amp1halfPNreal,
        amp2PNreal=amp2PNreal,
        amp2halfPNreal=amp2halfPNreal,
        amp3PNreal=amp3PNreal,
        amp3halfPNreal=amp3halfPNreal,
        amplog=amplog,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNimag=amp1PNimag,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNimag=amp2PNimag,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNimag=amp3PNimag,
        amp3halfPNimag=amp3halfPNimag,
        fac0=fac0,
    )


@jax.jit
def compute_amp_pn_coeffs_55(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
) -> AmpPNCoeffs:
    """Compute amplitude PN coefficients for the (5,5) mode."""
    eta2 = eta * eta

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    sqrt_66 = jnp.sqrt(66.0)
    sqrt_2_33 = jnp.sqrt(2.0 / 33.0)

    ampN = 0.0
    amp0halfPNreal = 0.0
    amp0halfPNimag = 0.0
    amp1PNreal = 0.0
    amp1PNimag = 0.0
    amp1halfPNreal = (625.0 * delta) / (96.0 * sqrt_66) - (625.0 * delta * eta) / (
        48.0 * sqrt_66
    )
    amp1halfPNimag = 0.0
    amp2PNreal = 0.0
    amp2PNimag = 0.0
    amp2halfPNreal = (
        (-164375.0 * delta) / (3744.0 * sqrt_66)
        + (26875.0 * delta * eta) / (234.0 * sqrt_66)
        - (2500.0 * sqrt_2_33 * delta * eta2) / 117.0
    )
    amp2halfPNimag = 0.0
    amp3PNreal = (3125.0 * delta * jnp.pi) / (96.0 * sqrt_66) - (
        3125.0 * delta * eta * jnp.pi
    ) / (48.0 * sqrt_66)
    amp3PNimag = (
        (-113125.0 * delta) / (1344.0 * sqrt_66)
        + (17639.0 * delta * eta) / (80.0 * sqrt_66)
        + (3125.0 * delta * jnp.log(5.0 / 2.0)) / (48.0 * sqrt_66)
        - (3125.0 * delta * eta * jnp.log(5.0 / 2.0)) / (24.0 * sqrt_66)
    )
    amp3halfPNreal = 0.0
    amp3halfPNimag = 0.0
    amplog = 0.0

    return AmpPNCoeffs(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp1PNreal=amp1PNreal,
        amp1halfPNreal=amp1halfPNreal,
        amp2PNreal=amp2PNreal,
        amp2halfPNreal=amp2halfPNreal,
        amp3PNreal=amp3PNreal,
        amp3halfPNreal=amp3halfPNreal,
        amplog=amplog,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNimag=amp1PNimag,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNimag=amp2PNimag,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNimag=amp3PNimag,
        amp3halfPNimag=amp3halfPNimag,
        fac0=fac0,
    )


def compute_amp_pn_coeffs(
    eta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    delta: float | Array,
    m1: float | Array,
    m2: float | Array,
    mode: int,
) -> AmpPNCoeffs:
    """
    Compute amplitude PN coefficients for a given mode.

    Parameters
    ----------
    eta : Array
        Symmetric mass ratio.
    chi1, chi2 : Array
        Dimensionless spin z-components.
    delta : Array
        Mass difference ratio (m1-m2)/M.
    m1, m2 : Array
        Component masses as fractions of total mass.
    mode : int
        Mode key (22, 21, 33, 44, 55).

    Returns
    -------
    AmpPNCoeffs
        Amplitude PN coefficients for the mode.
    """

    # Use lax.switch for JIT compatibility
    def mode_22():
        return compute_amp_pn_coeffs_22(eta, chi1, chi2, delta, m1, m2)

    def mode_21():
        return compute_amp_pn_coeffs_21(eta, chi1, chi2, delta, m1, m2)

    def mode_33():
        return compute_amp_pn_coeffs_33(eta, chi1, chi2, delta, m1, m2)

    def mode_44():
        return compute_amp_pn_coeffs_44(eta, chi1, chi2, delta, m1, m2)

    def mode_55():
        return compute_amp_pn_coeffs_55(eta, chi1, chi2, delta, m1, m2)

    # Mode index mapping: 22->0, 21->1, 33->2, 44->3, 55->4
    mode_idx = jax.lax.cond(
        mode == 22,
        lambda: 0,
        lambda: jax.lax.cond(
            mode == 21,
            lambda: 1,
            lambda: jax.lax.cond(
                mode == 33,
                lambda: 2,
                lambda: jax.lax.cond(mode == 44, lambda: 3, lambda: 4),  # 55
            ),
        ),
    )

    return jax.lax.switch(mode_idx, [mode_22, mode_21, mode_33, mode_44, mode_55])


# Powers of 5 array for phase computation
POWERS_OF_5 = jnp.array(
    [
        1.0,
        5.0 ** (1.0 / 8.0),
        5.0 ** (2.0 / 8.0),
        5.0 ** (3.0 / 8.0),
        5.0 ** (4.0 / 8.0),
        5.0 ** (5.0 / 8.0),
        5.0 ** (6.0 / 8.0),
        5.0 ** (7.0 / 8.0),
    ]
)
