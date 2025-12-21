# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Amplitude coefficient computation for IMRPhenomT(HM).
======================================================

This module implements the pAmp class functionality from phenomxpy,
computing all the coefficients needed for the IMR amplitude ansatze.

Credits for the original implementations: Cecilio García Quirós

.. autosummary::
    :toctree: _autosummary
    AmplitudeCoeffs
    compute_amplitude_coeffs_22
    compute_amplitude_coeffs_hm
    imr_amplitude
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..utils.utility import solve_3x3_explicit
from . import fits
from .internals import WaveformParams
from .phase import PhaseCoeffs, _inspiral_ansatz_domega, imr_omega


class AmplitudeCoeffs(NamedTuple):
    """
    All amplitude coefficients for a given mode.
    """

    mode: int | Array

    # PN coefficients arrays (ready for ansatz)
    # Real: [ampN, amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half, amplog]
    pn_real_coeffs: Array
    # Imag: [amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half]
    pn_imag_coeffs: Array

    # Pseudo-PN coefficients (3 coefficients)
    inspC1: float | Array
    inspC2: float | Array
    inspC3: float | Array

    # Ringdown coefficients
    alpha1RD: float | Array
    alpha1RD_prec: float | Array

    ampPeak: float | Array
    c1_prec: float | Array
    c2_prec: float | Array
    c3: float | Array
    c4_prec: float | Array

    # Intermediate coefficients
    mergerC1: float | Array
    mergerC2: float | Array
    mergerC3: float | Array
    mergerC4: float | Array

    # Cuts and times
    inspiral_cut: float | Array
    ringdown_cut: float | Array
    tshift: float | Array

    # Prefactor
    fac0: float | Array

    # Phase offset from amplitude
    omegaCutPNAMP: float | Array
    phiCutPNAMP: float | Array


@jax.jit
def _compute_pn_amplitude_coeffs(
    eta: float | Array,
    delta: float | Array,
    chi1: float | Array,
    chi2: float | Array,
    m1: float | Array,
    m2: float | Array,
    mode: int | Array,
) -> Tuple[Array, Array, Array]:
    """
    Compute PN amplitude coefficients for a specific mode.
    Returns (pn_real_coeffs, pn_imag_coeffs, fac0).
    """
    # Derived quantities
    m1_2 = m1 * m1
    m2_2 = m2 * m2

    s1z = chi1
    s2z = chi2

    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)

    Sc = m1_2 * s1z + m2_2 * s2z
    Sigmac = m2 * s2z - m1 * s1z

    eta2 = eta * eta
    eta3 = eta2 * eta

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    # Initialize all to 0.0
    ampN = 0.0
    amp0halfPNreal = 0.0
    amp0halfPNimag = 0.0
    amp1PNreal = 0.0
    amp1PNimag = 0.0
    amp1halfPNreal = 0.0
    amp1halfPNimag = 0.0
    amp2PNreal = 0.0
    amp2PNimag = 0.0
    amp2halfPNreal = 0.0
    amp2halfPNimag = 0.0
    amp3PNreal = 0.0
    amp3PNimag = 0.0
    amp3halfPNreal = 0.0
    amp3halfPNimag = 0.0
    amplog = 0.0

    # Mode 22
    def get_22():
        S0 = m1 * s1z + m2 * s2z
        return (
            1.0,  # ampN
            0.0,  # amp0halfPNreal
            0.0,  # amp0halfPNimag
            -107.0 / 42.0 + (55.0 * eta) / 42.0,  # amp1PNreal
            0.0,  # amp1PNimag
            (-4.0 * chis) / 3.0
            - (4.0 * chia * delta) / 3.0
            + (4.0 * chis * eta) / 3.0
            + 2.0 * jnp.pi,  # amp1halfPNreal
            0.0,  # amp1halfPNimag
            -2173.0 / 1512.0
            - (1069.0 * eta) / 216.0
            + (2047.0 * eta2) / 1512.0
            + S0**2,  # amp2PNreal
            0.0,  # amp2PNimag
            (-107.0 * jnp.pi) / 21.0 + (34.0 * eta * jnp.pi) / 21.0,  # amp2halfPNreal
            -24.0 * eta,  # amp2halfPNimag
            (
                27027409.0 / 646800.0
                - (278185.0 * eta) / 33264.0
                - (20261.0 * eta2) / 2772.0
                + (114635.0 * eta3) / 99792.0
                - (856.0 * 0.5772156649015329) / 105.0
                + (2.0 * jnp.pi**2) / 3.0
                + (41.0 * eta * jnp.pi**2) / 96.0
            ),  # amp3PNreal
            (428.0 * jnp.pi) / 105.0,  # amp3PNimag
            (-2173.0 * jnp.pi) / 756.0
            - (2495.0 * eta * jnp.pi) / 378.0
            + (40.0 * eta2 * jnp.pi) / 27.0,  # amp3halfPNreal
            (14333.0 * eta) / 162.0 - (4066.0 * eta2) / 945.0,  # amp3halfPNimag
            -428.0 / 105.0,  # amplog
        )

    # Mode 21
    def get_21():
        return (
            0.0,  # ampN
            delta / 3.0,  # amp0halfPNreal
            0.0,  # amp0halfPNimag
            -0.5 * chia - (chis * delta) / 2.0,  # amp1PNreal
            0.0,  # amp1PNimag
            (-17.0 * delta) / 84.0 + (5.0 * delta * eta) / 21.0,  # amp1halfPNreal
            0.0,  # amp1halfPNimag
            (delta * jnp.pi) / 3.0
            - (43.0 * delta * Sc) / 21.0
            - (79.0 * Sigmac) / 42.0
            + (139.0 * eta * Sigmac) / 42.0,  # amp2PNreal
            -1.0 / 6.0 * delta - (delta * jnp.log(16.0)) / 6.0,  # amp2PNimag
            (-43.0 * delta) / 378.0
            - (509.0 * delta * eta) / 378.0
            + (79.0 * delta * eta2) / 504.0,  # amp2halfPNreal
            0.0,  # amp2halfPNimag
            (-17.0 * delta * jnp.pi) / 84.0
            + (delta * eta * jnp.pi) / 14.0,  # amp3PNreal
            (17.0 * delta) / 168.0
            - (353.0 * delta * eta) / 84.0
            + (17.0 * delta * jnp.log(16.0)) / 168.0
            - (delta * eta * jnp.log(4096.0)) / 84.0,  # amp3PNimag
            0.0,  # amp3halfPNreal
            0.0,  # amp3halfPNimag
            0.0,  # amplog
        )

    # Mode 33
    def get_33():
        return (
            0.0,  # ampN
            (3.0 * jnp.sqrt(15.0 / 14.0) * delta) / 4.0,  # amp0halfPNreal
            0.0,  # amp0halfPNimag
            0.0,  # amp1PNreal
            0.0,  # amp1PNimag
            -3.0 * jnp.sqrt(15.0 / 14.0) * delta
            + (3.0 * jnp.sqrt(15.0 / 14.0) * delta * eta) / 2.0,  # amp1halfPNreal
            0.0,  # amp1halfPNimag
            (
                (9.0 * jnp.sqrt(15.0 / 14.0) * delta * jnp.pi) / 4.0
                - (3.0 * jnp.sqrt(105.0 / 2.0) * delta * Sc) / 8.0
                - (9.0 * jnp.sqrt(15.0 / 14.0) * Sigmac) / 8.0
                + (27.0 * jnp.sqrt(15.0 / 14.0) * eta * Sigmac) / 8.0
            ),  # amp2PNreal
            (-9.0 * jnp.sqrt(21.0 / 10.0) * delta) / 4.0
            + (9.0 * jnp.sqrt(15.0 / 14.0) * delta * jnp.log(3.0 / 2.0))
            / 2.0,  # amp2PNimag
            (
                (369.0 * jnp.sqrt(3.0 / 70.0) * delta) / 88.0
                - (919.0 * jnp.sqrt(3.0 / 70.0) * delta * eta) / 22.0
                + (887.0 * jnp.sqrt(3.0 / 70.0) * delta * eta2) / 88.0
            ),  # amp2halfPNreal
            0.0,  # amp2halfPNimag
            0.0,  # amp3PNreal
            0.0,  # amp3PNimag
            0.0,  # amp3halfPNreal
            0.0,  # amp3halfPNimag
            0.0,  # amplog
        )

    # Mode 44
    def get_44():
        return (
            0.0,  # ampN
            0.0,  # amp0halfPNreal
            0.0,  # amp0halfPNimag
            (8.0 * jnp.sqrt(5.0 / 7.0)) / 9.0
            - (8.0 * jnp.sqrt(5.0 / 7.0) * eta) / 3.0,  # amp1PNreal
            0.0,  # amp1PNimag
            0.0,  # amp1halfPNreal
            0.0,  # amp1halfPNimag
            -2372.0 / (99.0 * jnp.sqrt(35.0))
            + (5092.0 * jnp.sqrt(5.0 / 7.0) * eta) / 297.0
            - (100.0 * jnp.sqrt(35.0) * eta2) / 99.0,  # amp2PNreal
            0.0,  # amp2PNimag
            (32.0 * jnp.sqrt(5.0 / 7.0) * jnp.pi) / 9.0
            - (32.0 * jnp.sqrt(5.0 / 7.0) * eta * jnp.pi) / 3.0,  # amp2halfPNreal
            (
                (-16.0 * jnp.sqrt(7.0 / 5.0)) / 3.0
                + (1193.0 * eta) / (9.0 * jnp.sqrt(35.0))
                + (64.0 * jnp.sqrt(5.0 / 7.0) * jnp.log(2.0)) / 9.0
                - (64.0 * jnp.sqrt(5.0 / 7.0) * eta * jnp.log(2.0)) / 3.0
            ),  # amp2halfPNimag
            (
                1068671.0 / (45045.0 * jnp.sqrt(35.0))
                - (1088119.0 * eta) / (6435.0 * jnp.sqrt(35.0))
                + (293758.0 * eta2) / (1053.0 * jnp.sqrt(35.0))
                - (226097.0 * eta3) / (3861.0 * jnp.sqrt(35.0))
            ),  # amp3PNreal
            0.0,  # amp3PNimag
            0.0,  # amp3halfPNreal
            0.0,  # amp3halfPNimag
            0.0,  # amplog
        )

    # Mode 55
    def get_55():
        return (
            0.0,  # ampN
            0.0,  # amp0halfPNreal
            0.0,  # amp0halfPNimag
            0.0,  # amp1PNreal
            0.0,  # amp1PNimag
            (625.0 * delta) / (96.0 * jnp.sqrt(66.0))
            - (625.0 * delta * eta) / (48.0 * jnp.sqrt(66.0)),  # amp1halfPNreal
            0.0,  # amp1halfPNimag
            0.0,  # amp2PNreal
            0.0,  # amp2PNimag
            (
                (-164375.0 * delta) / (3744.0 * jnp.sqrt(66.0))
                + (26875.0 * delta * eta) / (234.0 * jnp.sqrt(66.0))
                - (2500.0 * jnp.sqrt(2.0 / 33.0) * delta * eta2) / 117.0
            ),  # amp2halfPNreal
            0.0,  # amp2halfPNimag
            (3125.0 * delta * jnp.pi) / (96.0 * jnp.sqrt(66.0))
            - (3125.0 * delta * eta * jnp.pi) / (48.0 * jnp.sqrt(66.0)),  # amp3PNreal
            (
                (-113125.0 * delta) / (1344.0 * jnp.sqrt(66.0))
                + (17639.0 * delta * eta) / (80.0 * jnp.sqrt(66.0))
                + (3125.0 * delta * jnp.log(5.0 / 2.0)) / (48.0 * jnp.sqrt(66.0))
                - (3125.0 * delta * eta * jnp.log(5.0 / 2.0)) / (24.0 * jnp.sqrt(66.0))
            ),  # amp3PNimag
            0.0,  # amp3halfPNreal
            0.0,  # amp3halfPNimag
            0.0,  # amplog
        )

    # Default (zeros)
    def get_default():
        return (0.0,) * 16

    # Select based on mode
    # We use lax.switch or nested conds. Since modes are integers, we can use select/cond.
    # But mode is passed as Int.
    # We can use a simple python dispatch if mode is static, but here it might be traced.
    # However, usually mode is static in these contexts.
    # If mode is traced, we need lax.cond.

    # Map mode integer to index 0..4
    # 22->0, 21->1, 33->2, 44->3, 55->4

    # Using lax.cond chain
    res = jax.lax.cond(
        mode == 22,
        get_22,
        lambda: jax.lax.cond(
            mode == 21,
            get_21,
            lambda: jax.lax.cond(
                mode == 33,
                get_33,
                lambda: jax.lax.cond(
                    mode == 44,
                    get_44,
                    lambda: jax.lax.cond(mode == 55, get_55, get_default),
                ),
            ),
        ),
    )

    (
        ampN,
        amp0halfPNreal,
        amp0halfPNimag,
        amp1PNreal,
        amp1PNimag,
        amp1halfPNreal,
        amp1halfPNimag,
        amp2PNreal,
        amp2PNimag,
        amp2halfPNreal,
        amp2halfPNimag,
        amp3PNreal,
        amp3PNimag,
        amp3halfPNreal,
        amp3halfPNimag,
        amplog,
    ) = res

    pn_real = jnp.array(
        [
            ampN,
            amp0halfPNreal,
            amp1PNreal,
            amp1halfPNreal,
            amp2PNreal,
            amp2halfPNreal,
            amp3PNreal,
            amp3halfPNreal,
            amplog,
        ]
    )

    pn_imag = jnp.array(
        [
            amp0halfPNimag,
            amp1PNimag,
            amp1halfPNimag,
            amp2PNimag,
            amp2halfPNimag,
            amp3PNimag,
            amp3halfPNimag,
        ]
    )

    return pn_real, pn_imag, fac0


@jax.jit
def compute_amplitude_coeffs_22(
    wf_pafams: WaveformParams,
    phase_coeffs: PhaseCoeffs,
) -> AmplitudeCoeffs:
    """
    Compute all amplitude coefficients for the 22 mode.
    """
    mode = 22
    pn_real, pn_imag, fac0 = _compute_pn_amplitude_coeffs(
        wf_pafams.eta,
        wf_pafams.delta,
        wf_pafams.chi1,
        wf_pafams.chi2,
        wf_pafams.m1,
        wf_pafams.m2,
        mode,
    )

    # Inspiral Coefficients
    tinsppoints = jnp.array([-2000.0, -250.0, -150.0])
    ampInspCP1 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, 22, 1
    )
    ampInspCP2 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, 22, 2
    )
    ampInspCP3 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, 22, 3
    )
    ampInspCP = jnp.array([ampInspCP1, ampInspCP2, ampInspCP3])

    inspC1, inspC2, inspC3 = _solve_inspiral_amplitude_system(
        tinsppoints, ampInspCP, wf_pafams.eta, pn_real, pn_imag, phase_coeffs, fac0
    )
    pseudo_pn = jnp.array([inspC1, inspC2, inspC3])

    # Ringdown Coefficients
    af = fits.final_spin_2017(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)
    Mf = fits.final_mass_2017(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)

    alpha1RD = 2.0 * jnp.pi * fits.fdamp_22(af) / Mf
    alpha2RD = 2.0 * jnp.pi * fits.fdamp_n2_22(af) / Mf
    alpha21RD = 0.5 * (alpha2RD - alpha1RD)

    alpha1RD_prec = alpha1RD
    alpha2RD_prec = alpha2RD
    alpha21RD_prec = alpha21RD

    ampPeak = fits.peak_amp_22(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)
    c3 = fits.rd_amp_c3_22(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)
    c2 = alpha21RD
    c2_prec = alpha21RD_prec

    tanhc3 = jnp.tanh(c3)
    coshc3 = jnp.cosh(c3)

    limit = 0.5 * alpha1RD / tanhc3
    c2 = jnp.where(c2 > jnp.abs(limit), -limit, c2)

    limit_prec = 0.5 * alpha1RD_prec / tanhc3
    c2_prec = jnp.where(c2_prec > jnp.abs(limit_prec), -limit_prec, c2_prec)

    c1 = ampPeak * alpha1RD * coshc3 * coshc3 / c2
    c4 = ampPeak - c1 * tanhc3
    c1_prec = ampPeak * alpha1RD_prec * coshc3 * coshc3 / c2_prec
    c4_prec = ampPeak - c1_prec * tanhc3

    # Intermediate Coefficients
    inspiral_cut = -150.0
    tshift = fits.tshift_22(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)
    ringdown_cut = tshift

    ampMergerCP1 = fits.intermediate_amp_cp1(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, 22
    )
    tcpMerger = -25.0

    mergerC1, mergerC2, mergerC3, mergerC4, dampMECO = (
        _solve_intermediate_amplitude_system(
            inspiral_cut,
            tcpMerger,
            tshift,
            alpha1RD,
            ampPeak,
            ampMergerCP1,
            wf_pafams.eta,
            pn_real,
            pn_imag,
            pseudo_pn,
            phase_coeffs,
            fac0,
        )
    )

    return AmplitudeCoeffs(
        mode=mode,
        pn_real_coeffs=pn_real,
        pn_imag_coeffs=pn_imag,
        inspC1=inspC1,
        inspC2=inspC2,
        inspC3=inspC3,
        alpha1RD=alpha1RD,
        alpha1RD_prec=alpha1RD_prec,
        c1_prec=c1_prec,
        c2_prec=c2_prec,
        c3=c3,
        c4_prec=c4_prec,
        mergerC1=mergerC1,
        mergerC2=mergerC2,
        mergerC3=mergerC3,
        mergerC4=mergerC4,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tshift=tshift,
        fac0=fac0,
        ampPeak=ampPeak,
        omegaCutPNAMP=jnp.array(0.0),
        phiCutPNAMP=jnp.array(0.0),
    )


@jax.jit
def compute_amplitude_coeffs_hm(
    wf_pafams: WaveformParams,
    phase_coeffs_22: PhaseCoeffs,
    mode: int | Array,
) -> AmplitudeCoeffs:
    """
    Compute all amplitude coefficients for HM modes.
    """
    pn_real, pn_imag, fac0 = _compute_pn_amplitude_coeffs(
        wf_pafams.eta,
        wf_pafams.delta,
        wf_pafams.chi1,
        wf_pafams.chi2,
        wf_pafams.m1,
        wf_pafams.m2,
        mode,
    )

    # Inspiral Coefficients
    tinsppoints = jnp.array([-2000.0, -250.0, -150.0])

    # Vectorized fit call for collocation points
    # fits.inspiral_amp_cp(eta, chi1, chi2, mode, k)
    ampInspCP1 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode, 1
    )
    ampInspCP2 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode, 2
    )
    ampInspCP3 = fits.inspiral_amp_cp(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode, 3
    )
    ampInspCP = jnp.array([ampInspCP1, ampInspCP2, ampInspCP3])

    inspC1, inspC2, inspC3 = _solve_inspiral_amplitude_system(
        tinsppoints, ampInspCP, wf_pafams.eta, pn_real, pn_imag, phase_coeffs_22, fac0
    )
    pseudo_pn = jnp.array([inspC1, inspC2, inspC3])

    # Ringdown Coefficients
    af = fits.final_spin_2017(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)
    Mf = fits.final_mass_2017(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2)

    alpha1RD = 2.0 * jnp.pi * fits.fdamp(af, mode) / Mf
    alpha2RD = 2.0 * jnp.pi * fits.fdamp_n2(af, mode) / Mf
    alpha21RD = 0.5 * (alpha2RD - alpha1RD)

    alpha1RD_prec = alpha1RD
    alpha2RD_prec = alpha2RD
    alpha21RD_prec = alpha21RD

    ampPeak = fits.peak_amp(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode)
    c3 = fits.rd_amp_c3(wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode)
    c2 = alpha21RD
    c2_prec = alpha21RD_prec

    tanhc3 = jnp.tanh(c3)
    coshc3 = jnp.cosh(c3)

    limit = jnp.abs(0.5 * alpha1RD / tanhc3)
    c2 = jnp.where(c2 > limit, -limit, c2)

    limit_prec = jnp.abs(0.5 * alpha1RD_prec / tanhc3)
    c2_prec = jnp.where(c2_prec > limit_prec, -limit_prec, c2_prec)

    c1 = ampPeak * alpha1RD * coshc3 * coshc3 / c2
    c4 = ampPeak - c1 * tanhc3
    c1_prec = ampPeak * alpha1RD_prec * coshc3 * coshc3 / c2_prec
    c4_prec = ampPeak - c1_prec * tanhc3

    # Intermediate Coefficients
    inspiral_cut = -150.0
    tshift = fits.tshift(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode
    )  # HM uses 22 tshift?
    # phenomxpy: self.tshift = IMRPhenomT_tshift(self._pWF)
    # IMRPhenomT_tshift uses 22 mode fits.
    ringdown_cut = tshift

    ampMergerCP1 = fits.intermediate_amp_cp1(
        wf_pafams.eta, wf_pafams.chi1, wf_pafams.chi2, mode
    )
    tcpMerger = -25.0

    mergerC1, mergerC2, mergerC3, mergerC4, dampMECO = (
        _solve_intermediate_amplitude_system(
            inspiral_cut,
            tcpMerger,
            tshift,
            alpha1RD,
            ampPeak,
            ampMergerCP1,
            wf_pafams.eta,
            pn_real,
            pn_imag,
            pseudo_pn,
            phase_coeffs_22,
            fac0,
        )
    )

    # Calculate phiCutPNAMP
    omega_cut = imr_omega(inspiral_cut, wf_pafams.eta, phase_coeffs_22)
    x_cut = jnp.power(omega_cut * 0.5, 2.0 / 3.0)
    amp2 = _inspiral_ansatz_amplitude(x_cut, fac0, pn_real, pn_imag, pseudo_pn)
    phiCutPNAMP = jnp.arctan2(jnp.imag(amp2), jnp.real(amp2))

    # Adjust if real part is negative (copysign check in phenomxpy)
    # if np.copysign(1, np.real(amp2)) == -1:
    phiCutPNAMP = jax.lax.cond(
        jnp.real(amp2) < 0, lambda p: p + jnp.pi, lambda p: p, phiCutPNAMP
    )

    omegaCutPNAMP = -jnp.real(
        _der_complex_amp_orientation(
            inspiral_cut,
            wf_pafams.eta,
            pn_real,
            pn_imag,
            pseudo_pn,
            phase_coeffs_22,
            fac0,
            return_phase=True,
        )
    )

    return AmplitudeCoeffs(
        mode=mode,
        pn_real_coeffs=pn_real,
        pn_imag_coeffs=pn_imag,
        inspC1=inspC1,
        inspC2=inspC2,
        inspC3=inspC3,
        alpha1RD=alpha1RD,
        alpha1RD_prec=alpha1RD_prec,
        c1_prec=c1_prec,
        c2_prec=c2_prec,
        c3=c3,
        c4_prec=c4_prec,
        mergerC1=mergerC1,
        mergerC2=mergerC2,
        mergerC3=mergerC3,
        mergerC4=mergerC4,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tshift=tshift,
        fac0=fac0,
        ampPeak=ampPeak,
        omegaCutPNAMP=omegaCutPNAMP,
        phiCutPNAMP=phiCutPNAMP,
    )


@jax.jit
def imr_amplitude(
    time: Array,
    eta: Array,
    amp_coeffs: AmplitudeCoeffs,
    phase_coeffs_22: PhaseCoeffs,
) -> Array:
    """
    Compute IMR amplitude at given times.
    """

    def _amp_scalar(t: Array) -> Array:

        is_post_inspiral = t >= amp_coeffs.inspiral_cut
        is_ringdown = t >= amp_coeffs.ringdown_cut

        # 0 if insp, 1 if interm, 2 if ringdown
        region_idx = is_post_inspiral.astype(jnp.int32) + is_ringdown.astype(jnp.int32)

        def _inspiral(t):
            # Need omega from 22 mode
            omega = imr_omega(t, eta, phase_coeffs_22)
            x = jnp.power(omega * 0.5, 2.0 / 3.0)
            pseudo_pn = jnp.array(
                [amp_coeffs.inspC1, amp_coeffs.inspC2, amp_coeffs.inspC3]
            )
            return _inspiral_ansatz_amplitude(
                x,
                amp_coeffs.fac0,
                amp_coeffs.pn_real_coeffs,
                amp_coeffs.pn_imag_coeffs,
                pseudo_pn,
            )

        def _intermediate(t):
            return _intermediate_ansatz_amplitude(
                t,
                amp_coeffs.mergerC1,
                amp_coeffs.mergerC2,
                amp_coeffs.mergerC3,
                amp_coeffs.mergerC4,
                amp_coeffs.alpha1RD,
                amp_coeffs.tshift,
            )

        def _ringdown(t):
            return _ringdown_ansatz_amplitude(
                t,
                amp_coeffs.c1_prec,
                amp_coeffs.c2_prec,
                amp_coeffs.c3,
                amp_coeffs.c4_prec,
                amp_coeffs.alpha1RD_prec,
                amp_coeffs.tshift,
            )

        # def _post_inspiral(t):
        #     return jax.lax.cond(
        #         t < amp_coeffs.ringdown_cut, _intermediate, _ringdown, t
        #     )

        # return jax.lax.cond(t < amp_coeffs.inspiral_cut, _inspiral, _post_inspiral, t)
        return jax.lax.switch(
            region_idx,
            [_inspiral, _intermediate, _ringdown],
            t,
        )

    # Vectorize
    time = jnp.asarray(time)
    time_shape = jnp.shape(time)
    time_flat = jnp.reshape(time, (-1,))
    amps_flat = jax.vmap(_amp_scalar)(time_flat)
    amps = jnp.reshape(amps_flat, time_shape)

    return amps


# =============================================================================
# Helper functions
# =============================================================================


@jax.jit
def _pn_ansatz_amplitude(
    x: Array,
    fac0: Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
) -> Array:
    """Evaluate PN ansatz amplitude at x = (omega/2)^(2/3)."""
    xhalf = jnp.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf

    # Real part
    ampreal = (
        pn_real[0]
        + pn_real[1] * xhalf
        + pn_real[2] * x
        + pn_real[3] * x1half
        + pn_real[4] * x2
        + pn_real[5] * x2half
        + pn_real[6] * x3
        + pn_real[7] * x3half
        + pn_real[8] * jnp.log(16.0 * x) * x3
    )

    # Imaginary part
    ampimag = (
        pn_imag[0] * xhalf
        + pn_imag[1] * x
        + pn_imag[2] * x1half
        + pn_imag[3] * x2
        + pn_imag[4] * x2half
        + pn_imag[5] * x3
        + pn_imag[6] * x3half
    )

    return fac0 * x * (ampreal + 1j * ampimag)


@jax.jit
def _inspiral_ansatz_amplitude(
    x: Array,
    fac0: Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
) -> Array:
    """Evaluate inspiral ansatz amplitude (PN + pseudo-PN)."""
    # PN part
    pn_amp = _pn_ansatz_amplitude(x, fac0, pn_real, pn_imag)

    # Pseudo-PN part (only affects real part)
    x2 = x * x
    x4 = x2 * x2
    x4half = x4 * jnp.sqrt(x)
    x5 = x4 * x

    pseudo_pn_term = pseudo_pn[0] * x4 + pseudo_pn[1] * x4half + pseudo_pn[2] * x5

    return pn_amp + fac0 * x * pseudo_pn_term


@jax.jit
def _intermediate_ansatz_amplitude(
    time: Array,
    c1: Array,
    c2: Array,
    c3: Array,
    c4: Array,
    alpha: Array,
    tshift: Array,
) -> Array:
    """Evaluate intermediate amplitude ansatz."""
    dt = time - tshift
    phi = alpha * dt
    phi2 = 2.0 * phi

    sech1 = 1.0 / jnp.cosh(phi)
    sech2 = 1.0 / jnp.cosh(phi2)

    return c1 + c2 * sech1 + c3 * jnp.power(sech2, 1.0 / 7.0) + c4 * dt * dt + 0.0j


@jax.jit
def _ringdown_ansatz_amplitude(
    time: Array,
    c1: Array,
    c2: Array,
    c3: Array,
    c4: Array,
    alpha: Array,
    tshift: Array,
) -> Array:
    """Evaluate ringdown amplitude ansatz."""
    dt = time - tshift
    return jnp.exp(-alpha * dt) * (c1 * jnp.tanh(c2 * dt + c3) + c4) + 0.0j


def _solve_inspiral_amplitude_system(
    times: jnp.ndarray,
    amp_vals: jnp.ndarray,
    eta: float | Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    phase_coeffs: PhaseCoeffs,
    fac0: float | Array,
) -> tuple:
    """Solve for inspiral pseudo-PN coefficients."""

    # We need omega at collocation points
    # Use vmap for efficiency
    omegas = imr_omega(times, eta, phase_coeffs)

    xx = jnp.power(0.5 * omegas, 2.0 / 3.0)
    xxhalf = jnp.sqrt(xx)
    xx4 = xx * xx * xx * xx

    # Compute PN offset
    pseudo_pn_zero = jnp.zeros(3)

    def get_offset(x):
        return jnp.real(
            _inspiral_ansatz_amplitude(x, fac0, pn_real, pn_imag, pseudo_pn_zero)
        )

    amp_offsets = jax.vmap(get_offset)(xx)

    B = (1.0 / fac0 / xx) * (amp_vals - amp_offsets)

    # Matrix
    # # c1 x^4 + c2 x^4.5 + c3 x^5
    row_0 = jnp.array([xx4[0], xx4[0] * xxhalf[0], xx4[0] * xx[0]])
    row_1 = jnp.array([xx4[1], xx4[1] * xxhalf[1], xx4[1] * xx[1]])
    row_2 = jnp.array([xx4[2], xx4[2] * xxhalf[2], xx4[2] * xx[2]])

    matrix = jnp.array(
        [
            row_0,
            row_1,
            row_2,
        ]
    )

    solution = solve_3x3_explicit(matrix, B)  # jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2]


def _der_complex_amp_orientation(
    time: float | Array,
    eta: float | Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
    phase_coeffs: PhaseCoeffs,
    fac0: float | Array,
    return_phase: bool = False,
) -> Array:
    """Compute derivative of complex inspiral amplitude."""
    omega = imr_omega(time, eta, phase_coeffs)
    x = jnp.power(omega * 0.5, 2.0 / 3.0)

    xhalf = jnp.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf
    x4 = x2 * x2
    x4half = x4 * xhalf
    x5 = x3 * x2

    # Real part
    ampreal = (
        pn_real[0]
        + pn_real[1] * xhalf
        + pn_real[2] * x
        + pn_real[3] * x1half
        + pn_real[4] * x2
        + pn_real[5] * x2half
        + pn_real[6] * x3
        + pn_real[7] * x3half
        + pn_real[8] * jnp.log(16.0 * x) * x3
        + pseudo_pn[0] * x4
        + pseudo_pn[1] * x4half
        + pseudo_pn[2] * x5
    )

    # Imaginary part
    ampimag = (
        pn_imag[0] * xhalf
        + pn_imag[1] * x
        + pn_imag[2] * x1half
        + pn_imag[3] * x2
        + pn_imag[4] * x2half
        + pn_imag[5] * x3
        + pn_imag[6] * x3half
    )

    # Derivatives w.r.t x
    dampreal = (
        0.5 * pn_real[1] / xhalf
        + pn_real[2]
        + 1.5 * pn_real[3] * xhalf
        + 2.0 * pn_real[4] * x
        + 2.5 * pn_real[5] * x1half
        + 3.0 * pn_real[6] * x2
        + 3.5 * pn_real[7] * x2half
        + pn_real[8] * x2 * (1.0 + 3.0 * jnp.log(16.0 * x))
        + 4.0 * pseudo_pn[0] * x3
        + 4.5 * pseudo_pn[1] * x3half
        + 5.0 * pseudo_pn[2] * x4
    )

    dampimag = (
        0.5 * pn_imag[0] / xhalf
        + pn_imag[1]
        + 1.5 * pn_imag[2] * xhalf
        + 2.0 * pn_imag[3] * x
        + 2.5 * pn_imag[4] * x1half
        + 3.0 * pn_imag[5] * x2
        + 3.5 * pn_imag[6] * x2half
    )

    der_x_per_omega = jnp.cbrt(2.0 / omega) / 3.0

    # domega/dt
    # We need to handle the branch manually or use the helper from phase
    # But phase helper is for 22 mode coefficients.
    # We have phase_coeffs which is 22 mode.

    def inspiral_branch(t):
        omega_pn_coeffs = jnp.array(
            [
                phase_coeffs.omega1PN,
                phase_coeffs.omega1halfPN,
                phase_coeffs.omega2PN,
                phase_coeffs.omega2halfPN,
                phase_coeffs.omega3PN,
                phase_coeffs.omega3halfPN,
            ]
        )
        omega_pseudo_pn_coeffs = jnp.array(
            [
                phase_coeffs.omegaInspC1,
                phase_coeffs.omegaInspC2,
                phase_coeffs.omegaInspC3,
                phase_coeffs.omegaInspC4,
                phase_coeffs.omegaInspC5,
                phase_coeffs.omegaInspC6,
            ]
        )
        return _inspiral_ansatz_domega(t, eta, omega_pn_coeffs, omega_pseudo_pn_coeffs)

    def merger_branch(t):
        # This logic is duplicated from phase.py but we need it here
        # Or we can expose it in phase.py
        # I exposed _ringdown_ansatz_domega but not the intermediate one directly?
        # Wait, phase.py has compute_domega_cut which does exactly this switch.
        # But it switches at tCut_threshold.
        # Here we want domega/dt at 'time'.
        # If time is in intermediate region, we use merger branch.

        arcsinh = jnp.arcsinh(phase_coeffs.alpha1RD * t)
        return (
            -phase_coeffs.omegaRING
            / jnp.sqrt(1.0 + (phase_coeffs.alpha1RD * t) ** 2)
            * (
                phase_coeffs.domegaPeak
                + phase_coeffs.alpha1RD
                * (
                    2.0 * phase_coeffs.omegaMergerC1 * arcsinh
                    + 3.0 * phase_coeffs.omegaMergerC2 * arcsinh * arcsinh
                    + 4.0 * phase_coeffs.omegaMergerC3 * arcsinh**3
                )
            )
        )

    der_omega_per_t = jax.lax.cond(
        time < phase_coeffs.inspiral_cut, inspiral_branch, merger_branch, time
    )

    amp = jnp.abs(ampreal + 1j * ampimag)

    if return_phase:
        return (
            (dampimag * ampreal - dampreal * ampimag)
            / (amp * amp)
            * der_x_per_omega
            * der_omega_per_t
        )

    return (
        fac0
        * (ampreal * (dampreal * x + ampreal) + ampimag * (dampimag * x + ampimag))
        / amp
        * der_x_per_omega
        * der_omega_per_t
    )


def _solve_intermediate_amplitude_system(
    tCut: float | Array,
    tcpMerger: float | Array,
    tshift: float | Array,
    alpha1RD: float | Array,
    ampPeak: float | Array,
    ampMergerCP1: float | Array,
    eta: float | Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
    phase_coeffs: PhaseCoeffs,
    fac0: Array,
) -> tuple:
    """Solve for intermediate amplitude coefficients."""

    # Compute omega at tCut
    omega_cut = imr_omega(tCut, eta, phase_coeffs)
    x_cut = jnp.power(omega_cut * 0.5, 2.0 / 3.0)

    # Inspiral amplitude at tCut
    ampinsp_cplx = _inspiral_ansatz_amplitude(x_cut, fac0, pn_real, pn_imag, pseudo_pn)
    ampinsp = ampinsp_cplx

    # Match sign
    ampinsp = jnp.copysign(jnp.abs(ampinsp), jnp.real(ampinsp))

    phi = alpha1RD * (tCut - tshift)
    phi2 = 2.0 * phi

    sech1 = 1.0 / jnp.cosh(phi)
    sech2 = 1.0 / jnp.cosh(phi2)

    # Row 0: Match amplitude at tCut
    row_0 = jnp.array([1.0, sech1, jnp.power(sech2, 1.0 / 7.0), (tCut - tshift) ** 2])

    # Row 1: Match amplitude at tcpMerger
    phib = alpha1RD * (tcpMerger - tshift)
    sech1b = 1.0 / jnp.cosh(phib)
    sech2b = 1.0 / jnp.cosh(2.0 * phib)

    row_1 = jnp.array(
        [1.0, sech1b, jnp.power(sech2b, 1.0 / 7.0), (tcpMerger - tshift) ** 2]
    )

    # Row 2: Match amplitude at peak (t=tshift)
    row_2 = jnp.ones(4)

    # Row 3: Match derivative at tCut
    dampMECO = jnp.copysign(1.0, jnp.real(ampinsp_cplx)) * _der_complex_amp_orientation(
        tCut, eta, pn_real, pn_imag, pseudo_pn, phase_coeffs, fac0, return_phase=False
    )

    tanh = jnp.tanh(phi)
    sinh = jnp.sinh(phi2)

    aux1 = -alpha1RD * sech1 * tanh
    aux2 = (-2.0 / 7.0) * alpha1RD * sinh * jnp.power(sech2, 8.0 / 7.0)
    aux3 = 2.0 * (tCut - tshift)

    row_3 = jnp.array([0.0, aux1, aux2, aux3])

    matrix = jnp.array([row_0, row_1, row_2, row_3])

    B = jnp.array([ampinsp, ampMergerCP1, ampPeak, dampMECO])
    solution = jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2], solution[3], dampMECO
