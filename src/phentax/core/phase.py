# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Phase and omega coefficient computation for IMRPhenomTHM.

This module implements the pPhase class functionality from phenomxpy,
computing all the coefficients needed for the IMR omega and phase ansatze.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array

from ..utils.utility import solve_3x3_explicit
from . import collocation, fits, pn_coeffs
from .internals import WaveformParams, compute_wf_length_params

jax.config.update("jax_enable_x64", True)


class PhaseCoeffs(eqx.Module):
    """
    All phase/omega coefficients for a given mode.

    Contains PN coefficients, pseudo-PN coefficients, ringdown parameters,
    and intermediate region coefficients.

    Parameters
    ----------
    mode : int | Array
        Mode number (e.g., 22, 33, 44, etc.).
    omega1PN : float | Array
        1PN coefficient for omega inspiral.
    omega1halfPN : float | Array
        1.5PN coefficient for omega inspiral.
    omega2PN : float | Array
        2PN coefficient for omega inspiral.
    omega2halfPN : float | Array
        2.5PN coefficient for omega inspiral.
    omega3PN : float | Array
        3PN coefficient for omega inspiral.
    omega3halfPN : float | Array
        3.5PN coefficient for omega inspiral.
    omegaInspC1 : float | Array
        1st pseudo-PN coefficient for omega inspiral.
    omegaInspC2 : float | Array
        2nd pseudo-PN coefficient for omega inspiral.
    omegaInspC3 : float | Array
        3rd pseudo-PN coefficient for omega inspiral.
    omegaInspC4 : float | Array
        4th pseudo-PN coefficient for omega inspiral.
    omegaInspC5 : float | Array
        5th pseudo-PN coefficient for omega inspiral.
    omegaInspC6 : float | Array
        6th pseudo-PN coefficient for omega inspiral.
    omegaRING : float | Array
        Ringdown frequency (2*pi*fring).
    alpha1RD : float | Array
        Ringdown damping rate (2*pi*fdamp).
    omegaRING_prec : float | Array
        Ringdown frequency for precessing case.
    omegaPeak : float | Array
        Peak frequency.
    c1 : float | Array
        1st ringdown ansatz coefficient.
    c2 : float | Array
        2nd ringdown ansatz coefficient.
    c3 : float | Array
        3rd ringdown ansatz coefficient.
    c4 : float | Array
        4th ringdown ansatz coefficient.
    c1_prec : float | Array
        1st ringdown ansatz coefficient for precessing case.
    omegaMergerC1 : float | Array
        1st intermediate region coefficient.
    omegaMergerC2 : float | Array
        2nd intermediate region coefficient.
    omegaMergerC3 : float | Array
        3rd intermediate region coefficient.
    omegaCut : float | Array
        Omega at inspiral cut.
    domegaCut : float | Array
        domega/dt at inspiral cut.
    domegaPeak : float | Array
        domega/dt at peak.
    inspiral_cut : float | Array
        Transition time inspiral -> intermediate.
    ringdown_cut : float | Array
        Transition time intermediate -> ringdown (= 0, peak time).
    tt0 : float | Array
        t0 from fit.
    tEarly : float | Array
        Early time for phase offset.
    omegaCutPNAMP : float | Array
        Omega contribution from complex amplitude at transtion time ``pAmp.inspiral_cut``.
    phiCutPNAMP : float | Array
        Phase contribution from complex amplitude at transtion time ``pAmp.inspiral_cut``.
    phOffInsp : float | Array
        Phase offset for inspiral.
    phOffMerger : float | Array
        Phase offset for intermediate region.
    phOffRD : float | Array
        Phase offset for ringdown.
    phoff : float | Array
        phase offset for different modes.
    phiref0 : float | Array
        reference phase of the 22 mode at t=tref.
    powers_of_5 : Array
        Powers of 5^(n/8) for n=0..7 for phase computation.
    """

    mode: int | Array

    # PN coefficients (TaylorT3)
    omega1PN: float | Array
    omega1halfPN: float | Array
    omega2PN: float | Array
    omega2halfPN: float | Array
    omega3PN: float | Array
    omega3halfPN: float | Array

    # Pseudo-PN coefficients (6 coefficients for omega inspiral)
    omegaInspC1: float | Array
    omegaInspC2: float | Array
    omegaInspC3: float | Array
    omegaInspC4: float | Array
    omegaInspC5: float | Array
    omegaInspC6: float | Array

    # Ringdown quantities
    omegaRING: float | Array  # 2*pi*fring
    alpha1RD: float | Array  # 2*pi*fdamp
    omegaRING_prec: float | Array  # For precessing case

    # Peak frequency
    omegaPeak: float | Array

    # Ringdown ansatz coefficients
    c1: float | Array
    c2: float | Array
    c3: float | Array
    c4: float | Array
    c1_prec: float | Array

    # intermediate ansatz coefficients
    omegaMergerC1: float | Array
    omegaMergerC2: float | Array
    omegaMergerC3: float | Array

    # intermediate region values
    omegaCut: float | Array  # omega at inspiral cut
    domegaCut: float | Array  # domega/dt at inspiral cut
    domegaPeak: float | Array  # domega/dt at peak

    # Times and cuts
    inspiral_cut: float | Array  # tCut: transition time inspiral -> intermediate
    ringdown_cut: float | Array  # = 0 (peak time)
    tt0: float | Array  # t0 from fit
    tEarly: float | Array  # Early time for phase offset

    omegaCutPNAMP: (
        float | Array
    )  # Omega contribution from complex amplitude at transtion time ``pAmp.inspiral_cut``.
    phiCutPNAMP: (
        float | Array
    )  # Phase contribution from complex amplitude at transtion time ``pAmp.inspiral_cut``.

    # Phase continuity offsets
    phOffInsp: float | Array
    phOffMerger: float | Array
    phOffRD: float | Array

    # phase offset for different modes
    phoff: float | Array
    # reference phase at t=tref
    phiref0: float | Array

    # Powers of 5 for phase computation
    powers_of_5: Array


def _compute_pn_and_pseudo_pn(
    wf_params: WaveformParams,
) -> Tuple[
    jnp.ndarray,
    collocation.OmegaPseudoPNCoeffs,
    float | Array,
    float | Array,
    Array,
]:
    """
    Compute PN and pseudo-PN coefficients, which are common to all modes.
    """
    # Powers of 5 for phase computation
    base = jnp.power(5.0, 1.0 / 8.0)
    powers_of_5 = jnp.array(
        [1.0, base, base**2, base**3, base**4, base**5, base**6, base**7]
    )

    # PN Coefficients
    omega_pn = pn_coeffs.compute_omega_pn_coeffs(
        wf_params.eta,
        wf_params.chi1,
        wf_params.chi2,
        wf_params.delta,
        wf_params.m1,
        wf_params.m2,
    )
    omega_pn_array = jnp.array(
        [
            omega_pn.omega1PN,
            omega_pn.omega1halfPN,
            omega_pn.omega2PN,
            omega_pn.omega2halfPN,
            omega_pn.omega3PN,
            omega_pn.omega3halfPN,
        ]
    )

    # Collocation points and pseudo-PN coefficients
    omega_cp_values, tt0, tEarly = collocation.compute_omega_collocation_points(
        wf_params.eta, wf_params.chi1, wf_params.chi2, omega_pn_array
    )

    pseudo_pn = collocation.compute_omega_pseudo_pn_coeffs(
        omega_pn_array, omega_cp_values
    )

    return omega_pn_array, pseudo_pn, tt0, tEarly, powers_of_5


@jax.jit
def compute_phase_coeffs_22(
    wf_params: WaveformParams,
) -> tuple[WaveformParams, PhaseCoeffs]:
    """
    Compute all phase coefficients for the 22 mode.

    Parameters
    ----------
    wf_params : WaveformParams
        Waveform parameters for the waveform.

    Returns
    -------
    tuple[WaveformParams, PhaseCoeffs]
        Updated derived parameters and phase coefficients for mode 22.
    """
    # Common coefficients
    omega_pn_array, pseudo_pn, tt0, tEarly, powers_of_5 = _compute_pn_and_pseudo_pn(
        wf_params
    )

    # Final state
    af = fits.final_spin_2017(wf_params.eta, wf_params.chi1, wf_params.chi2)
    Mfinal = fits.final_mass_2017(wf_params.eta, wf_params.chi1, wf_params.chi2)

    # Ringdown quantities
    omegaRING = 2.0 * jnp.pi * fits.fring_22(af) / Mfinal
    alpha1RD = 2.0 * jnp.pi * fits.fdamp_22(af) / Mfinal
    omegaRING_prec = omegaRING

    omegaPeak = fits.peak_freq_22(wf_params.eta, wf_params.chi1, wf_params.chi2)

    c2 = fits.rd_freq_d2_22(wf_params.eta, wf_params.chi1, wf_params.chi2)
    c3 = fits.rd_freq_d3_22(wf_params.eta, wf_params.chi1, wf_params.chi2)
    c4 = 0.0
    c1 = (1.0 + c3 + c4) * (omegaRING - omegaPeak) / c2 / (c3 + 2.0 * c4)
    c1_prec = c1

    # Cuts
    inspiral_cut = -26.982976386771437 / wf_params.eta
    ringdown_cut = 0.0

    # int | Arrayermediate region
    pseudo_pn_array = jnp.array(
        [
            pseudo_pn.c1,
            pseudo_pn.c2,
            pseudo_pn.c3,
            pseudo_pn.c4,
            pseudo_pn.c5,
            pseudo_pn.c6,
        ]
    )

    omegaCut = _inspiral_ansatz_omega_single(
        inspiral_cut, wf_params.eta, omega_pn_array, pseudo_pn_array, m=2
    )

    domegaCut = _inspiral_ansatz_domega(
        inspiral_cut, wf_params.eta, omega_pn_array, pseudo_pn_array, m=2
    )

    domegaPeak = -_ringdown_ansatz_domega(0.0, c1, c2, c3, c4) / omegaRING

    tcpMerger = -5.0 / (wf_params.eta * jnp.power(0.95, 8))
    omegaMergerCP = (
        1.0
        - fits.intermediate_freq_cp1_22(wf_params.eta, wf_params.chi1, wf_params.chi2)
        / omegaRING
    )
    omegaCutBar = 1.0 - omegaCut / omegaRING
    domegaCut = -domegaCut / omegaRING

    omegaMergerC1, omegaMergerC2, omegaMergerC3 = _solve_intermediate_omega_system(
        alpha1RD,
        inspiral_cut,
        tcpMerger,
        omegaCutBar,
        omegaMergerCP,
        domegaCut,
        domegaPeak,
        omegaPeak,
        omegaRING,
    )

    # Phase offsets
    thetabarini = jnp.power(wf_params.eta * (tt0 - tEarly), -1.0 / 8.0)
    pn_phase_at_thetabarini = _pn_ansatz_phase(
        thetabarini, wf_params.eta, powers_of_5, omega_pn_array
    )
    inspiral_phase_at_tEarly = _inspiral_ansatz_phase_value_22(
        tEarly,
        wf_params.eta,
        powers_of_5,
        omega_pn_array,
        pseudo_pn_array,
        0.0,
    )
    phOffInsp = pn_phase_at_thetabarini - inspiral_phase_at_tEarly

    inspiral_phase_at_cut = _inspiral_ansatz_phase_value_22(
        inspiral_cut,
        wf_params.eta,
        powers_of_5,
        omega_pn_array,
        pseudo_pn_array,
        phOffInsp,
    )

    intermediate_phase_at_cut = _intermediate_ansatz_phase_value(
        inspiral_cut,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        0.0,
    )
    phOffMerger = inspiral_phase_at_cut - intermediate_phase_at_cut

    phOffRD = _intermediate_ansatz_phase_value(
        0.0,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        phOffMerger,
    )

    PhaseCoeffs22 = PhaseCoeffs(
        mode=22,
        omega1PN=omega_pn_array[0],
        omega1halfPN=omega_pn_array[1],
        omega2PN=omega_pn_array[2],
        omega2halfPN=omega_pn_array[3],
        omega3PN=omega_pn_array[4],
        omega3halfPN=omega_pn_array[5],
        omegaInspC1=pseudo_pn.c1,
        omegaInspC2=pseudo_pn.c2,
        omegaInspC3=pseudo_pn.c3,
        omegaInspC4=pseudo_pn.c4,
        omegaInspC5=pseudo_pn.c5,
        omegaInspC6=pseudo_pn.c6,
        omegaRING=omegaRING,
        alpha1RD=alpha1RD,
        omegaRING_prec=omegaRING_prec,
        omegaPeak=omegaPeak,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c1_prec=c1_prec,
        omegaMergerC1=omegaMergerC1,
        omegaMergerC2=omegaMergerC2,
        omegaMergerC3=omegaMergerC3,
        omegaCut=omegaCut,
        domegaCut=domegaCut,
        domegaPeak=domegaPeak,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tt0=tt0,
        tEarly=tEarly,
        omegaCutPNAMP=jnp.array(0.0),
        phiCutPNAMP=jnp.array(0.0),
        phoff=jnp.array(0.0),
        phiref0=jnp.array(0.0),
        phOffInsp=phOffInsp,
        phOffMerger=phOffMerger,
        phOffRD=phOffRD,
        powers_of_5=powers_of_5,
    )

    def _compute_min(_):
        return get_time_of_frequency(
            wf_params.Mf_min,
            wf_params.eta,
            PhaseCoeffs22,
            wf_params.t_low,
            wf_params.atol,
            wf_params.rtol,
        )

    def _use_existing_min(_):
        return wf_params.Mt_min

    # This works inside vmap because isnan returns a boolean tracer
    _Mt_min = jax.lax.cond(
        jnp.isnan(wf_params.Mt_min), _compute_min, _use_existing_min, operand=None
    )

    # wf_params = wf_params._replace(Mt_min=_Mt_min)
    wf_params = eqx.tree_at(lambda p: p.Mt_min, wf_params, _Mt_min)

    # check here if fmin and fref are the same to avoid a second root solving
    _Mt_ref = jax.lax.cond(
        jnp.isnan(wf_params.Mt_ref),
        lambda: jax.lax.cond(
            wf_params.Mf_min == wf_params.Mf_ref,
            lambda: wf_params.Mt_min,
            lambda: wf_params.Mt_ref,
        ),
        lambda: wf_params.Mt_ref,
    )

    # wf_params = wf_params._replace(Mt_ref=_Mt_ref)
    wf_params = eqx.tree_at(lambda p: p.Mt_ref, wf_params, _Mt_ref)

    def _compute_ref(_):
        return get_time_of_frequency(
            wf_params.Mf_ref,
            wf_params.eta,
            PhaseCoeffs22,
            wf_params.t_low,
            wf_params.atol,
            wf_params.rtol,
        )

    def _use_existing_ref(_):
        return wf_params.Mt_ref

    # This works inside vmap because isnan returns a boolean tracer
    _Mt_ref = jax.lax.cond(
        jnp.isnan(_Mt_ref), _compute_ref, _use_existing_ref, operand=None
    )

    # wf_params = wf_params._replace(Mt_ref=_Mt_ref)
    wf_params = eqx.tree_at(lambda p: p.Mt_ref, wf_params, _Mt_ref)

    wf_params = compute_wf_length_params(
        wf_params
    )  # compute waveform length parameters based on Mt_min
    phiref0 = imr_phase(_Mt_ref, wf_params.eta, PhaseCoeffs22)  # phase at tref
    # phiref0 = imr_phase(wf_params.t_ref, wf_params.eta, PhaseCoeffs22)  # phase at tref
    return wf_params, eqx.tree_at(lambda p: p.phiref0, PhaseCoeffs22, phiref0)


@jax.jit
def compute_phase_coeffs_hm(
    wf_params: WaveformParams,
    phase_coeffs_22: PhaseCoeffs,
    OmegaCutPNAMP: Array,
    PhiCutPNAMP: Array,
    mode: int | Array,
) -> PhaseCoeffs:
    """
    Compute all phase/omega coefficients for HM modes.

    Parameters
    ----------
    wf_params : WaveformParams
        Waveform parameters for the waveform.
    phase_coeffs_22 : PhaseCoeffs
        Phase coefficients for the 22 mode.
    OmegaCutPNAMP : Array
        Omega contribution from complex amplitude at transtion time ``pAmp.inspiral_cut`` for HM modes.
    PhiCutPNAMP : Array
        Phase contribution from complex amplitude at transtion time ``pAmp.inspiral_cut`` for HM modes.
    mode : int | Array
        Mode number (e.g., 33, 44, etc.).
    """
    # Common coefficients
    omega_pn_array, pseudo_pn, tt0, tEarly, powers_of_5 = _compute_pn_and_pseudo_pn(
        wf_params
    )

    m = mode % 10

    # Final state
    af = fits.final_spin_2017(wf_params.eta, wf_params.chi1, wf_params.chi2)
    Mfinal = fits.final_mass_2017(wf_params.eta, wf_params.chi1, wf_params.chi2)

    # Ringdown quantities
    omegaRING = 2.0 * jnp.pi * fits.fring(af, mode) / Mfinal
    alpha1RD = 2.0 * jnp.pi * fits.fdamp(af, mode) / Mfinal
    omegaRING_prec = omegaRING

    omegaPeak = fits.peak_freq(wf_params.eta, wf_params.chi1, wf_params.chi2, mode)

    c2 = fits.rd_freq_d2(wf_params.eta, wf_params.chi1, wf_params.chi2, mode)
    c3 = fits.rd_freq_d3(wf_params.eta, wf_params.chi1, wf_params.chi2, mode)
    c4 = 0.0
    c1 = (1.0 + c3 + c4) * (omegaRING - omegaPeak) / c2 / (c3 + 2.0 * c4)
    c1_prec = c1

    # Cuts
    inspiral_cut = -150.0
    ringdown_cut = 0.0

    # int | Arrayermediate region
    omegaCut = (
        m
        / 2.0
        * imr_omega(inspiral_cut, eta=wf_params.eta, phase_coeffs=phase_coeffs_22)
    )
    domegaCut = compute_domega_cut(
        inspiral_cut, phase_coeffs_22.inspiral_cut, wf_params.eta, phase_coeffs_22
    )
    domegaCut = -m / 2.0 * domegaCut / omegaRING

    domegaPeak = -_ringdown_ansatz_domega(0.0, c1, c2, c3, c4) / omegaRING

    tcpMerger = -25.0
    omegaMergerCP = (
        1.0
        - fits.intermediate_freq_cp1(
            wf_params.eta, wf_params.chi1, wf_params.chi2, mode
        )
        / omegaRING
    )
    omegaCutBar = 1.0 - (omegaCut + OmegaCutPNAMP) / omegaRING

    omegaMergerC1, omegaMergerC2, omegaMergerC3 = _solve_intermediate_omega_system(
        alpha1RD,
        inspiral_cut,
        tcpMerger,
        omegaCutBar,
        omegaMergerCP,
        domegaCut,
        domegaPeak,
        omegaPeak,
        omegaRING,
    )

    # Phase offsets
    phOffInsp = 0.0
    phMECOinsp = (
        m
        / 2.0
        * imr_phase(inspiral_cut, eta=wf_params.eta, phase_coeffs=phase_coeffs_22)
    )
    phMECOmerger = _intermediate_ansatz_phase_value(
        inspiral_cut,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        0.0,
    )
    phOffMerger = phMECOinsp - phMECOmerger

    phOffRD = _intermediate_ansatz_phase_value(
        0.0,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        phOffMerger,
    )

    phoff = _get_offset(mode)

    return PhaseCoeffs(
        mode=mode,
        omega1PN=omega_pn_array[0],
        omega1halfPN=omega_pn_array[1],
        omega2PN=omega_pn_array[2],
        omega2halfPN=omega_pn_array[3],
        omega3PN=omega_pn_array[4],
        omega3halfPN=omega_pn_array[5],
        omegaInspC1=pseudo_pn.c1,
        omegaInspC2=pseudo_pn.c2,
        omegaInspC3=pseudo_pn.c3,
        omegaInspC4=pseudo_pn.c4,
        omegaInspC5=pseudo_pn.c5,
        omegaInspC6=pseudo_pn.c6,
        omegaRING=omegaRING,
        alpha1RD=alpha1RD,
        omegaRING_prec=omegaRING_prec,
        omegaPeak=omegaPeak,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c1_prec=c1_prec,
        omegaMergerC1=omegaMergerC1,
        omegaMergerC2=omegaMergerC2,
        omegaMergerC3=omegaMergerC3,
        omegaCut=omegaCut,
        domegaCut=domegaCut,
        domegaPeak=domegaPeak,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tt0=tt0,
        tEarly=tEarly,
        omegaCutPNAMP=OmegaCutPNAMP,
        phiCutPNAMP=PhiCutPNAMP,
        phoff=phoff,
        phiref0=phase_coeffs_22.phiref0,
        phOffInsp=phOffInsp,
        phOffMerger=phOffMerger,
        phOffRD=phOffRD,
        powers_of_5=powers_of_5,
    )


def imr_omega(
    time: float | Array, eta: float | Array, phase_coeffs: PhaseCoeffs
) -> float | Array:
    """
    Compute the frequency :math:`\\omega(t) = 2\\pi f(t)` at given times for a given mode.

    Parameters
    ----------
    time : float | Array
        Time(s) at which to compute the phase.
    eta : float | Array
        Symmetric mass ratio.
    phase_coeffs : PhaseCoeffs
        Phase coefficients for the mode.

    Returns
    -------
    Array
        Phase value(s) at the given time(s).
    """
    m = phase_coeffs.mode % 10
    # Prepare coefficient arrays for helper functions
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

    @jax.jit
    def _omega_scalar(t: Array) -> Array:
        """Compute omega at a single time t."""

        is_post_inspiral = t >= phase_coeffs.inspiral_cut
        is_ringdown = t >= phase_coeffs.ringdown_cut

        # 0 if insp, 1 if interm, 2 if ringdown
        region_idx = is_post_inspiral.astype(jnp.int32) + is_ringdown.astype(jnp.int32)

        def _inspiral(t):
            return _inspiral_ansatz_omega_single(
                t,
                eta,
                omega_pn_coeffs,
                omega_pseudo_pn_coeffs,
                m,
            )

        def _intermediate(t):
            return _intermediate_ansatz_omega_single(
                t,
                phase_coeffs.alpha1RD,
                phase_coeffs.omegaPeak,
                phase_coeffs.domegaPeak,
                phase_coeffs.omegaRING,
                phase_coeffs.omegaMergerC1,
                phase_coeffs.omegaMergerC2,
                phase_coeffs.omegaMergerC3,
            )

        def _ringdown(t):
            return _ringdown_ansatz_omega_single(
                t,
                phase_coeffs.c1,
                phase_coeffs.c2,
                phase_coeffs.c3,
                phase_coeffs.c4,
                phase_coeffs.omegaRING,
            )

        return jax.lax.switch(
            region_idx,
            [_inspiral, _intermediate, _ringdown],
            t,
        )

    # Vectorize over time array
    time = jnp.asarray(time)
    time_shape = jnp.shape(time)
    time_flat = jnp.reshape(time, (-1,))
    omegas_flat = jax.vmap(_omega_scalar)(time_flat)
    omegas = jnp.reshape(omegas_flat, time_shape)

    return omegas


def imr_omega_dot(
    time: float | Array, eta: float | Array, phase_coeffs: PhaseCoeffs
) -> float | Array:
    """
    Compute the frequency derivative :math:`\\dot{\\omega}(t)` at given times for a given mode with JAX autodiff.

    Parameters
    ----------
    time : float | Array
        Time(s) at which to compute the phase.
    eta : float | Array
        Symmetric mass ratio.
    phase_coeffs : PhaseCoeffs
        Phase coefficients for the mode.

    Returns
    -------
    Array
        Phase derivative value(s) at the given time(s).
    """
    domega_dt = jax.grad(lambda t: imr_omega(t, eta, phase_coeffs))(time)
    return domega_dt


def imr_phase(
    time: float | Array,
    eta: float | Array,
    phase_coeffs: PhaseCoeffs,
    phase_22: float | Array = 0.0,
) -> Array:
    """
    Compute the phase at given times for a given mode.

    Parameters
    ----------
    time : float | Array
        Time(s) at which to compute the phase.
    eta : float | Array
        Symmetric mass ratio.
    phase_coeffs : PhaseCoeffs
        Phase coefficients for the mode.
    phase_22 : float | Array, optional
        Phase of the (2,2) mode at the same times (default is 0.0). This is used for the higher modes' inspiral phase computation.

    Returns
    -------
    Array
        Phase value(s) at the given time(s).
    """
    # Prepare coefficient arrays for helper functions
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

    @jax.jit
    def _phase_scalar(t: Array, _phase_22: float | Array) -> Array:
        # Determine region index: 0=Inspiral, 1=Intermediate, 2=Ringdown
        # Using boolean arithmetic is often faster than branching logic for indices
        is_post_inspiral = t >= phase_coeffs.inspiral_cut
        is_ringdown = t >= phase_coeffs.ringdown_cut

        # 0 if insp, 1 if interm, 2 if ringdown
        region_idx = is_post_inspiral.astype(jnp.int32) + is_ringdown.astype(jnp.int32)

        def _inspiral(t, _phase_22):
            return _inspiral_ansatz_phase_value(
                t,
                eta,
                phase_coeffs.powers_of_5,
                omega_pn_coeffs,
                omega_pseudo_pn_coeffs,
                phase_coeffs.phOffInsp,
                phase_coeffs.mode,
                phase_22=_phase_22,
            )

        def _intermediate(t, _):
            val = _intermediate_ansatz_phase_value(
                t,
                phase_coeffs.alpha1RD,
                phase_coeffs.omegaMergerC1,
                phase_coeffs.omegaMergerC2,
                phase_coeffs.omegaMergerC3,
                phase_coeffs.omegaPeak,
                phase_coeffs.domegaPeak,
                phase_coeffs.omegaRING,
                phase_coeffs.phOffMerger,
            )
            return val - phase_coeffs.phiCutPNAMP

        def _ringdown(t, _):
            val = _ringdown_ansatz_phase_value(
                t,
                phase_coeffs.c1_prec,
                phase_coeffs.c2,
                phase_coeffs.c3,
                phase_coeffs.c4,
                phase_coeffs.omegaRING_prec,
                phase_coeffs.phOffRD,
            )
            return val - phase_coeffs.phiCutPNAMP

        # Use lax.switch which is cleaner than nested conds
        # We need to pass _phase_22 to all, even if unused, to match signature
        return jax.lax.switch(
            region_idx, [_inspiral, _intermediate, _ringdown], t, _phase_22
        )

    # Vectorize over time array
    time = jnp.asarray(time)
    phase_22 = jnp.asarray(phase_22)
    time_shape = jnp.shape(time)
    time_flat = jnp.reshape(time, (-1,))

    # phase_22 may be a scalar or an array; broadcast to match time_flat shape
    if jnp.ndim(phase_22) == 0:
        phase_22_flat = jnp.full_like(time_flat, phase_22)
    else:
        phase_22_flat = jnp.reshape(phase_22, (-1,))

    phases_flat = jax.vmap(_phase_scalar, in_axes=(0, 0))(time_flat, phase_22_flat)
    phases = jnp.reshape(phases_flat, time_shape)

    return phases


@jax.jit
def get_time_of_frequency(
    freq: float | Array,
    eta: float | Array,
    phase_coeffs: PhaseCoeffs,
    t_low: float | Array = 0.0,
    t_high: float | Array = 500.0,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> float | Array:
    """
    Get time corresponding to a given frequency using root finding.

    Parameters
    ----------
    freq : float | Array
        (Dimensionless) frequency at which to find the corresponding time.
    eta : float | Array
        Symmetric mass ratio.
    phase_coeffs : PhaseCoeffs
        Phase coefficients for the mode.
    t_low : float | Array, optional
        Lower bound for the time search (default is 0.0. In this case, it is adjusted based on the frequency).
    t_high : float | Array, optional
        Upper bound for the time search (default is 500.0).
    atol : float | Array, optional
        Absolute Bisection tolerance
    rtol : float | Array, optional
        Relative bisection tolerance
    """

    t_low = jax.lax.cond(
        t_low == 0,
        lambda: -0.015 * freq ** (-2.7),  # enlarging this a bit
        lambda: t_low,
    )

    def time_of_freq(t, freq):
        time = jax.lax.cond(
            t < phase_coeffs.tEarly,
            lambda t: t - phase_coeffs.tt0,
            lambda t: t,
            t,
        )
        omega = imr_omega(time, eta, phase_coeffs)
        return 2 * jnp.pi * freq - omega

    solver = optx.Bisection(  # type: ignore
        atol=atol,
        rtol=rtol,
    )
    time_root: optx.Solution = optx.root_find(
        time_of_freq,
        solver,
        args=freq,
        y0=-0.01 * freq ** (-2.7),
        options={"lower": t_low, "upper": t_high},
        max_steps=1000,
    )

    return time_root.value


# =============================================================================
# Helper functions
# =============================================================================


@jax.jit
def compute_domega_cut(
    tCut: float | Array,
    tCut_threshold: float | Array,
    eta: float | Array,
    phase_coeffs_22: PhaseCoeffs,
) -> float | Array:
    """
    Compute domegaCut using JAX conditional.

    Parameters
    ----------
    tCut : float | Array
        Transition time inspiral -> intermediate.
    tCut_threshold : float | Array
        Threshold time to switch between inspiral and merger branch.
    eta : float | Array
        Symmetric mass ratio.
    phase_coeffs_22 : PhaseCoeffs

    """

    def inspiral_branch(t):
        omega_pn_coefficients = jnp.array(
            [
                phase_coeffs_22.omega1PN,
                phase_coeffs_22.omega1halfPN,
                phase_coeffs_22.omega2PN,
                phase_coeffs_22.omega2halfPN,
                phase_coeffs_22.omega3PN,
                phase_coeffs_22.omega3halfPN,
            ]
        )

        omega_pseudo_pn_coefficients = jnp.array(
            [
                phase_coeffs_22.omegaInspC1,
                phase_coeffs_22.omegaInspC2,
                phase_coeffs_22.omegaInspC3,
                phase_coeffs_22.omegaInspC4,
                phase_coeffs_22.omegaInspC5,
                phase_coeffs_22.omegaInspC6,
            ]
        )

        return _inspiral_ansatz_domega(
            t, eta, omega_pn_coefficients, omega_pseudo_pn_coefficients
        )

    def merger_branch(t):
        arcsinh = jnp.arcsinh(phase_coeffs_22.alpha1RD * t)
        return (
            -phase_coeffs_22.omegaRING
            / jnp.sqrt(1.0 + (phase_coeffs_22.alpha1RD * t) ** 2)
            * (
                phase_coeffs_22.domegaPeak
                + phase_coeffs_22.alpha1RD
                * (
                    2.0 * phase_coeffs_22.omegaMergerC1 * arcsinh
                    + 3.0 * phase_coeffs_22.omegaMergerC2 * arcsinh * arcsinh
                    + 4.0 * phase_coeffs_22.omegaMergerC3 * arcsinh**3
                )
            )
        )

    return jax.lax.cond(tCut < tCut_threshold, inspiral_branch, merger_branch, tCut)


@jax.jit
def _get_offset(mode: int | Array) -> Array:
    """Get mode-dependent offset for phase computation."""

    phoff = jax.lax.cond(
        mode == 33,
        lambda: -jnp.pi * 0.5,
        lambda: jax.lax.cond(
            mode == 44,
            lambda: jnp.pi,
            lambda: jax.lax.cond(
                mode == 55,
                lambda: jnp.pi * 0.5,
                lambda: jax.lax.cond(
                    mode == 21,
                    lambda: jnp.pi * 0.5,
                    lambda: jnp.array(0.0),
                ),
            ),
        ),
    )

    return phoff


# =============================================================================
# Helper functions for ansatz evaluation
# =============================================================================


@jax.jit
def _inspiral_ansatz_omega_single(
    time: Array,
    eta: Array,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
    m: int | Array = 2,
) -> Array:
    """Evaluate inspiral omega ansatz at a single time."""
    theta = jnp.power(-eta * time / 5.0, -1.0 / 8.0)

    # TaylorT3 part
    taylort3 = collocation.pn_ansatz_omega(theta, omega_pn_coeffs)

    # Pseudo-PN part
    theta8 = jnp.power(theta, 8)
    theta9 = theta8 * theta
    theta10 = theta9 * theta
    theta11 = theta10 * theta
    theta12 = theta11 * theta
    theta13 = theta12 * theta

    fac = theta * theta * theta / 8.0
    pseudo_pn_sum = (
        omega_pseudo_pn_coeffs[0] * theta8
        + omega_pseudo_pn_coeffs[1] * theta9
        + omega_pseudo_pn_coeffs[2] * theta10
        + omega_pseudo_pn_coeffs[3] * theta11
        + omega_pseudo_pn_coeffs[4] * theta12
        + omega_pseudo_pn_coeffs[5] * theta13
    )

    return (taylort3 + 2.0 * fac * pseudo_pn_sum) * (m / 2.0)


@jax.jit
def _intermediate_ansatz_omega_single(
    time: Array,
    alpha1RD: Array,
    omegaPeak: Array,
    domegaPeak: Array,
    omegaRING: Array,
    omegaMergerC1: Array,
    omegaMergerC2: Array,
    omegaMergerC3: Array,
):
    x = jnp.arcsinh(alpha1RD * time)
    w = (
        1
        - omegaPeak / omegaRING
        + x
        * (
            domegaPeak / alpha1RD
            + x * (omegaMergerC1 + x * (omegaMergerC2 + x * omegaMergerC3))
        )
    )

    return omegaRING * (1 - w)


@jax.jit
def _ringdown_ansatz_omega_single(
    time: Array,
    c1: Array,
    c2: Array,
    c3: Array,
    c4: Array,
    omegaRING: Array,
) -> Array:
    """Evaluate ringdown omega ansatz at a single time."""
    expC = jnp.exp(-c2 * time)
    expC2 = expC * expC
    num = -c1 * c2 * (2 * c4 * expC2 + c3 * expC)
    den = 1 + c4 * expC2 + c3 * expC

    return num / den + omegaRING


@jax.jit
def _inspiral_ansatz_domega(
    time: Array,
    eta: Array,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
    m: int | Array = 2,
) -> Array:
    """Compute d(omega)/dt for inspiral ansatz at a single time."""
    # Use JAX autodiff
    return jax.grad(
        lambda t: _inspiral_ansatz_omega_single(
            t, eta, omega_pn_coeffs, omega_pseudo_pn_coeffs, m
        )
    )(time)


@jax.jit
def _ringdown_ansatz_domega(
    time: Array,
    c1: Array,
    c2: Array,
    c3: Array,
    c4: Array,
) -> Array:
    """Compute d(omega)/dt for ringdown ansatz at a single time."""
    expC = jnp.exp(c2 * time)
    expC2 = expC * expC

    num = c1 * c2 * c2 * expC * (4 * c4 * expC + c3 * (c4 + expC2))
    den = c4 + expC * (c3 + expC)
    return num / (den * den)


@jax.jit
def _solve_intermediate_omega_system(
    alpha1RD: Array,
    tCut: Array,
    tcpMerger: Array,
    omegaCutBar: Array,
    omegaMergerCP: Array,
    domegaCut: Array,
    domegaPeak: Array,
    omegaPeak: Array,
    omegaRING: Array,
) -> tuple:
    """Solve 3x3 linear system for intermediate omega coefficients."""
    # System from Eqs. 28, 29, 31 in arXiv:2012.11923

    ascut = jnp.arcsinh(alpha1RD * tCut)
    ascut2 = ascut * ascut
    ascut3 = ascut * ascut2
    ascut4 = ascut * ascut3

    bascut = jnp.arcsinh(alpha1RD * tcpMerger)
    bascut2 = bascut * bascut
    bascut3 = bascut * bascut2
    bascut4 = bascut * bascut3

    dencut = jnp.sqrt(1.0 + tCut * tCut * alpha1RD * alpha1RD)

    # Build matrix and RHS

    B = jnp.array(
        [
            omegaCutBar
            - (1.0 - omegaPeak / omegaRING)
            - (domegaPeak / alpha1RD) * ascut,
            omegaMergerCP
            - (1.0 - omegaPeak / omegaRING)
            - (domegaPeak / alpha1RD) * bascut,
            domegaCut - domegaPeak / dencut,
        ]
    )

    matrix = jnp.array(
        [
            jnp.array([ascut2, ascut3, ascut4]),
            jnp.array([bascut2, bascut3, bascut4]),
            jnp.array(
                [
                    2.0 * alpha1RD * ascut / dencut,
                    3.0 * alpha1RD * ascut2 / dencut,
                    4.0 * alpha1RD * ascut3 / dencut,
                ]
            ),
        ]
    )

    # Solve
    solution = solve_3x3_explicit(matrix, B)  # jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2]


@jax.jit
def _pn_ansatz_phase(
    thetabar: Array,
    eta: Array,
    powers_of_5: jnp.ndarray,
    omega_pn_coeffs: jnp.ndarray,
) -> Array:
    """Evaluate PN ansatz phase at thetabar."""
    # This is the integration of the PN omega ansatz
    # From phenomxpy numba_pn_ansatz_22_phase
    thetabar = thetabar * powers_of_5[1]
    thetabar2 = thetabar * thetabar
    thetabar3 = thetabar * thetabar2
    thetabar4 = thetabar * thetabar3
    thetabar5 = thetabar * thetabar4
    thetabar6 = thetabar * thetabar5
    thetabar7 = thetabar * thetabar6
    logthetabar = jnp.log(thetabar)

    aux = (
        1
        / eta
        / thetabar5
        * (
            -168
            - 280 * omega_pn_coeffs[0] * thetabar2
            - 420 * omega_pn_coeffs[1] * thetabar3
            - 840 * omega_pn_coeffs[2] * thetabar4
            + 840 * omega_pn_coeffs[3] * (logthetabar - 0.125 * jnp.log(5)) * thetabar5
            - 321 * thetabar6
            + 840 * omega_pn_coeffs[4] * thetabar6
            + 321 * logthetabar * thetabar6
            + 420 * omega_pn_coeffs[5] * thetabar7
        )
    ) / 84.0

    return aux


@jax.jit
def _inspiral_ansatz_phase_value_22(
    time: Array,
    eta: Array,
    powers_of_5: jnp.ndarray,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
    phOffInsp: Array,
) -> Array:
    """Evaluate inspiral ansatz phase at a single time."""

    thetabar = jnp.power(-eta * time, -1.0 / 8.0)
    thetabar2 = thetabar * thetabar
    thetabar3 = thetabar * thetabar2
    thetabar4 = thetabar * thetabar3
    thetabar5 = thetabar * thetabar4
    thetabar6 = thetabar * thetabar5
    thetabar7 = thetabar * thetabar6

    logmtime = jnp.log(-time)
    log_theta_bar = jnp.log(jnp.power(5, 0.125)) - 0.125 * (jnp.log(eta) + logmtime)

    aux = (
        -(
            1
            / powers_of_5[5]
            / (eta * eta)
            / time
            / thetabar7
            * (
                3 * (-107 + 280 * omega_pn_coeffs[4]) * powers_of_5[6]
                + 321 * log_theta_bar * powers_of_5[6]
                + 420 * omega_pn_coeffs[5] * thetabar * powers_of_5[7]
                + 56 * (25 * omega_pseudo_pn_coeffs[0] + 3 * eta * time) * thetabar2
                + 1050 * omega_pseudo_pn_coeffs[1] * powers_of_5[1] * thetabar3
                + 280
                * (3 * omega_pseudo_pn_coeffs[2] + eta * omega_pn_coeffs[0] * time)
                * powers_of_5[2]
                * thetabar4
                + 140
                * (5 * omega_pseudo_pn_coeffs[3] + 3 * eta * omega_pn_coeffs[1] * time)
                * powers_of_5[3]
                * thetabar5
                + 120
                * (5 * omega_pseudo_pn_coeffs[4] + 7 * eta * omega_pn_coeffs[2] * time)
                * powers_of_5[4]
                * thetabar6
                + 525 * omega_pseudo_pn_coeffs[5] * powers_of_5[5] * thetabar7
                + 105
                * eta
                * omega_pn_coeffs[3]
                * time
                * logmtime
                * powers_of_5[5]
                * thetabar7
            )
        )
        / 84.0
    )

    return aux + phOffInsp


@jax.jit
def _inspiral_ansatz_phase_hm(
    phase_22: Array,
    mode: int | Array,
):
    """Compute inspiral ansatz phase for HM modes based on the 22 mode phase."""
    m = mode % 10
    return (m / 2.0) * phase_22


def _inspiral_ansatz_phase_value(
    time: Array,
    eta: Array,
    powers_of_5: jnp.ndarray,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
    phOffInsp: Array,
    mode: int | Array,
    phase_22: float | Array = 0.0,
) -> Array:
    """Evaluate inspiral ansatz phase at a single time for given mode."""

    phase = jax.lax.cond(
        mode == 22,
        lambda: _inspiral_ansatz_phase_value_22(
            time,
            eta,
            powers_of_5,
            omega_pn_coeffs,
            omega_pseudo_pn_coeffs,
            phOffInsp,
        ),
        lambda: _inspiral_ansatz_phase_hm(
            phase_22,
            mode,
        ),
    )
    return phase


@jax.jit
def _intermediate_ansatz_phase_value(
    time: Array,
    alpha1RD: Array,
    omegaMergerC1: Array,
    omegaMergerC2: Array,
    omegaMergerC3: Array,
    omegaPeak: Array,
    domegaPeak: Array,
    omegaRING: Array,
    phOffMerger: Array,
) -> Array:
    """Evaluate intermediate ansatz phase at a single time."""
    x = jnp.arcsinh(alpha1RD * time)
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    term1 = jnp.sqrt(1.0 + (alpha1RD * alpha1RD) * time * time)

    aux = omegaRING * time * (
        1.0
        - (
            2.0 * omegaMergerC1
            + 24.0 * omegaMergerC3
            + (6.0 * omegaMergerC2 + domegaPeak / alpha1RD) * x
            + (1.0 - omegaPeak / omegaRING)
            + (omegaMergerC1 + 12.0 * omegaMergerC3) * x2
            + omegaMergerC2 * x3
            + omegaMergerC3 * x4
        )
    ) - (omegaRING / alpha1RD) * term1 * (
        -domegaPeak / alpha1RD
        - 6.0 * omegaMergerC2
        - x * (2.0 * omegaMergerC1 + 24.0 * omegaMergerC3)
        - 3.0 * omegaMergerC2 * x2
        - 4.0 * omegaMergerC3 * x3
    )

    return aux + phOffMerger


@jax.jit
def _ringdown_ansatz_phase_value(
    time: Array,
    c1_prec: Array,
    c2: Array,
    c3: Array,
    c4: Array,
    omegaRING_prec: Array,
    phOffRD: Array,
) -> Array:
    """Evaluate ringdown ansatz phase at a single time."""
    expC = jnp.exp(-c2 * time)
    num = 1 + c3 * expC + c4 * expC * expC
    den = 1 + c3 + c4
    aux = jnp.log(num / den)
    return c1_prec * aux + omegaRING_prec * time + phOffRD
