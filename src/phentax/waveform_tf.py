# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós


"""
Time-frequency waveforms
================================

IMRPhenomTHM_TF interface class for time-frequency waveform generation.
Subclasses IMRPhenomTHM and adds Fresnel representations, intended for pre-merger signals.
"""

from functools import partial
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtyping import Array

from phentax.core import (
    AmplitudeCoeffs,
    PhaseCoeffs,
    compute_amplitude_coeffs_hm,
    compute_phase_coeffs_hm,
    imr_omega,
    imr_omega_dot,
)
from phentax.core.internals import WaveformParams
from phentax.utils.config import setup_logging
from phentax.utils.utility import (
    df_dt_to_Hz_squared,
    mass_to_hz,
    mass_to_second,
    second_to_mass,
)
from phentax.utils.ylm import spin_weighted_spherical_harmonic_all_modes
from phentax.waveform import IMRPhenomTHM

logger = setup_logging(__name__)

ALLOWED_POSITIVE_HMS = [21, 33, 44, 55]


class IMRPhenomTHM_TF(IMRPhenomTHM):
    """
    IMRPhenomTHM_TF class for time-frequency waveform generation.

    Subclasses IMRPhenomTHM and adds Fresnel representations of h_plus/h_cross in the STFT
    domain, computed directly in time-frequency. Current implementations:
        - `get_tf_fresnel_waveform_vanilla_TF`: box-car window, waveform parameters defined
          at the beginning of each segment.
        - `get_tf_fresnel_tukey_midpoint`: Tukey window, waveform parameters defined at the
          middle of each segment (more accurate).
    """

    def __init__(
        self,
        higher_modes: Optional[Array | list | str] = "all",
        include_negative_modes: bool = True,
        coarse_grain: bool = False,
        t_low_fit: bool = True,  # Use default fit for t_low if True.
        atol: float = 1e-12,
        rtol: float = 1e-12,
    ):
        super().__init__(
            higher_modes=higher_modes,
            include_negative_modes=include_negative_modes,
            coarse_grain=coarse_grain,
            t_low_fit=t_low_fit,
            atol=atol,
            rtol=rtol,
            # Observation time is unused by the TF methods; 1 month is a placeholder.
            T=1 * 30 * 24 * 3600,
        )

    def initial_processing_TF(
        self,
        m1: float | Array,
        m2: float | Array,
        chi1z: float | Array,
        chi2z: float | Array,
        distance: float | Array,
        phi_ref: float | Array,
        f_ref: float | Array,
        f_min: float | Array,
        inclination: float | Array,
        psi: float | Array,
        delta_t: float | Array = 15.0,
        t_min: float | Array = jnp.nan,
        t_ref: float | Array = jnp.nan,
    ) -> tuple[WaveformParams, AmplitudeCoeffs, PhaseCoeffs]:
        """
        Initial processing to compute waveform parameters.

        Unlike `initial_processing`, no time array is computed here, as TF waveforms may use
        different time grids. Only the following are computed:
            - Transformed waveform parameters
            - Amplitude and phase coefficients for the (2,2) mode

        Parameters
        ----------
        m1 : float | Array
            Mass of the first black hole in solar masses.
        m2 : float | Array
            Mass of the second black hole in solar masses.
        chi1z : float | Array
            Dimensionless spin of the first black hole along the orbital angular momentum.
        chi2z : float | Array
            Dimensionless spin of the second black hole along the orbital angular momentum.
        distance : float | Array
            Luminosity distance to the binary in megaparsecs.
        phi_ref : float | Array
            Reference phase at frequency f_ref in radians.
        f_ref : float | Array
            Reference frequency in Hz.
        f_min : float | Array
            Minimum frequency in Hz.
        inclination : float | Array
            Inclination angle of the binary in radians.
        psi : float | Array
            Polarization angle in radians.
        delta_t : float | Array, default 15.0
            Time step in seconds. Only used to round t_min to a multiple of delta_t.
        t_min : float | Array, default jnp.nan
            Minimum time for waveform generation in seconds. If NaN, will be set by the minimum frequency.
        t_ref : float | Array, default jnp.nan
            Reference time for waveform generation in seconds. If NaN, will be set by the reference frequency.

        Returns
        -------
        wf_params : WaveformParams
            Waveform parameters of the binary, including derived parameters like total mass
            and symmetric mass ratio.
        amplitude_coeffs_22 : AmplitudeCoeffs
            Amplitude coefficients for the (2,2) mode.
        phase_coeffs_22 : PhaseCoeffs
            Phase coefficients for the (2,2) mode.
        """

        wf_params = self._process_parameters(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref,
            inclination,
            psi,
            delta_t,
            t_min,
            t_ref,
            f_min,
            f_ref,
        )
        wf_params, amplitude_coeffs_22, phase_coeffs_22 = jax.vmap(
            self._compute_coeffs_22
        )(wf_params)

        return wf_params, amplitude_coeffs_22, phase_coeffs_22

    @jax.jit(static_argnames="self")
    def _compute_phase_coeffs_hm(
        self,
        mode: int | Array,
        wf_params: WaveformParams,
        phase_coeffs_22: PhaseCoeffs,
    ) -> tuple[PhaseCoeffs, AmplitudeCoeffs]:
        """
        Compute the phase and amplitude coefficients for a given higher mode (beyond 22).

        Parameters
        ----------
        mode : int | Array
            The (l,m) mode to compute, encoded as lm (e.g., 21 for (2,1)).
        wf_params : WaveformParams
            Waveform parameters of a single binary.
        phase_coeffs_22 : PhaseCoeffs
            Phase coefficients for the (2,2) mode.

        Returns
        -------
        phase_coeffs : PhaseCoeffs
            Phase coefficients for the requested mode.
        amplitude_coeffs : AmplitudeCoeffs
            Amplitude coefficients for the requested mode.
        """
        amplitude_coeffs = compute_amplitude_coeffs_hm(wf_params, phase_coeffs_22, mode)
        phase_coeffs = compute_phase_coeffs_hm(
            wf_params,
            phase_coeffs_22,
            OmegaCutPNAMP=amplitude_coeffs.omegaCutPNAMP,
            PhiCutPNAMP=amplitude_coeffs.phiCutPNAMP,
            mode=mode,
        )

        return phase_coeffs, amplitude_coeffs

    def _place_waveform_at_time(
        self,
        waveform_modes: Array,
        indices_modes: Array,
        K_plus: Array,
        K_cross: Array,
        n_freq: int,
    ) -> tuple[Array, Array]:
        """
        Place waveforms from all modes into the frequency grid for a single time step.

        Parameters
        ----------
        waveform_modes : Array
            Complex waveform values, shape (n_modes, 2*closest_f_bins).
        indices_modes : Array
            Frequency grid index of each waveform value, shape (n_modes, 2*closest_f_bins).
        K_plus : Array
            Complex coefficients for the plus polarization, shape (n_modes,).
        K_cross : Array
            Complex coefficients for the cross polarization, shape (n_modes,).
        n_freq : int
            Number of bins in the frequency grid. Must be a Python int (static).

        Returns
        -------
        result_plus : Array
            Plus polarization summed across modes at the correct frequency positions, shape (n_freq,).
        result_cross : Array
            Cross polarization summed across modes at the correct frequency positions, shape (n_freq,).
        """
        # Multiply by K *before* flattening, so each mode's scalar broadcasts across
        # its 2*closest_f_bins frequency entries. Shape (n_modes, 2*closest_f_bins).
        h_plus_modes = K_plus[:, None] * waveform_modes
        h_cross_modes = K_cross[:, None] * waveform_modes

        # Flatten across modes and frequency bins.
        flat_h_plus = h_plus_modes.flatten()
        flat_h_cross = h_cross_modes.flatten()
        flat_indices = indices_modes.flatten()

        # Zero the contributions from out-of-bounds indices.
        valid_mask = (flat_indices >= 0) & (flat_indices < n_freq)
        flat_h_plus = jnp.where(valid_mask, flat_h_plus, 0.0)
        flat_h_cross = jnp.where(valid_mask, flat_h_cross, 0.0)

        # Clip indices to the valid range so .at[].add() stays in bounds; the zeroed
        # values above mean nothing is added for these entries.
        flat_indices = jnp.clip(flat_indices, 0, n_freq - 1)

        result_plus = (
            jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_plus)
        )
        result_cross = (
            jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_cross)
        )

        return result_plus, result_cross

    def _process_source(
        self,
        waveforms_per_source: Array,
        indices_per_source: Array,
        K_plus: Array,
        K_cross: Array,
        n_freq: int,
    ) -> tuple[Array, Array]:
        """
        Place the waveform values of a single source onto the frequency grid for all time steps.

        Parameters
        ----------
        waveforms_per_source : Array
            Waveform values for one source on the per-mode frequency bins,
            shape (n_times - 1, n_modes, 2*closest_f_bins).
        indices_per_source : Array
            Frequency grid index of each waveform value, shape (n_times - 1, n_modes, 2*closest_f_bins).
        K_plus : Array
            Complex coefficients for the plus polarization, shape (n_modes,).
        K_cross : Array
            Complex coefficients for the cross polarization, shape (n_modes,).
        n_freq : int
            Number of bins in the frequency grid. Must be a Python int (static).

        Returns
        -------
        tf_plus : Array
            Plus polarization TF map for this source, shape (n_times - 1, n_freq).
        tf_cross : Array
            Cross polarization TF map for this source, shape (n_times - 1, n_freq).
        """
        # vmap over time steps. n_freq is bound with partial so it stays a Python int.
        return jax.vmap(
            partial(self._place_waveform_at_time, n_freq=n_freq),
            in_axes=(0, 0, None, None),
        )(waveforms_per_source, indices_per_source, K_plus, K_cross)

    @jax.jit(static_argnums=[0, 16])
    def get_tf_fresnel_waveform_vanilla_TF(
        self,
        time_grid: Array,
        frequency_grid: Array,
        m1: float | Array,
        m2: float | Array,
        chi1z: float | Array,
        chi2z: float | Array,
        distance: float | Array,
        phi_ref: float | Array,
        f_ref: float | Array,
        f_min: float | Array,
        inclination: float | Array,
        psi: float | Array,
        delta_t: float = 15.0,
        t_min: float = jnp.nan,
        t_ref: float = jnp.nan,
        closest_f_bins: int = 10,
    ) -> tuple[Array, Array]:
        """
        Simplest implementation of time-frequency Fresnel waveform generation.

        Assumes a box-car window, and that the waveform parameters are defined at the
        beginning of each segment.

        Note: this function uses an un-optimized version of the Fresnel kernel, which is
        much more readable than the optimized version, so it is kept for reference.

        Time convention:
            - It is assumed that the time_grid supplied by the user is long enough to include
              the longest waveform in the batch.
            - The time grid is internally shifted so that grid time 0 is the start of the longest
              waveform in the batch; every source merges at the same grid time.
            - t(f_min) is computed for each source, but only the earliest one sets the grid start.

        Note on f_min for batches:
            Every source is evaluated from the start of the longest waveform, so for all other
            sources the early segments have a 22-mode frequency below f_min. This is still valid
            inspiral (t_min/f_min only set the start time, not the model coefficients), but f_min
            is then the starting frequency of the longest source only, and a mixed batch will not
            match the time-domain IMRPhenomTHM, which starts each source at its own t(f_min).
            For a batch-independent start, pass a scalar t_min instead.

            In comparison to "natural ground truth" which will have all the modes at all the times,
            this is a more natural construction.

        Parameters
        ----------
        time_grid : Array
            Time grid over which to compute the waveform, shape (n_times,). Units: seconds.
            Note: these are the segment *edges*, relative to the start of the longest waveform in the batch.
        frequency_grid : Array
            Frequency grid over which to compute the waveform, shape (n_freq,). Units: Hz.
            Must be uniformly spaced.
        m1 : float | Array
            Mass of the primary object, shape (num_sources,). Units: solar masses.
        m2 : float | Array
            Mass of the secondary object, shape (num_sources,). Units: solar masses.
        chi1z : float | Array
            Dimensionless spin of the primary object along the orbital angular momentum, shape (num_sources,).
        chi2z : float | Array
            Dimensionless spin of the secondary object along the orbital angular momentum, shape (num_sources,).
        distance : float | Array
            Luminosity distance to the source, shape (num_sources,). Units: Mpc.
        phi_ref : float | Array
            Reference phase of the waveform, shape (num_sources,). Units: radians.
        f_ref : float | Array
            Reference frequency at which phi_ref is defined. Units: Hz.
            Ignored if t_ref is supplied.
        f_min : float | Array
            22-mode frequency at which the longest source in the batch starts (see the note
            above). Units: Hz. Ignored if t_min is supplied.
        inclination : float | Array
            Inclination angle of the binary's orbital plane, shape (num_sources,). Units: radians.
        psi : float | Array
            Polarization angle of the waveform, shape (num_sources,). Units: radians.
        delta_t : float, optional
            Time step, by default 15.0. Units: seconds.
            Only rounds the start time t_min to a multiple of delta_t; it does not set any sampling.
        t_min : float, optional
            Start time of the waveform relative to merger, by default NaN. Units: seconds.
            If NaN, set by f_min instead.
        t_ref : float, optional
            Reference time at which phi_ref is defined, relative to merger, by default NaN. Units: seconds.
            If NaN, set by f_ref instead.
        closest_f_bins : int, optional
            Number of frequency bins on each side of the closest grid frequency to evaluate,
            for each source and mode, by default 10. Static argument.

        Returns
        -------
        tf_grid_plus : Array
            Time-frequency grid of the plus polarization, shape (num_sources, n_times - 1, n_freq). Units: strain.
            Row i is the segment [time_grid[i], time_grid[i+1]].
        tf_grid_cross : Array
            Time-frequency grid of the cross polarization, shape (num_sources, n_times - 1, n_freq). Units: strain.
            Row i is the segment [time_grid[i], time_grid[i+1]].
        """

        num_sources = jnp.atleast_1d(m1).shape[0]

        wf_params, amplitude_coeffs_22, phase_coeffs_22 = self.initial_processing_TF(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref,
            f_ref,
            f_min,
            inclination,
            psi,
            delta_t,
            t_min,
            t_ref,
        )

        # Time (relative to merger, in mass units) at which each source's 22 mode is at
        # f_min, shape (num_sources,). One value per source, not a batch minimum.
        min_time_M_units = wf_params.Mt_min

        # Convert each start time to seconds. mass_to_second expects (n_sources, n_times)
        # input (it broadcasts total_mass[:, None]), so add a time axis and drop it again;
        # passing the 1D array directly gives an (n_sources, n_sources) outer product.
        time_min_physical_all = mass_to_second(
            min_time_M_units[:, None], wf_params.total_mass
        )[:, 0]

        # Earliest (most negative) start time in the batch, in seconds: the start of the
        # longest waveform, where its 22 mode is at f_min. The minimum must be taken in
        # seconds, since mass units differ between sources.
        min_time_s_units = jnp.min(time_min_physical_all)

        # The time grid starts at 0, and every source's merger is at t = 0, so adding
        # min_time_s_units gives the same grid relative to merger, starting at the start of
        # the longest waveform. Units: seconds. Every other source is therefore evaluated
        # from before its own t(f_min), i.e. below f_min (see the docstring note).
        # time_grid itself is left unchanged; it is only used for segment lengths below.
        t_rel = time_grid + min_time_s_units

        # The merger-relative grid in mass units, separately for each binary,
        # shape (num_sources, n_times).
        time_grids_mass_units = jax.vmap(second_to_mass, in_axes=(None, 0))(
            t_rel, wf_params.total_mass
        )

        new_mask = jnp.ones_like(t_rel, dtype=bool)

        # Amplitudes and phases for every binary over the time grid.
        # new_mask is the same for all sources, hence in_axes=None for it.
        amplitudes, phases = jax.vmap(
            self._compute_all_modes, in_axes=(0, None, 0, 0, 0)
        )(
            time_grids_mass_units,
            new_mask,
            wf_params,
            amplitude_coeffs_22,
            phase_coeffs_22,
        )
        # Amplitudes and phases shape: (num_sources, num_modes, n_times)
        n_modes = amplitudes.shape[1]

        # Higher mode phase coefficients.
        phase_hm_coeffs, _ = jax.vmap(
            lambda wp, pc22: jax.vmap(
                lambda mode: self._compute_phase_coeffs_hm(mode, wp, pc22)
            )(self.higher_modes)
        )(wf_params, phase_coeffs_22)

        # Combine 22 and HM phase coefficients.
        overall_phase_coeffs = jax.tree_util.tree_map(
            lambda p22, phm: jnp.concatenate([p22[:, None], phm], axis=1),
            phase_coeffs_22,
            phase_hm_coeffs,
        )

        # amp_factor shape: (num_sources,) -> reshape for broadcasting.
        amplitudes *= wf_params.amp_factor[:, None, None]

        # Frequency grid spacing (assumed uniform).
        dF = frequency_grid[1] - frequency_grid[0]

        # Per-segment waveform values, placed onto the TF grid later.
        # Shape (n_times - 1, num_sources, n_modes, 2*closest_f_bins).
        waveform_storage = jnp.zeros(
            (time_grid.size - 1, num_sources, n_modes, 2 * closest_f_bins),
            dtype=jnp.complex128,
        )

        # Frequency grid indices corresponding to each entry of waveform_storage.
        frequency_indices_storage = jnp.zeros(
            (time_grid.size - 1, num_sources, n_modes, 2 * closest_f_bins),
            dtype=jnp.int32,
        )

        # This loop could also be vmapped over time steps, but it is kept as a loop for readability in this case.
        for time_index in range(
            time_grid.size - 1
        ):  # -1 since each segment ends at the next edge

            # Segment length, and the segment start in mass units for each binary,
            # shape (num_sources,).
            seg_len = time_grid[time_index + 1] - time_grid[time_index]
            t_0_mass = time_grids_mass_units[:, time_index]

            Amps = amplitudes[:, :, time_index]  # (num_sources, n_modes)
            Phases = phases[:, :, time_index]  # (num_sources, n_modes)

            # Compute f_0 and f_dot at the segment start for every (source, mode) combination.
            # overall_phase_coeffs holds the phase coefficients of all (positive) modes for
            # all binaries; each field has shape (num_sources, n_modes, ...).
            #
            # Outer vmap (over sources): for source i, t_0_mass[i] and eta[i] are scalars
            # and pc = overall_phase_coeffs[i] has shape (n_modes, ...) per field.
            # Inner vmap (over modes): evaluates imr_omega(t, eta, p) for the phase
            # coefficients p of each mode.
            #
            # Result: f_0 and f_dot have shape (num_sources, n_modes), in geometric units.
            f_0 = jax.vmap(
                lambda t, eta, pc: jax.vmap(lambda p: imr_omega(t, eta, p))(pc)
            )(t_0_mass, wf_params.eta, overall_phase_coeffs) / (2 * jnp.pi)

            f_dot = jax.vmap(
                lambda t, eta, pc: jax.vmap(lambda p: imr_omega_dot(t, eta, p))(pc)
            )(t_0_mass, wf_params.eta, overall_phase_coeffs) / (2 * jnp.pi)

            # Convert f_0 and f_dot to Hz and Hz^2 respectively.
            f_0 = jax.vmap(mass_to_hz, in_axes=(0, 0))(f_0, wf_params.total_mass)
            f_dot = jax.vmap(df_dt_to_Hz_squared, in_axes=(0, 0))(
                f_dot, wf_params.total_mass
            )

            # NOTE: bins around f_0 that fall outside the frequency grid are masked later.

            # Closest frequency grid index for each source and mode, shape (num_sources, n_modes).
            # Uniform grid, so use direct arithmetic instead of materializing an
            # (num_sources, n_modes, n_freq) tensor.
            closest_frequency_indexes = jnp.round(
                (f_0 - frequency_grid[0]) / dF
            ).astype(jnp.int32)
            closest_frequency_indexes = jnp.clip(
                closest_frequency_indexes, 0, frequency_grid.size - 1
            )
            closest_frequencies = frequency_grid[closest_frequency_indexes]

            # Frequencies from closest_frequencies - closest_f_bins*dF up to
            # closest_frequencies + (closest_f_bins - 1)*dF in steps of dF,
            # shape (num_sources, n_modes, 2*closest_f_bins).
            frequencies = jax.vmap(
                lambda center_freqs: jnp.arange(
                    0,
                    2 * closest_f_bins,
                )
                * dF
                + center_freqs
                - closest_f_bins * dF,
            )(closest_frequencies.flatten()).reshape(
                num_sources, n_modes, 2 * closest_f_bins
            )

            # Frequency grid index of each entry of frequencies, used to place the
            # waveform values in the final TF grid.
            frequency_indices = jax.vmap(
                lambda indices: jnp.arange(
                    0,
                    2 * closest_f_bins,
                )
                + indices
                - closest_f_bins,
            )(closest_frequency_indexes.flatten()).reshape(
                num_sources, n_modes, 2 * closest_f_bins
            )

            # Amps, Phases, f_0 and f_dot have one value per (source, mode), while
            # frequencies has an extra frequency axis, so add a new axis for broadcasting.
            h_prefactor = (
                Amps[:, :, jnp.newaxis]
                * jnp.exp(1j * Phases[:, :, jnp.newaxis])
                / jnp.sqrt(2 * f_dot[:, :, jnp.newaxis])
                * jnp.exp(
                    -1j
                    * jnp.pi
                    * ((f_0[:, :, jnp.newaxis] - frequencies) ** 2)
                    / f_dot[:, :, jnp.newaxis]
                )
            )

            # Fresnel integrals over the segment, from tau = 0 (segment start) to tau = seg_len.
            v_nm_end = v_tukey(f_dot, seg_len, frequencies, f_0, 0.0)
            v_nm_begin = v_tukey(f_dot, 0.0, frequencies, f_0, 0.0)
            S_vn_end, C_vn_end = jax.scipy.special.fresnel(v_nm_end)
            S_vn_begin, C_vn_begin = jax.scipy.special.fresnel(v_nm_begin)

            I = C_vn_end - C_vn_begin + 1j * (S_vn_end - S_vn_begin)
            waveform = h_prefactor * I

            # NOTE: this is *not* yet the TF grid: the frequency axis holds the bins around
            # each mode's f_0, and is mapped onto the frequency grid below.
            waveform_storage = waveform_storage.at[time_index, :, :, :].set(waveform)
            frequency_indices_storage = frequency_indices_storage.at[
                time_index, :, :, :
            ].set(frequency_indices)

        # NOTE: in principle the likelihood/inner product could be computed directly from
        # waveform_storage, without building the full TF grid.

        # Transpose to (num_sources, n_times - 1, n_modes, 2*closest_f_bins) for vmapping.
        waveform_storage_transposed = waveform_storage.transpose(1, 0, 2, 3)
        frequency_indices_transposed = frequency_indices_storage.transpose(1, 0, 2, 3)

        # Number of frequency bins in the full grid.
        n_freq = frequency_grid.size

        # Spherical harmonics for all modes and sources, shape (num_sources, n_modes, 1).
        y_lms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.ells,
            self.mms,
        )

        y_lmms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.negative_ls,
            self.negative_mms,
        )

        # Shapes of K are (num_sources, n_modes), where n_modes includes only the positive
        # modes (negative modes are obtained by reflection symmetry for non-precessing binaries).
        # Overall conjugate to flip the Fourier convention (compared to the Marsat appendix).
        K_plus_lms = (
            1
            / 2
            * (
                y_lms[:, :, 0] + (-1) ** self.negative_ls * y_lmms[:, :, 0].conj()
            ).conj()
        )
        # Overall conjugate (including the 1j prefactor) to flip the Fourier convention.
        K_cross_lms = (
            1j
            / 2
            * (y_lms[:, :, 0] - (-1) ** self.negative_ls * y_lmms[:, :, 0].conj())
        ).conj()

        # vmap over sources. Together with the vmap over time steps in _process_source,
        # this gives TF grids of shape (num_sources, n_times - 1, n_freq).
        tf_grid_plus, tf_grid_cross = jax.vmap(
            partial(self._process_source, n_freq=n_freq)
        )(
            waveform_storage_transposed,
            frequency_indices_transposed,
            K_plus_lms,
            K_cross_lms,
        )

        # Rotate by polarization angle psi for each source (same for all modes).
        tf_grid_plus, tf_grid_cross = jax.vmap(self.rotate_by_polarization_angle)(
            tf_grid_plus, tf_grid_cross, wf_params.psi
        )

        return tf_grid_plus, tf_grid_cross

    @jax.jit(static_argnums=[0, 15, 16])
    def get_tf_fresnel_tukey_midpoint(
        self,
        time_grid: Array,
        frequency_grid: Array,
        m1: float | Array,
        m2: float | Array,
        chi1z: float | Array,
        chi2z: float | Array,
        distance: float | Array,
        phi_ref: float | Array,
        f_ref: float | Array,
        f_min: float | Array,
        inclination: float | Array,
        psi: float | Array,
        t_min: float = jnp.nan,
        t_ref: float = jnp.nan,
        closest_f_bins: int = 10,
        tukey_alpha: float = 0.5,
    ) -> tuple[Array, Array]:
        """
        Time-frequency Fresnel waveform generation with a Tukey window, whose frequency
        domain correction is taken into account analytically.
        Returns the source-frame polarizations, no detector response applied.

        Waveform parameters are defined at the centre of each time segment.

        Time convention:
            - It is assumed that the time_grid supplied by the user is long enough to include
              the longest waveform in the batch.
            - The time grid is internally shifted so that grid time 0 is the start of the longest
              waveform in the batch; every source merges at the same grid time.
            - t(f_min) is computed for each source, but only the earliest one sets the grid start.

        Note on f_min for batches:
            Every source is evaluated from the start of the longest waveform, so for all other
            sources the early segments have a 22-mode frequency below f_min. This is still valid
            inspiral (t_min/f_min only set the start time, not the model coefficients), but f_min
            is then the starting frequency of the longest source only, and a mixed batch will not
            match the time-domain IMRPhenomTHM, which starts each source at its own t(f_min).
            For a batch-independent start, pass a scalar t_min instead.

        Parameters
        ----------
        time_grid : Array
            Time grid over which to compute the waveform, shape (n_times,). Units: seconds.
            Note: these are the segment *edges*, relative to the start of the longest waveform in the batch,
            so the output has n_times - 1 segments. Must be uniformly spaced.
        frequency_grid : Array
            Frequency grid over which to compute the waveform, shape (n_freq,). Units: Hz.
            Must be uniformly spaced.
        m1 : float | Array
            Mass of the primary object, shape (num_sources,). Units: solar masses.
        m2 : float | Array
            Mass of the secondary object, shape (num_sources,). Units: solar masses.
        chi1z : float | Array
            Dimensionless spin of the primary object along the orbital angular momentum, shape (num_sources,).
        chi2z : float | Array
            Dimensionless spin of the secondary object along the orbital angular momentum, shape (num_sources,).
        distance : float | Array
            Luminosity distance to the source, shape (num_sources,). Units: Mpc.
        phi_ref : float | Array
            Reference phase of the waveform, shape (num_sources,). Units: radians.
        f_ref : float | Array
            Reference frequency at which phi_ref is defined. Units: Hz.
            Ignored if t_ref is supplied.
        f_min : float | Array
            22-mode frequency at which the longest source in the batch starts (see the note
            above). Units: Hz. Ignored if t_min is supplied.
        inclination : float | Array
            Inclination angle of the binary's orbital plane, shape (num_sources,). Units: radians.
        psi : float | Array
            Polarization angle of the waveform, shape (num_sources,). Units: radians.
        t_min : float, optional
            Start time of the waveform relative to merger, by default NaN. Units: seconds.
            If NaN, set by f_min instead.
        t_ref : float, optional
            Reference time at which phi_ref is defined, relative to merger, by default NaN. Units: seconds.
            If NaN, set by f_ref instead.
        closest_f_bins : int, optional
            Number of frequency bins on each side of the closest grid frequency to evaluate,
            for each source and mode, by default 10. Static argument.
        tukey_alpha : float, optional
            Alpha parameter for the Tukey window, by default 0.5. Static argument, so each
            new value triggers a recompilation.

        Returns
        -------
        tf_grid_plus : Array
            Time-frequency grid of the plus polarization, shape (num_sources, n_times - 1, n_freq). Units: strain.
        tf_grid_cross : Array
            Time-frequency grid of the cross polarization, shape (num_sources, n_times - 1, n_freq). Units: strain.
        """

        wf_params, amplitude_coeffs_22, phase_coeffs_22 = self.initial_processing_TF(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref,
            f_ref,
            f_min,
            inclination,
            psi,
            # delta_t: in the time domain this is the sampling cadence; here it only rounds
            # the waveform start time to a multiple of delta_t, so delta_t=1 is safe.
            1,
            t_min,
            t_ref,
        )

        # Time (relative to merger, in mass units) at which each source's 22 mode is at
        # f_min, shape (num_sources,). One value per source, not a batch minimum.
        min_time_M_units = wf_params.Mt_min

        # Convert each start time to seconds. mass_to_second expects (n_sources, n_times)
        # input (it broadcasts total_mass[:, None]), so add a time axis and drop it again;
        # passing the 1D array directly gives an (n_sources, n_sources) outer product.
        time_min_physical_all = mass_to_second(
            min_time_M_units[:, None], wf_params.total_mass
        )[:, 0]

        # Earliest (most negative) start time in the batch, in seconds: the start of the
        # longest waveform, where its 22 mode is at f_min. The minimum must be taken in
        # seconds, since mass units differ between sources.
        min_time_s_units = jnp.min(time_min_physical_all)

        # The time grid starts at 0, and every source's merger is at t = 0, so adding
        # min_time_s_units gives the same grid relative to merger, starting at the start of
        # the longest waveform. Units: seconds. Every other source is therefore evaluated
        # from before its own t(f_min), i.e. below f_min (see the docstring note).
        # time_grid itself is left unchanged; it is only used for the segment width below.
        t_rel = time_grid + min_time_s_units

        # Segment midpoints relative to merger, shape (n_times - 1,).
        t_grid_midpoints = 0.5 * (t_rel[:-1] + t_rel[1:])

        # Midpoints in mass units, separately for each binary, shape (num_sources, n_times - 1).
        t_grid_midpoints_mass_units = jax.vmap(second_to_mass, in_axes=(None, 0))(
            t_grid_midpoints, wf_params.total_mass
        )
        midpoint_mask = jnp.ones_like(t_grid_midpoints, dtype=bool)

        # Amplitudes and phases at the midpoint of each segment for each binary,
        # shape (num_sources, n_modes, n_times - 1).
        amplitudes, phases = jax.vmap(
            self._compute_all_modes, in_axes=(0, None, 0, 0, 0)
        )(
            t_grid_midpoints_mass_units,
            midpoint_mask,
            wf_params,
            amplitude_coeffs_22,
            phase_coeffs_22,
        )

        n_modes = amplitudes.shape[1]

        # Higher mode phase coefficients.
        phase_hm_coeffs, _ = jax.vmap(
            lambda wp, pc22: jax.vmap(
                lambda mode: self._compute_phase_coeffs_hm(mode, wp, pc22)
            )(self.higher_modes)
        )(wf_params, phase_coeffs_22)

        # Combine 22 and HM phase coefficients.
        overall_phase_coeffs = jax.tree_util.tree_map(
            lambda p22, phm: jnp.concatenate([p22[:, None], phm], axis=1),
            phase_coeffs_22,
            phase_hm_coeffs,
        )

        # amp_factor shape: (num_sources,) -> reshape for broadcasting.
        amplitudes *= wf_params.amp_factor[:, None, None]

        # Frequency grid spacing (assumed uniform).
        dF = frequency_grid[1] - frequency_grid[0]

        # Midpoints in mass units for computing f and f_dot, shape (n_times - 1, num_sources).
        tmid_mass_all = t_grid_midpoints_mass_units.T

        # Transpose to (n_times - 1, num_sources, n_modes) for scanning over time segments.
        amps_all = amplitudes.transpose(2, 0, 1)
        phases_all = phases.transpose(2, 0, 1)

        # f_0 and f_dot at the midpoints for all segments, sources and modes,
        # shape (n_times - 1, num_sources, n_modes), in geometric units.
        f_0_all = jax.vmap(
            lambda t_mid_mass: jax.vmap(
                lambda t, eta, pc: jax.vmap(lambda p: imr_omega(t, eta, p))(pc)
            )(t_mid_mass, wf_params.eta, overall_phase_coeffs)
        )(tmid_mass_all) / (2 * jnp.pi)

        f_dot_all = jax.vmap(
            lambda t_mid_mass: jax.vmap(
                lambda t, eta, pc: jax.vmap(lambda p: imr_omega_dot(t, eta, p))(pc)
            )(t_mid_mass, wf_params.eta, overall_phase_coeffs)
        )(tmid_mass_all) / (2 * jnp.pi)

        # Convert f_0 to Hz.
        f_0_all = jax.vmap(
            lambda f0_t: jax.vmap(mass_to_hz, in_axes=(0, 0))(
                f0_t, wf_params.total_mass
            )
        )(f_0_all)

        # Convert f_dot to Hz^2.
        f_dot_all = jax.vmap(
            lambda fd_t: jax.vmap(df_dt_to_Hz_squared, in_axes=(0, 0))(
                fd_t, wf_params.total_mass
            )
        )(f_dot_all)

        # Closest frequency grid index for each segment, source and mode.
        closest_frequency_indexes_all = jnp.round(
            (f_0_all - frequency_grid[0]) / dF
        ).astype(jnp.int32)

        # Clip to the valid range. For modes whose f_0 is off the grid this only moves the
        # window of evaluated bins; bins that fall off the grid are masked when placing
        # the values into the TF grid.
        closest_frequency_indexes_all = jnp.clip(
            closest_frequency_indexes_all, 0, frequency_grid.size - 1
        )

        # Closest grid frequencies for each segment, source and mode.
        closest_frequencies_all = frequency_grid[closest_frequency_indexes_all]

        # Offsets of the 2*closest_f_bins bins around each closest frequency.
        freq_offsets = jnp.arange(0, 2 * closest_f_bins, dtype=frequency_grid.dtype)
        idx_offsets = jnp.arange(0, 2 * closest_f_bins, dtype=jnp.int32)

        # Frequencies and grid indices of the bins around each closest frequency,
        # shape (n_times - 1, num_sources, n_modes, 2*closest_f_bins).
        frequencies_all = (
            closest_frequencies_all[:, :, :, None]
            + (freq_offsets[None, None, None, :] - closest_f_bins) * dF
        )
        frequency_indices_all = closest_frequency_indexes_all[:, :, :, None] + (
            idx_offsets[None, None, None, :] - closest_f_bins
        )

        # Width of each time segment (the time grid must be uniform).
        dT = time_grid[1] - time_grid[0]

        alpha_offset = 1 / (tukey_alpha * dT)

        def scan_step(_, scan_inputs):
            (
                Amps,
                Phases,
                f_0,
                f_dot,
                frequencies,
                frequency_indices,
            ) = scan_inputs

            # Vectorized Fresnel integral with a Tukey window.
            # Prefactor common to all terms of the integral. The last factor shifts the
            # time origin from the segment midpoint to the segment start (t_mid - t_0 = dT/2).
            h_prefactor = (
                Amps[:, :, jnp.newaxis]
                * jnp.exp(1j * Phases[:, :, jnp.newaxis])
                / jnp.sqrt(2 * f_dot[:, :, jnp.newaxis])
                * jnp.exp(-2 * 1j * jnp.pi * frequencies * (0.5 * dT))
            )

            # Phase factor for the flat part.
            normal_phase_prefactor = jnp.exp(
                -1j
                * jnp.pi
                * ((f_0[:, :, jnp.newaxis] - frequencies) ** 2)
                / f_dot[:, :, jnp.newaxis]
            )

            # Phase factors for the roll-on/roll-off terms, shifted by -/+ alpha_offset.
            phase_prefactor_minus = jnp.exp(
                -1j
                * jnp.pi
                * (f_0[:, :, jnp.newaxis] - frequencies - alpha_offset) ** 2
                / f_dot[:, :, jnp.newaxis]
            )
            phase_prefactor_plus = jnp.exp(
                -1j
                * jnp.pi
                * (f_0[:, :, jnp.newaxis] - frequencies + alpha_offset) ** 2
                / f_dot[:, :, jnp.newaxis]
            )

            # Edges of the Tukey window pieces relative to the segment midpoint:
            # roll-on over [tau0, tau1], flat over [tau1, tau2], roll-off over [tau2, tau3].
            tau0 = -0.5 * dT
            tau1 = 0.5 * (-dT + tukey_alpha * dT)
            tau2 = 0.5 * (dT - tukey_alpha * dT)
            tau3 = 0.5 * dT

            # All Fresnel arguments, shape (12, num_sources, n_modes, 2*closest_f_bins).
            # fmt: off
            args = jnp.stack(
                [v_tukey(f_dot, t, frequencies, f_0, 0.0) for t in (tau0, tau1, tau2, tau3)]            # Flat parts
                + [v_tukey(f_dot, t, frequencies, f_0, alpha_offset) for t in (tau0, tau1, tau2, tau3)]  # Tapers, +alpha_offset
                + [v_tukey(f_dot, t, frequencies, f_0, -alpha_offset) for t in (tau0, tau1, tau2, tau3)] # Tapers, -alpha_offset
            )
            # fmt: on

            S, C = jax.scipy.special.fresnel(args)

            F = C + 1j * S

            # Unpack the Fresnel terms.
            # fmt: off
            flat_t0,  flat_t1,  flat_t2,  flat_t3  = F[0],  F[1],  F[2],  F[3]
            plus_t0,  plus_t1,  plus_t2,  plus_t3  = F[4],  F[5],  F[6],  F[7]
            minus_t0, minus_t1, minus_t2, minus_t3 = F[8],  F[9],  F[10], F[11]
            # fmt: on

            W = jnp.exp(2j * jnp.pi / tukey_alpha * (0.5 - tukey_alpha / 2))
            W_conj = jnp.conj(W)

            # Each difference is the Fresnel term at the upper limit of a window piece
            # minus the term at its lower limit.
            # fmt: off
            # Roll-on terms
            h_flat_roll_on      = 0.5 * normal_phase_prefactor * (flat_t1  - flat_t0)
            h_roll_on_positive  = W      / 4 * phase_prefactor_plus  * (plus_t1  - plus_t0)
            h_roll_on_negative  = W_conj / 4 * phase_prefactor_minus * (minus_t1 - minus_t0)

            # Flat middle
            h_mid               =       normal_phase_prefactor * (flat_t2  - flat_t1)

            # Roll-off terms
            h_roll_off_flat     = 0.5 * normal_phase_prefactor * (flat_t3  - flat_t2)
            h_roll_off_positive = W_conj / 4 * phase_prefactor_plus  * (plus_t3  - plus_t2)
            h_roll_off_negative = W      / 4 * phase_prefactor_minus * (minus_t3 - minus_t2)
            # fmt: on

            # Sum the roll-on, flat and roll-off pieces to get the full Tukey-windowed integral.
            h_overall = h_prefactor * (
                h_flat_roll_on
                + h_roll_on_positive
                + h_roll_on_negative
                + h_mid
                + h_roll_off_flat
                + h_roll_off_positive
                + h_roll_off_negative
            )
            return None, (h_overall, frequency_indices)

        _, (waveform_steps, frequency_indices_steps) = jax.lax.scan(
            scan_step,
            None,
            (
                amps_all,
                phases_all,
                f_0_all,
                f_dot_all,
                frequencies_all,
                frequency_indices_all,
            ),
        )

        waveform_storage = waveform_steps
        frequency_indices_storage = frequency_indices_steps

        # NOTE: in principle the likelihood/inner product could be computed directly from
        # waveform_storage, without building the full TF grid.

        # Transpose to (num_sources, n_times - 1, n_modes, 2*closest_f_bins) for vmapping.
        waveform_storage_transposed = waveform_storage.transpose(1, 0, 2, 3)
        frequency_indices_transposed = frequency_indices_storage.transpose(1, 0, 2, 3)

        # Number of frequency bins in the full grid.
        n_freq = frequency_grid.size

        # Spherical harmonics for all modes and sources, shape (num_sources, n_modes, 1).
        y_lms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.ells,
            self.mms,
        )

        y_lmms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.negative_ls,
            self.negative_mms,
        )

        # Shapes of K are (num_sources, n_modes), where n_modes includes only the positive
        # modes (negative modes are obtained by reflection symmetry for non-precessing binaries).
        # Overall conjugate to flip the Fourier convention (compared to the Marsat appendix).
        K_plus_lms = (
            1
            / 2
            * (
                y_lms[:, :, 0] + (-1) ** self.negative_ls * y_lmms[:, :, 0].conj()
            ).conj()
        )
        # Overall conjugate (including the 1j prefactor) to flip the Fourier convention.
        K_cross_lms = (
            1j
            / 2
            * (y_lms[:, :, 0] - (-1) ** self.negative_ls * y_lmms[:, :, 0].conj())
        ).conj()

        # vmap over sources. Together with the vmap over time steps in _process_source,
        # this gives TF grids of shape (num_sources, n_times - 1, n_freq).
        # Partial as all others arguments have shapes (num_sources, n_modes, 2*closest_f_bins) and are constant for all time steps.
        # While n_freq is a constant for all sources. 
        tf_grid_plus, tf_grid_cross = jax.vmap(
            partial(self._process_source, n_freq=n_freq)
        )(
            waveform_storage_transposed,
            frequency_indices_transposed,
            K_plus_lms,
            K_cross_lms,
        )

        # Rotate by polarization angle psi for each source (same for all modes).
        tf_grid_plus, tf_grid_cross = jax.vmap(self.rotate_by_polarization_angle)(
            tf_grid_plus, tf_grid_cross, wf_params.psi
        )

        return tf_grid_plus, tf_grid_cross


@jax.jit
def v_tukey(f_dot_0, tau, f, f_0, alpha_term):
    """
    Fresnel argument for the time-frequency kernels, with each mode linearised about an expansion time t_e.

    Within a segment each mode is linearised as f(t) = f_0 + f_dot_0 * tau, with tau = t - t_e. Completing the
    square in the phase 2*pi*[(f_0 - f + alpha_term)*tau + f_dot_0*tau**2 / 2] gives the Fresnel argument

        v = sqrt(2*f_dot_0) * (tau + (f_0 - f + alpha_term) / f_dot_0),

    so the integral between tau_a and tau_b is C(v(tau_b)) - C(v(tau_a)) + i*(S(v(tau_b)) - S(v(tau_a))).

    Used in three ways:
        - Box-car kernel (`IMRPhenomTHM_TF.get_tf_fresnel_waveform_vanilla_TF`): t_e is the segment start t_0,
          alpha_term = 0, evaluated at tau = t_1 - t_0 (upper limit) and tau = 0 (lower limit).
        - Flat (unit-weight) pieces of the Tukey kernel (`IMRPhenomTHM_TF.get_tf_fresnel_tukey_midpoint`):
          t_e is the segment midpoint, alpha_term = 0, evaluated at tau0..tau3. Pairs with normal_phase_prefactor.
        - Roll-on/roll-off pieces of the Tukey kernel: in the tapers the window is
          1/2 * (1 - cos(2*pi*(tau - tau_edge) / (alpha*dT))). Writing the cosine as two exponentials shifts the
          frequency of the integrand by +/- 1/(alpha*dT), hence alpha_term = +/- alpha_offset. The +alpha_term
          results pair with phase_prefactor_plus, the -alpha_term results with phase_prefactor_minus.

    Parameters
    ----------
    f_dot_0 : Array
        Frequency derivative of each mode at t_e, shape (num_sources, n_modes). Units: Hz^2. Must be > 0.
        num_sources: binaries in the batch. n_modes: positive-m modes, (2,2) first, then self.higher_modes.
    tau : float
        Time relative to t_e at which the argument is evaluated. Units: seconds.
    f : Array
        Frequency bins at which the kernel is evaluated, shape (num_sources, n_modes, n_bins). Units: Hz.
        n_bins = 2*closest_f_bins: the bins around the grid frequency closest to f_0, for that source and mode.
    f_0 : Array
        Frequency of each mode at t_e, shape (num_sources, n_modes). Units: Hz.
    alpha_term : float
        Frequency shift from the Tukey taper, +/- 1/(tukey_alpha*dT) (alpha_offset in the caller),
        or 0 for the box-car kernel and the flat Tukey pieces. Units: Hz.

    Returns
    -------
    fresnel_argument : Array
        Dimensionless Fresnel argument, shape (num_sources, n_modes, n_bins).
    """
    fresnel_argument = jnp.sqrt(2 * f_dot_0[:, :, jnp.newaxis]) * (
        tau + (f_0[:, :, jnp.newaxis] - f + alpha_term) / f_dot_0[:, :, jnp.newaxis]
    )
    return fresnel_argument
