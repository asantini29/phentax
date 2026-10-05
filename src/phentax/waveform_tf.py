# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós


"""
Waveform
================================

IMRPhenomTHM_TF interface class for waveform generation.
Subclasses IMRPhenomTHM and adds fresnel representations, intended for pre-merger representation. 
"""

from statistics import mode
from typing import Optional

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from interpax import CubicSpline
from jaxtyping import Array

from phentax.core import (
    AmplitudeCoeffs,
    PhaseCoeffs,
    compute_amplitude_coeffs_22,
    compute_amplitude_coeffs_hm,
    compute_phase_coeffs_22,
    compute_phase_coeffs_hm,
    imr_amplitude,
    imr_phase,
    imr_omega,
    imr_omega_dot
)
from phentax.core.internals import WaveformParams, compute_waveform_params
from phentax.utils.coarse_graining import (
    generate_adaptive_grid,
    generate_uniform_grid,
    masked_evaluate,
)
from phentax.utils.config import setup_logging
from phentax.utils.utility import check_equal_bhs, mass_to_second, mode_to_lm,second_to_mass, mass_to_hz, df_dt_to_Hz_squared
from phentax.utils.ylm import (
    spin_weighted_spherical_harmonic,
    spin_weighted_spherical_harmonic_all_modes,
)
from phentax.waveform import IMRPhenomTHM

logger = setup_logging(__name__)

ALLOWED_POSITIVE_HMS = [21, 33, 44, 55]

from functools import partial

class IMRPhenomTHM_TF(IMRPhenomTHM):
    """
    IMRPhenomTHM_TF class for waveform generation.
    Subclasses IMRPhenomTHM and adds various fresnel representations. Current implementations: 
        - h_plus/h_cross in STFT domain directly in time-frequency.
            - Vanilla Fresnel representation, box-car window, waveform parameters defined at beginning of each segment. 
            - Tukey-Central Fresnel representation, Tukey window, waveform parameters defined at middle of each segment (More accurate). 
        - XYZ in STFT domain directly in time-frequency. Uses a time-frequency leading order (local) response function. 
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
            T = 1*30*24*3600, # Tobs unused in this case - 1 month in seconds as a placeholder here, note: unusued. 
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
        ) -> tuple[WaveformParams, Array, Array, AmplitudeCoeffs, PhaseCoeffs]:
            """
            Initial processing to compute waveform parameters.

            Note explicitly not computing over times here, as TF waveforms may use different time grids. We just want the:
                - Transformed waveform parameters
                - Amplitude and phase coeffs for 22 mode

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
            delta_t : float | Array, default 5.0
                Time step for waveform generation in seconds.
            t_min : float | Array, default jnp.nan
                Minimum time for waveform generation in seconds. If NaN, will be set by the minimum frequency.
            t_ref : float | Array, default jnp.nan
                Reference time for waveform generation in seconds. If NaN, will be set by the reference frequency.

            Returns
            -------
            wf_params : WaveformParams
                Waveform parameters of the binary, including derived parameters like total mass, symmetric mass ratio,
                and time arrays.
            times : Array
                Time array in units of mass for waveform generation.
            mask : Array
                Boolean mask indicating valid time points.
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
    ) -> tuple[Array]:
        """
        Utility function to compute phase coefficients for a given higher mode (beyond 22).
        """
        
        # m = mode % 10
        amplitude_coeffs = compute_amplitude_coeffs_hm(wf_params, phase_coeffs_22, mode)
        phase_coeffs = compute_phase_coeffs_hm(
                wf_params,
                phase_coeffs_22,
                OmegaCutPNAMP=amplitude_coeffs.omegaCutPNAMP,
                PhiCutPNAMP=amplitude_coeffs.phiCutPNAMP,
                mode=mode,
            )
            
        return(phase_coeffs,amplitude_coeffs)

    @jax.jit(static_argnums=[0,16])
    def get_tf_fresnel_waveform_vanilla_TF(self,
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
                                closest_f_bins: int = 10                            
                            ) -> tuple[Array, Array, Array, Array]: # Check dimensionality of this when done. 

        """
        Simplest implementation of time-frequency Fresnel waveform generation.

        Assumes box-car window, and that the waveform parameters are defined at the beginning of each segment.
        
        Note: This function uses an un-optimized version of the fresnel kernel, 
            but it is much more readable than the optimized version, so it is being kept in for now for understanding. 

        Time convention:
            - It is assumed that t_grid supplied by the user is long-enough to include the longest waveform in the batch.
            - Time grid is internally shifted so that the longest waveform in the batch starts at its own t_min (in physical time).
            - t(f_min) is computed for each source and used to project the waveform.

        Parameters
        ----------
        time_grid : Array
            Time grid over which to compute the waveform, shape (n_times,). Units: seconds.
            Note: These are the segment *edges*, relative to the start of the longest waveform in the batch.
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
            Minimum frequency, sets the start time of the waveform. Units: Hz.
            Ignored if t_min is supplied.
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
            Number of frequency bins to consider around the closest frequency for each source and mode, by default 10.

        Returns
        -------
        tf_grid_plus : Array
            Time-frequency grid of the plus polarization, shape (num_sources, n_times, n_freq). Units: strain.
            Row i is the segment [time_grid[i], time_grid[i+1]]; the last row is always zero.
        tf_grid_cross : Array
            Time-frequency grid of the cross polarization, shape (num_sources, n_times, n_freq). Units: strain.
            Row i is the segment [time_grid[i], time_grid[i+1]]; the last row is always zero.
        """

        num_sources = jnp.atleast_1d(m1).shape[0]

        # print("Number of sources: ", num_sources)
        # Ignore times for now
        wf_params, amplitude_coeffs_22, phase_coeffs_22 = (
                    self.initial_processing_TF(
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
                )
        # This is the minimum time for every waveform in the batch in mass units 
        min_time_M_units = wf_params.Mt_min 
        # print('min time M units:',min_time_M_units)

        #Convert every single min time to physical units (seconds) to see what is the longest waveform in absolute (real) time
        time_min_physical_all = mass_to_second(min_time_M_units, wf_params.total_mass)

        # print('All time min physical (s):',time_min_physical_all)

        # Pad to smallest min time (real units) to ensure all waveforms fit in the time grid
        min_time_s_units = jnp.min(time_min_physical_all)
        # print('Padded min time (real units):',min_time_s_units)

        # Check the length of the longest signal is not longer than the grid: 
        # assert jnp.abs(min_time_s_units) > jnp.abs(time_grid[-1]), "Time grid is not long enough to contain the longest waveform!"
        # print('Time min physical (s):',min_time_s_units)

        # print('Original time grid:',time_grid)

        # shift time grid to be negative, limited by the length of the longest waveform. (Physical units, seconds)
        time_grid += min_time_s_units
        # print('Shifted time grid (should be negative):',time_grid)
    

        # Work out what OUR time grid corresponds to in mass units for each binary 
        time_grids_mass_units = jax.vmap(second_to_mass, in_axes=(None, 0))(time_grid, wf_params.total_mass)

        new_mask = jnp.ones_like(time_grid, dtype=bool)


        # print('Time grid in mass units:',time_grids_mass_units,time_grids_mass_units.shape)

        # Compute amplitudes and phases for every binary over the time grid and mask    
        # Note: new_mask is the same for all sources, so we use in_axes=None for it (which are in seconds intiially and mapped uniquely to mass units for each binary)
        amplitudes, phases = jax.vmap(self._compute_all_modes, in_axes = (0, None, 0, 0, 0))(
            time_grids_mass_units,
            new_mask,
            wf_params,
            amplitude_coeffs_22,
            phase_coeffs_22,
        )  
        # Amplitudes and phases shape: [num_sources, num_modes, num_times]
        n_modes = amplitudes.shape[1]
        # print('Amplitudes shape:',amplitudes.shape)
        # print('Phases shape:',phases.shape)
       
       # Get higher order modes phase coeffs
        phase_hm_coeffs,_ = jax.vmap(
            lambda wp, pc22: jax.vmap(
                lambda mode: self._compute_phase_coeffs_hm(mode, wp, pc22)
            )(self.higher_modes)
        )(wf_params, phase_coeffs_22)

        # print('Phase HM coeffs:',phase_hm_coeffs)

        # Combine 22 and HM phase coeffs 
        overall_phase_coeffs = jax.tree_util.tree_map(
            lambda p22, phm: jnp.concatenate([p22[:, None], phm], axis=1),
            phase_coeffs_22,
            phase_hm_coeffs,
        )
        # print('Phase overall coeffs:',overall_phase_coeffs)

        # amplitudes shape: [num_sources, num_modes, num_times]
        # amp_factor shape: [num_sources] -> need to reshape for broadcasting
        amplitudes *= wf_params.amp_factor[:, None, None]

        # # For each source find the time at which to begin the waveform. (Is this handled inside the waveform parameter computation? i have no idea)
        # t_min_index = jnp.nonzero(jnp.abs(amplitudes[:,0,:]),axis=1)

        # Back to sensible and positive time grid
        time_grid -= min_time_s_units
        
        # Frequency grid spacing (assumed uniform)
        dF = frequency_grid[1] - frequency_grid[0]

        # Waveform storage container, will be combined into TF later
        waveform_storage = jnp.zeros((time_grid.size,num_sources,n_modes,2*closest_f_bins), dtype=jnp.complex128)

        # Storage for frequency indices at each time step (# 2*closest_f_bins because we are storing bins on either side of the closest frequency for each harmonic for each time)
        frequency_indices_storage = jnp.zeros((time_grid.size,num_sources,n_modes,2*closest_f_bins), dtype=jnp.int32)

        #tf grid 
        # tf_grid = jnp.zeros((num_sources,time_grid.size, frequency_grid.size), dtype=jnp.complex128)

        # In principle can also be vmapped over time steps here, however I think it would be completely unreadable and a nightmare to debug/maintain.. 
        for time_index in range(time_grid.size - 1):  # -1 to avoid index out of bounds with t_1
                
                t_0 = time_grid[time_index]
                t_1 = time_grid[time_index+1]

                # Convert t_0 to mass units for each binary for each binary
                t_0_mass = jax.vmap(second_to_mass, in_axes=(None, 0))(t_0 + min_time_s_units, wf_params.total_mass) # nSources, 

                Amps = amplitudes[:,:, time_index] # Source, #Mode , # Time 
                Phases = phases[:,:, time_index]   # Source, #Mode , # Time 

                # So we are vmapping across both sources and modes now. For each time 

                # overall_phase_coeffs contains phase coeffs for all modes at once for all binaries (Only positive modes remember)
                # Using this compute f_0 and f_dot for each source and each mode at this time
                # 
                # Shapes:
                #   t_0_mass: (num_sources,)
                #   wf_params.eta: (num_sources,)
                #   overall_phase_coeffs: dictionary like structure but each field has shape (num_sources, num_modes, ...)
                #
                # Nested vmap explanation:
                # -------------------------
                # We need to compute imr_omega for every (source, mode) combination.
                # 
                # OUTER vmap (over sources):
                #   - Iterates over axis 0 of t_0_mass, eta, and overall_phase_coeffs
                #   - For source i: t_0_mass[i] is scalar, eta[i] is scalar, 
                #     overall_phase_coeffs[i] is a pytree with shape (num_modes, ...) per field
                #
                # INNER vmap (over modes for a single source):
                #   - For a single source, iterates over axis 0 of phase_coeffs (the modes axis)
                #   - Computes imr_omega(t, eta, p) for each mode's phase coeffs p
                #   - t and eta are fixed scalars from the outer vmap
                #
                # Result: f_0 has shape (num_sources, num_modes) (same with f_dot)
                
                # pc is the phase coefficents for a single source, shape (num_modes, ...), p is the phase coeffs for a single mode, shape (...)
                f_0 = jax.vmap(
                    lambda t, eta, pc: jax.vmap(lambda p: imr_omega(t, eta, p))(pc)
                )(t_0_mass, wf_params.eta, overall_phase_coeffs) / (2 * jnp.pi)  # IN MASS UNITS
                # print('F0 (mass units) shape:', f_0.shape)

                f_dot = jax.vmap(
                    lambda t, eta, pc: jax.vmap(lambda p: imr_omega_dot(t, eta, p))(pc)
                )(t_0_mass, wf_params.eta, overall_phase_coeffs) / (2 * jnp.pi)  # IN MASS UNITS
                # print('F dot (mass units) shape:', f_dot.shape)

                # Convert f_0 and f_dot to Hz and Hz^2 respectively
                f_0 = jax.vmap(mass_to_hz, in_axes=(0, 0))(f_0, wf_params.total_mass)
                f_dot = jax.vmap(df_dt_to_Hz_squared, in_axes=(0, 0))(f_dot, wf_params.total_mass)
                # print('F0 (Hz) shape:', f_0.shape) 
                # print('F dot (Hz^2) shape:', f_dot.shape)

                # NOTE: the potential for frequencies around f0 going out of bounds is dealt with later. 

                # Compute closest frequency indexes in the grid for each source and mode
                # Uniform grid → direct arithmetic instead of materializing an (nSrc, nMode, nFreq) tensor
                closest_frequency_indexes = jnp.round((f_0 - frequency_grid[0]) / dF).astype(jnp.int32)
                closest_frequency_indexes = jnp.clip(closest_frequency_indexes, 0, frequency_grid.size - 1)
                closest_frequencies = frequency_grid[closest_frequency_indexes]
                # Closest frequency shape is (num_sources, num_modes) and closest frequency indexes is also (num_sources, num_modes). 
                # Same shape as f_0 and f_dot

                # print('Closest frequency indexes shape:', closest_frequency_indexes.shape,closest_frequency_indexes.flatten().shape)

                # for every frequency in closest_frequencies, generate a frequency array around it +/- closest_f_bins by going in steps of dF
                frequencies = jax.vmap(
                    lambda center_freqs: jnp.arange(
                        0,
                        2*closest_f_bins,
                    ) * dF + center_freqs - closest_f_bins*dF, # Starting from center_freqs - closest_f_bins*dF to center_freqs + closest_f_bins*dF
                )(closest_frequencies.flatten()).reshape(num_sources,n_modes,2*closest_f_bins)  #
                # shape of frequencies is (num_sources, num_modes, 2*closest_f_bins) 

                # Holds indices for each frequency in frequencies, which will be used to place the waveform values in the correct position in the final tf grid.
                frequency_indices = jax.vmap(
                    lambda indices: jnp.arange(
                        0,
                        2*closest_f_bins,
                    )+ indices - closest_f_bins, # Starting from indices - closest_f_bins to indices + closest_f_bins
                )(closest_frequency_indexes.flatten()).reshape(num_sources,n_modes,2*closest_f_bins)

                # The newaxis here is accounting for the frequency dimension, 
                #     remember we generated a single A, f, f_dot for each source and mode,
                #      but now we have an array of frequencies for each source and mode, so we need to add a new axis to Amps,
                #      Phases, f_0 and f_dot to allow for broadcasting when computing the waveform values for each frequency.
                h_prefactor = Amps[:,:,jnp.newaxis]*jnp.exp(1j*Phases[:,:,jnp.newaxis])/jnp.sqrt(2*f_dot[:,:,jnp.newaxis]) * jnp.exp(-1j*jnp.pi*((f_0[:,:,jnp.newaxis] - frequencies)**2)/f_dot[:,:,jnp.newaxis])

                # Fresnel integral stuff (all )
                v_nm_end = v(f_dot,t_0,t_1,frequencies,f_0)
                v_nm_begin = v(f_dot,t_0,t_0,frequencies,f_0)
                S_vn_end, C_vn_end = jax.scipy.special.fresnel(v_nm_end)
                S_vn_begin, C_vn_begin = jax.scipy.special.fresnel(v_nm_begin)

                I = C_vn_end - C_vn_begin + 1j*(S_vn_end - S_vn_begin)
                waveform = h_prefactor * I
                
                # Store each waveform in its own 'tf grid' for now. 
                # NOTE: Final implementation will use something much closer to the direct likelihood/inner product computation. Maybe....
                # NOTE: In its current form this is *NOT* the actual TF grid, the frequency axis has not yet been placed in the correct position,
                #         we are just storing the waveforms for each time step and each mode in a temporary array here, 
                waveform_storage = waveform_storage.at[time_index,:,:,:].set(waveform)

                # Store the corresponding frequency indices for each waveform value, which will be used to place the values in the correct position in the final tf grid.
                frequency_indices_storage = frequency_indices_storage.at[time_index,:,:,:].set(frequency_indices)

        # Note in theory one can do the direct likelihood/inner product computation directly from waveform storage I think. 


        # TEMPORARY/DEV
        # Transpose to (num_sources, num_times, num_modes, 2*closest_f_bins) for vmapping (basically changing around order of axes)
        waveform_storage_transposed = waveform_storage.transpose(1, 0, 2, 3)
        frequency_indices_transposed = frequency_indices_storage.transpose(1, 0, 2, 3)

        # Number of frequency bins in the full grid
        n_freq = frequency_grid.size # TODO: can be precomputed before this function. 

        # Generate spherical harmonics for all modes and sources at once.
        y_lms = spin_weighted_spherical_harmonic_all_modes(
                    jnp.atleast_1d(inclination)[:, None],
                    jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
                    self.ells,
                    self.mms,
                ) #Shape (num_sources, num_modes,1)
        
        y_lmms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.negative_ls,
            self.negative_mms, 
            ) #Shape (num_sources, num_modes,1)
        
        K_plus_lms = 1/2*(y_lms[:,:,0] + (-1)**self.negative_ls*y_lmms[:,:,0].conj()).conj() # Overall Conj to flip the fourier convention (Compared to that of Marsat appendix. )
        K_cross_lms = (1j/2*(y_lms[:,:,0] - (-1)**self.negative_ls*y_lmms[:,:,0].conj())).conj() # Overall Conj (including the 1j prefactor!) to flip the fourier convention (Compared to that of Marsat appendix.)
        # Shapes of K are (num_sources, num_modes) where num_modes includes only the positive modes (we are doing the reflection trick for negative modes for non-precessing binaries)
        

        # total_fresnel_waveforms_h_plus = jnp.einsum('ni,ijk->jk',K_plus_lms,tf_grid)
        # total_fresnel_waveforms_h_cross = jnp.einsum('ni,ijk->jk',K_cross_lms,tf_grid)

        # h_plus_rotated, h_cross_rotated = imr.rotate_by_polarization_angle(total_fresnel_waveforms_h_plus, total_fresnel_waveforms_h_cross, psi)


        def place_waveform_at_time(waveform_modes, indices_modes,K_plus, K_cross):
            """
            Place waveforms from all modes into the frequency grid for a single time step.
            
            waveform_modes: (n_modes, 2*closest_f_bins) - complex waveform values
            indices_modes: (n_modes, 2*closest_f_bins) - frequency bin indices
            K_plus: (n_modes,) - complex coefficients for plus polarization
            K_cross: (n_modes,) - complex coefficients for cross polarization
            
            Returns: (n_freq,) - summed waveform across modes at correct frequency positions
            """
            # Multiply K coefficients *before* flattening so each mode's scalar
            # broadcasts across its 2*closest_f_bins frequency entries.
            # K_plus/K_cross: (n_modes,), waveform_modes: (n_modes, 2*closest_f_bins) 
            h_plus_modes = K_plus[:, None] * waveform_modes   # (n_modes, 2*closest_f_bins)
            h_cross_modes = K_cross[:, None] * waveform_modes

            # Now flatten across modes and frequency bins
            flat_h_plus = h_plus_modes.flatten()
            flat_h_cross = h_cross_modes.flatten()
            flat_indices = indices_modes.flatten()

            # Mask out-of-bounds indices: zero their contributions
            valid_mask = (flat_indices >= 0) & (flat_indices < n_freq)
            flat_h_plus = jnp.where(valid_mask, flat_h_plus, 0.0)
            flat_h_cross = jnp.where(valid_mask, flat_h_cross, 0.0)
            
            # Clip indices to valid range so .at[].add() doesn't error with out-of-bounds indices.
            # The zeroed values mean nothing is actually added for these entries.
            flat_indices = jnp.clip(flat_indices, 0, n_freq - 1)

            result_plus = jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_plus)
            result_cross = jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_cross)

            return result_plus, result_cross

        def process_source(waveforms_per_source, indices_per_source,K_plus, K_cross):
            """
            Process all time steps for a single source.
            
            waveforms_per_source: (n_times, n_modes, 2*closest_f_bins), this is a version of the TF grid for one source (with the wrong frequency axis)
            indices_per_source: (n_times, n_modes, 2*closest_f_bins)
            K_plus: (n_modes,) - complex coefficients for plus polarization
            K_cross: (n_modes,) - complex coefficients for cross polarization
            
            Returns: (n_times, n_freq) - TF map for this source
            """
            # This is vmapping across timesteps
            return jax.vmap(place_waveform_at_time, in_axes = (0, 0, None, None) )(waveforms_per_source, indices_per_source,K_plus, K_cross)

        # NOTE: this is a 2 nested vmap, its just done like this right now for readability and debugging. 
        # Vmap over sources to get final tf_grid with shape (num_sources, num_times, num_freq)
        tf_grid_plus, tf_grid_cross = jax.vmap(process_source)(
            waveform_storage_transposed,
            frequency_indices_transposed,
            K_plus_lms,
            K_cross_lms
        ) # vmap over *SOURCES*

        # Rotate by polarization angle psi for each source (applied to all modes in the same way)
        tf_grid_plus, tf_grid_cross = jax.vmap(self.rotate_by_polarization_angle)(
            tf_grid_plus, tf_grid_cross, wf_params.psi
        )
        
        
        return(tf_grid_plus, tf_grid_cross) #Each (time_grid, frequency_grid, tf_grid) # Returning the full TF grid for all sources.

    @jax.jit(static_argnums=[0,15,16])
    def get_tf_fresnel_tukey_midpoint(self,
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
                            ) -> tuple[Array, Array, Array, Array]: # Check dimensionality of this when done. 

        """
        Implementation of time-frequency Fresnel waveform generation. Accounting for the frequency domain correction from the tukey window.
        Returns the source-frame polarizations, no detector applied.

        Waveforms are defined with respect to the centre of the time-segments, and the tukey correction is analytically taken into account.
        
        Time convention: 
            - It is assumed that t_grid supplied by the user is long-enough to include the longest waveform in the batch. 
            - Time grid is internally shifted so that the longest waveform in the batch starts at its own t_min (in physical time). 
            - t(f_min) is computed for each source and used to project the waveform.

        Parameters
        ----------
        time_grid : Array
            Time grid over which to compute the waveform, shape (n_times,). Units: seconds.
            Note: These are the segment *edges*, relative to the start of the longest waveform in the batch,
            so the output carries n_times - 1 tranches. Must be uniformly spaced.
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
            Minimum frequency, sets the start time of the waveform. Units: Hz.
            Ignored if t_min is supplied.
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
            Number of frequency bins to consider around the closest frequency for each source and mode, by default 10.
        tukey_alpha : float, optional
            Alpha parameter for the Tukey window, by default 0.5.

        Returns
        -------
        tf_grid_plus : Array
            Time-frequency grid of the plus polarization, shape (num_sources, num_tranches, n_freq). Units: strain.
        tf_grid_cross : Array
            Time-frequency grid of the cross polarization, shape (num_sources, num_tranches, n_freq). Units: strain.
        """

        num_sources = jnp.atleast_1d(m1).shape[0]

        # print("Number of sources: ", num_sources)
        # Ignore times for now 
        wf_params, amplitude_coeffs_22, phase_coeffs_22 = (
                    self.initial_processing_TF(
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
                        1, #delta_t, in time-domain this is the sampling cadence, in the TF domain this simply acts as a shift to the start time of the waveform, it acts as "setting the waveform start time to a multiple of delta_t" so delta_t=1 is safe. 
                        t_min,
                        t_ref,
                    )
                )
            
        # This is the minimum time for every waveform in the batch in mass units 
        min_time_M_units = wf_params.Mt_min 
        # print('min time M units:',min_time_M_units)

        #Convert every single min time to physical units (seconds) to see what is the longest waveform in absolute (real) time
        time_min_physical_all = mass_to_second(min_time_M_units, wf_params.total_mass)

        # print('All time min physical (s):',time_min_physical_all)

        # Pad to smallest min time (real units) to ensure all waveforms fit in the time grid
        min_time_s_units = jnp.min(time_min_physical_all)

        # shift time grid to be negative, limited by the length of the longest waveform. (Physical units, seconds)
        time_grid += min_time_s_units
    

        t_grid_midpoints = 0.5 * (time_grid[:-1] + time_grid[1:])  # Midpoints between time grid edges, shape (n_times - 1,)
        
        t_grid_midpoints_mass_units = jax.vmap(second_to_mass, in_axes=(None, 0))(t_grid_midpoints, wf_params.total_mass)
        midpoint_mask = jnp.ones_like(t_grid_midpoints, dtype=bool)  
        
        # Compute amplitudes and phases at midpoints of each tranche for each binary
        amplitudes, phases = jax.vmap(self._compute_all_modes, in_axes = (0, None, 0, 0, 0))(
            t_grid_midpoints_mass_units,
            midpoint_mask,
            wf_params,
            amplitude_coeffs_22,
            phase_coeffs_22,
        )

        n_modes = amplitudes.shape[1]

       
       # Get higher order modes phase coeffs
        phase_hm_coeffs,_ = jax.vmap(
            lambda wp, pc22: jax.vmap(
                lambda mode: self._compute_phase_coeffs_hm(mode, wp, pc22)
            )(self.higher_modes)
        )(wf_params, phase_coeffs_22)

        # print('Phase HM coeffs:',phase_hm_coeffs)

        # Combine 22 and HM phase coeffs 
        overall_phase_coeffs = jax.tree_util.tree_map(
            lambda p22, phm: jnp.concatenate([p22[:, None], phm], axis=1),
            phase_coeffs_22,
            phase_hm_coeffs,
        )
        # print('Phase overall coeffs:',overall_phase_coeffs)

        # amplitudes shape: [num_sources, num_modes, num_times]
        # amp_factor shape: [num_sources] -> need to reshape for broadcasting
        amplitudes *= wf_params.amp_factor[:, None, None]

        # # For each source find the time at which to begin the waveform. (Is this handled inside the waveform parameter computation? i have no idea)
        # t_min_index = jnp.nonzero(jnp.abs(amplitudes[:,0,:]),axis=1)

        # Back to sensible and positive time grid
        time_grid -= min_time_s_units
        t_grid_midpoints = 0.5 * (time_grid[:-1] + time_grid[1:])
        
        # Frequency grid spacing (assumed uniform)
        dF = frequency_grid[1] - frequency_grid[0]

        # Build per-step inputs and scan over time tranches to keep the loop JAX-native.
        t0_all = time_grid[:-1] # Beginning times for all the tranches (Physical units, seconds)
        t1_all = time_grid[1:] # Ending times for all the tranches (Physical units, seconds)
        tmid_all = t_grid_midpoints # Midpoint times for all the tranches (Physical units, seconds)


        # Convert midpoints to mass units for computations of f,fdot 
        tmid_mass_all = jax.vmap(
            lambda t_mid: jax.vmap(second_to_mass, in_axes=(None, 0))(
                t_mid + min_time_s_units, wf_params.total_mass # Need the shift here in t_mid as remember t_mid in seconds starts at 0 
            )
        )(tmid_all)

        # Transpose amplitudes and phases to shape (num_times, num_sources, num_modes) for scanning over time tranches.
        amps_all = amplitudes.transpose(2, 0, 1)    
        phases_all = phases.transpose(2, 0, 1)

        # Precompute frequency-evolution quantities shared by each time tranche.
        f_0_all = (
            jax.vmap(
                lambda t_mid_mass: jax.vmap(
                    lambda t, eta, pc: jax.vmap(lambda p: imr_omega(t, eta, p))(pc)
                )(t_mid_mass, wf_params.eta, overall_phase_coeffs)
            )(tmid_mass_all)
            / (2 * jnp.pi)
        )# IN MASS UNITS

        # Precompute f_dot at midpoints for all sources and modes.
        f_dot_all = (
            jax.vmap(
                lambda t_mid_mass: jax.vmap(
                    lambda t, eta, pc: jax.vmap(lambda p: imr_omega_dot(t, eta, p))(pc)
                )(t_mid_mass, wf_params.eta, overall_phase_coeffs)
            )(tmid_mass_all)
            / (2 * jnp.pi)
        )# IN MASS UNITS

        f_0_all = jax.vmap(
            lambda f0_t: jax.vmap(mass_to_hz, in_axes=(0, 0))(f0_t, wf_params.total_mass)
        )(f_0_all)# Convert f_0 to Hz

        f_dot_all = jax.vmap(
            lambda fd_t: jax.vmap(df_dt_to_Hz_squared, in_axes=(0, 0))(
                fd_t, wf_params.total_mass
            )
        )(f_dot_all)# Convert f_dot to Hz^2
        
        # Work out for each source, for each mode, for each time segment, what are the closest frequencies on the grid.
        closest_frequency_indexes_all = jnp.round(
            (f_0_all - frequency_grid[0]) / dF
        ).astype(jnp.int32)

        # Clip to valid range to avoid out-of-bounds indexing later (we will mask out contributions from out-of-bounds frequencies anyway, so the clipping won't affect the final result)
        closest_frequency_indexes_all = jnp.clip(
            closest_frequency_indexes_all, 0, frequency_grid.size - 1
        )

        # Get the actual closest frequencies from the grid for each source, mode, and time segment.
        closest_frequencies_all = frequency_grid[closest_frequency_indexes_all]

        # For every frequency in closest_frequencies_all, we will generate a frequency array around it +/- closest_f_bins by going in steps of dF inside the scan loop.
        freq_offsets = jnp.arange(0, 2 * closest_f_bins, dtype=frequency_grid.dtype)
        idx_offsets = jnp.arange(0, 2 * closest_f_bins, dtype=jnp.int32)
        
        # Compute the frequency arrays and corresponding indices for all sources, modes, and time segments at once.
        frequencies_all = (
            closest_frequencies_all[:, :, :, None]
            + (freq_offsets[None, None, None, :] - closest_f_bins) * dF
        )
        frequency_indices_all = (
            closest_frequency_indexes_all[:, :, :, None]
            + (idx_offsets[None, None, None, :] - closest_f_bins)
        )

        #TODO: This can be removed
        dT = t1_all[0] - t0_all[0] # Assuming uniform time grid, this is the width of each time segment.

        alpha_offset = 1/(tukey_alpha*dT)

        def scan_step(_, scan_inputs):
            t_0, t_midpoint, Amps, Phases, f_0, f_dot, frequencies, frequency_indices = scan_inputs

        
            # Rejigged efficient vectorized version of the fresnel integral with tukey window:
            # Prefactor for the whole integral, applies to all the terms. 
            h_prefactor = (
                Amps[:, :, jnp.newaxis]
                * jnp.exp(1j * Phases[:, :, jnp.newaxis])
                / jnp.sqrt(2 * f_dot[:, :, jnp.newaxis])
                * jnp.exp(-2 * 1j * jnp.pi * frequencies * (t_midpoint - t_0))
            )

            # Precomputing phase prefactors used a few times. 
            normal_phase_prefactor = jnp.exp(-1j*jnp.pi*((f_0[:, :, jnp.newaxis] - frequencies)**2)/f_dot[:, :, jnp.newaxis]) # Phase factor for the flat part

            # Phase factors for the positive and negative alpha offsets to the tukey-fresnel integral terms 
            phase_prefactor_minus = jnp.exp(-1j*jnp.pi*(f_0[:, :, jnp.newaxis] - frequencies - alpha_offset)**2/f_dot[:, :, jnp.newaxis])
            phase_prefactor_plus = jnp.exp(-1j*jnp.pi*(f_0[:, :, jnp.newaxis] - frequencies + alpha_offset)**2/f_dot[:, :, jnp.newaxis])


            # Times at which to evaluate arguments of the fresnel terms within the SFT segment.
            # These are just the key times within an SFT segment with a tukey window applied. 
            tau0 = -0.5*dT
            tau1 = 0.5*(-dT + tukey_alpha*dT)
            tau2 = 0.5*(dT - tukey_alpha*dT)
            tau3 = 0.5*dT

            # All the fresnel arguments. 
            args = jnp.stack(
                        [v_new(f_dot, t, frequencies, f_0)                  for t in (tau0, tau1, tau2, tau3)] # Arguments for flat parts
                        + [v_tukey(f_dot, t, frequencies, f_0,  alpha_offset) for t in (tau0, tau1, tau2, tau3)] # Arguments for roll-on/roll-off parts (positive offset integrals)
                        + [v_tukey(f_dot, t, frequencies, f_0, -alpha_offset) for t in (tau0, tau1, tau2, tau3)] # Arguments for roll-on/roll-off parts (negative offset integrals) 
                )                                                       # (12, num_sources, n_modes, n_bins)

            S, C = jax.scipy.special.fresnel(args)

            F = C + 1j * S

            # Unpack fresnel terms 
            flat_t0,  flat_t1,  flat_t2,  flat_t3  = F[0],  F[1],  F[2],  F[3] 
            plus_t0,  plus_t1,  plus_t2,  plus_t3  = F[4],  F[5],  F[6],  F[7]
            minus_t0, minus_t1, minus_t2, minus_t3 = F[8],  F[9],  F[10], F[11]


            W = jnp.exp(2j * jnp.pi / tukey_alpha * (0.5 - tukey_alpha / 2))
            W_conj = jnp.conj(W)

            # The subtraction is simply the fresnel term evaluated at the upper limit minus the fresnel term evaluated at the lower limit, for each of the three pieces of the integral.
            
            # Roll on terms
            h_flat_roll_on      = 0.5 * normal_phase_prefactor * (flat_t1  - flat_t0)
            h_roll_on_positive  = W      / 4 * phase_prefactor_plus  * (plus_t1  - plus_t0)
            h_roll_on_negative  = W_conj / 4 * phase_prefactor_minus * (minus_t1 - minus_t0)

            # Flat middle. 
            h_mid               =       normal_phase_prefactor * (flat_t2  - flat_t1)

            # Roll off terms
            h_roll_off_flat     = 0.5 * normal_phase_prefactor * (flat_t3  - flat_t2)
            h_roll_off_positive = W_conj / 4 * phase_prefactor_plus  * (plus_t3  - plus_t2)
            h_roll_off_negative = W      / 4 * phase_prefactor_minus * (minus_t3 - minus_t2)

            # Sum the three pieces to get the full integral with the tukey window taken into account.
            h_overall =(h_prefactor * (h_flat_roll_on + 
                                        h_roll_on_positive + 
                                        h_roll_on_negative + 
                                        h_mid + 
                                        h_roll_off_flat + 
                                        h_roll_off_positive + 
                                        h_roll_off_negative))
            return None, (h_overall, frequency_indices)

        _, (waveform_steps, frequency_indices_steps) = jax.lax.scan(
            scan_step,
            None,
            (
                t0_all,
                tmid_all,
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

        # Note in theory one can do the direct likelihood/inner product computation directly from waveform storage I think. 
        # TEMPORARY/DEV
        # Transpose to (num_sources, num_times, num_modes, 2*closest_f_bins) for vmapping (basically changing around order of axes)
        waveform_storage_transposed = waveform_storage.transpose(1, 0, 2, 3)
        frequency_indices_transposed = frequency_indices_storage.transpose(1, 0, 2, 3)

        # Number of frequency bins in the full grid
        n_freq = frequency_grid.size # TODO: can be precomputed before this function. 

        # Generate spherical harmonics for all modes and sources at once.
        y_lms = spin_weighted_spherical_harmonic_all_modes(
                    jnp.atleast_1d(inclination)[:, None],
                    jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
                    self.ells,
                    self.mms,
                ) #Shape (num_sources, num_modes,1)
        
        y_lmms = spin_weighted_spherical_harmonic_all_modes(
            jnp.atleast_1d(inclination)[:, None],
            jnp.pi / 2.0 - jnp.atleast_1d(phi_ref)[:, None],
            self.negative_ls,
            self.negative_mms, 
            ) #Shape (num_sources, num_modes,1)
        
        K_plus_lms = 1/2*(y_lms[:,:,0] + (-1)**self.negative_ls*y_lmms[:,:,0].conj()).conj() # Overall Conj to flip the fourier convention (Compared to that of Marsat appendix. )
        K_cross_lms = (1j/2*(y_lms[:,:,0] - (-1)**self.negative_ls*y_lmms[:,:,0].conj())).conj() # Overall Conj (including the 1j prefactor!) to flip the fourier convention (Compared to that of Marsat appendix.)
        # Shapes of K are (num_sources, num_modes) where num_modes includes only the positive modes (we are doing the reflection trick for negative modes for non-precessing binaries)
    
        def place_waveform_at_time(waveform_modes, indices_modes,K_plus, K_cross):
            """
            Place waveforms from all modes into the frequency grid for a single time step.
            
            waveform_modes: (n_modes, 2*closest_f_bins) - complex waveform values
            indices_modes: (n_modes, 2*closest_f_bins) - frequency bin indices
            K_plus: (n_modes,) - complex coefficients for plus polarization
            K_cross: (n_modes,) - complex coefficients for cross polarization
            
            Returns: (n_freq,) - summed waveform across modes at correct frequency positions
            """
            # Multiply K coefficients *before* flattening so each mode's scalar
            # broadcasts across its 2*closest_f_bins frequency entries.
            # K_plus/K_cross: (n_modes,), waveform_modes: (n_modes, 2*closest_f_bins) 
            h_plus_modes = K_plus[:, None] * waveform_modes   # (n_modes, 2*closest_f_bins)
            h_cross_modes = K_cross[:, None] * waveform_modes

            # Now flatten across modes and frequency bins
            flat_h_plus = h_plus_modes.flatten()
            flat_h_cross = h_cross_modes.flatten()
            flat_indices = indices_modes.flatten()

            # Mask out-of-bounds indices: zero their contributions
            valid_mask = (flat_indices >= 0) & (flat_indices < n_freq)
            flat_h_plus = jnp.where(valid_mask, flat_h_plus, 0.0)
            flat_h_cross = jnp.where(valid_mask, flat_h_cross, 0.0)
            
            # Clip indices to valid range so .at[].add() doesn't error with out-of-bounds indices.
            # The zeroed values mean nothing is actually added for these entries.
            flat_indices = jnp.clip(flat_indices, 0, n_freq - 1)

            result_plus = jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_plus)
            result_cross = jnp.zeros(n_freq, dtype=jnp.complex128).at[flat_indices].add(flat_h_cross)

            return result_plus, result_cross

        def process_source(waveforms_per_source, indices_per_source,K_plus, K_cross):
            """
            Process all time steps for a single source.
            
            waveforms_per_source: (n_times, n_modes, 2*closest_f_bins), this is a version of the TF grid for one source (with the wrong frequency axis)
            indices_per_source: (n_times, n_modes, 2*closest_f_bins)
            K_plus: (n_modes,) - complex coefficients for plus polarization
            K_cross: (n_modes,) - complex coefficients for cross polarization
            
            Returns: (n_times, n_freq) - TF map for this source
            """
            # This is vmapping across timesteps
            return jax.vmap(place_waveform_at_time, in_axes = (0, 0, None, None) )(waveforms_per_source, indices_per_source,K_plus, K_cross)

        # NOTE: this is a 2 nested vmap, its just done like this right now for readability and debugging. 
        # Vmap over sources to get final tf_grid with shape (num_sources, num_times, num_freq)
        tf_grid_plus, tf_grid_cross = jax.vmap(process_source)(
            waveform_storage_transposed,
            frequency_indices_transposed,
            K_plus_lms,
            K_cross_lms
        ) # vmap over *SOURCES*

        # Rotate by polarization angle psi for each source (applied to all modes in the same way)
        tf_grid_plus, tf_grid_cross = jax.vmap(self.rotate_by_polarization_angle)(
            tf_grid_plus, tf_grid_cross, wf_params.psi
        )
        
        
        return(tf_grid_plus, tf_grid_cross) #Each (time_grid, frequency_grid, tf_grid) # Returning the full TF grid for all sources.

@jax.jit
def v(f_dot_0,t_0,t_1,f,f_0):
    """
    Fresnel argument for the box-car (vanilla) kernel, with the mode expanded about the segment start t_0.

    Within a segment each mode is linearised as f(t) = f_0 + f_dot_0 * (t - t_0). Completing the square in the
    phase 2*pi*[(f_0 - f)*(t - t_0) + f_dot_0*(t - t_0)**2 / 2] gives the Fresnel argument

        v = sqrt(2*f_dot_0) * ((t_1 - t_0) + (f_0 - f) / f_dot_0),

    so the segment integral is C(v(t_1)) - C(v(t_0)) + i*(S(v(t_1)) - S(v(t_0))).
    Used by `IMRPhenomTHM_TF.get_tf_fresnel_waveform_vanilla_TF`, which calls it with t_1 = t_0 for the lower limit.

    Parameters
    ----------
    f_dot_0 : Array
        Frequency derivative of each mode at t_0, shape (num_sources, n_modes). Units: Hz^2. Must be > 0.
        num_sources: binaries in the batch. n_modes: positive-m modes, (2,2) first, then self.higher_modes.
    t_0 : float
        Expansion time, the start of the segment. Units: seconds.
    t_1 : float
        Time at which the argument is evaluated, the end of the segment (or t_0 for the lower limit). Units: seconds.
    f : Array
        Frequency bins at which the kernel is evaluated, shape (num_sources, n_modes, n_bins). Units: Hz.
        n_bins = 2*closest_f_bins: the bins around the grid frequency closest to f_0, for that source and mode.
    f_0 : Array
        Frequency of each mode at t_0, shape (num_sources, n_modes). Units: Hz.

    Returns
    -------
    fresnel_argument : Array
        Dimensionless Fresnel argument, shape (num_sources, n_modes, n_bins).
    """
    fresnel_argument = jnp.sqrt(2*f_dot_0[:,:,jnp.newaxis])*((t_1-t_0) + (f_0[:,:,jnp.newaxis]-f)/f_dot_0[:,:,jnp.newaxis])
    return fresnel_argument

@jax.jit
def v_tukey(f_dot_0,tau,f,f_0,alpha_term):
    """
    Fresnel argument for the roll-on/roll-off pieces of the Tukey-window kernel, expanded about the segment midpoint.

    In the tapers the Tukey window is 1/2 * (1 - cos(2*pi*(tau - tau_edge) / (alpha*dT))). Writing the cosine as two
    exponentials shifts the frequency of the integrand by +/- 1/(alpha*dT), so the argument is that of `v_new`
    with f_0 - f replaced by f_0 - f + alpha_term:

        v = sqrt(2*f_dot_0) * (tau + (f_0 - f + alpha_term) / f_dot_0).


    Used by `IMRPhenomTHM_TF.get_tf_fresnel_tukey_midpoint`, evaluated at the four segment edges
    tau0..tau3 for both signs of alpha_term. The +alpha_term results pair with phase_prefactor_plus,
    the -alpha_term results with phase_prefactor_minus.

    Parameters
    ----------
    f_dot_0 : Array
        Frequency derivative of each mode at the segment midpoint, shape (num_sources, n_modes). Units: Hz^2. Must be > 0.
        num_sources: binaries in the batch. n_modes: positive-m modes, (2,2) first, then self.higher_modes.
    tau : float
        Time relative to the segment midpoint at which the argument is evaluated, one of
        -dT/2, -dT/2*(1 - alpha), dT/2*(1 - alpha), dT/2. Units: seconds.
    f : Array
        Frequency bins at which the kernel is evaluated, shape (num_sources, n_modes, n_bins). Units: Hz.
        n_bins = 2*closest_f_bins: the bins around the grid frequency closest to f_0, for that source and mode.
    f_0 : Array
        Frequency of each mode at the segment midpoint, shape (num_sources, n_modes). Units: Hz.
    alpha_term : float
        Frequency shift from the Tukey taper, +/- 1/(tukey_alpha*dT) (alpha_offset in the callers). Units: Hz.

    Returns
    -------
    fresnel_argument : Array
        Dimensionless Fresnel argument, shape (num_sources, n_modes, n_bins).
    """
    fresnel_argument = jnp.sqrt(2*f_dot_0[:,:,jnp.newaxis])*(tau + (f_0[:,:,jnp.newaxis]-f+alpha_term)/f_dot_0[:,:,jnp.newaxis])
    return fresnel_argument

@jax.jit
def v_new(f_dot_0,tau,f,f_0):
    """
    Fresnel argument for the flat (unit-weight) pieces of the Tukey-window kernel, expanded about the segment midpoint.

    Within a segment each mode is linearised as f(tau) = f_0 + f_dot_0 * tau, with tau measured from the midpoint:

        v = sqrt(2*f_dot_0) * (tau + (f_0 - f) / f_dot_0).

    This is `v_tukey` with alpha_term = 0, and `v` with the expansion point moved from the segment
    start to the midpoint. Used by `IMRPhenomTHM_TF.get_tf_fresnel_tukey_midpoint` evaluated at the four segment edges tau0..tau3;
    the results pair with normal_phase_prefactor.

    Parameters
    ----------
    f_dot_0 : Array
        Frequency derivative of each mode at the segment midpoint, shape (num_sources, n_modes). Units: Hz^2. Must be > 0.
        num_sources: binaries in the batch. n_modes: positive-m modes, (2,2) first, then self.higher_modes.
    tau : float
        Time relative to the segment midpoint at which the argument is evaluated, one of
        -dT/2, -dT/2*(1 - alpha), dT/2*(1 - alpha), dT/2. Units: seconds.
    f : Array
        Frequency bins at which the kernel is evaluated, shape (num_sources, n_modes, n_bins). Units: Hz.
        n_bins = 2*closest_f_bins: the bins around the grid frequency closest to f_0, for that source and mode.
    f_0 : Array
        Frequency of each mode at the segment midpoint, shape (num_sources, n_modes). Units: Hz.

    Returns
    -------
    fresnel_argument : Array
        Dimensionless Fresnel argument, shape (num_sources, n_modes, n_bins).
    """
    fresnel_argument = jnp.sqrt(2*f_dot_0[:,:,jnp.newaxis])*(tau + (f_0[:,:,jnp.newaxis]-f)/f_dot_0[:,:,jnp.newaxis])
    return fresnel_argument