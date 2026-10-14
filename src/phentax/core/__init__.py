# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

"""
Core functions for the evaluation of IMRPhenomT(HM) amplitude and phase.
"""

from .amplitude import (
    AmplitudeCoeffs,
    compute_amplitude_coeffs_22,
    compute_amplitude_coeffs_hm,
    imr_amplitude,
    imr_amplitude_dot
)
from .phase import (
    PhaseCoeffs,
    compute_phase_coeffs_22,
    compute_phase_coeffs_hm,
    imr_phase,
    imr_omega,
    imr_omega_dot,
)
