# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Constants
=========

Physical constants and conversion factors.
"""
import lisaconstants as lc

# Physical constants in SI units
MSUN_SI = 1.988409902147041637325262574352366540e30  # Solar mass [kg]
MTSUN_SI = 4.925490947641266978197229498498379006e-6  # Solar mass [s]
MRSUN_SI = 1.476625038050124729627979840144936351e3  # Solar mass [m]
PC_SI = 3.085677581491367278913937957796471611e16  # Parsec [m]
C_SI = 299792458.0  # Speed of light [m/s]
G_SI = 6.67430e-11  # Gravitational constant [m^3/(kg*s^2)]

# Conversion factors
MPC_TO_M = 1e6 * PC_SI  # Megaparsec to meters

# year
YRSID_SI = lc.ASTRONOMICAL_YEAR
