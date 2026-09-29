# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Constants
=========

Physical constants and conversion factors.
Currently, we use the `lisaconstants` package.
Some constants, like `G_SI`, are slightly different from the original `phenomxpy` implementation.
"""
import lisaconstants as lc

# Physical constants in SI units
# MSUN_SI = 1.988409902147041637325262574352366540e30  # Solar mass [kg]
# MTSUN_SI = 4.925490947641266978197229498498379006e-6  # Solar mass [s]
# MRSUN_SI = 1.476625038050124729627979840144936351e3  # Solar mass [m]
# PC_SI = 3.085677581491367278913937957796471611e16  # Parsec [m]
# C_SI = 299792458.0  # Speed of light [m/s]
# G_SI = 6.67430e-11  # Gravitational constant [m^3/(kg*s^2)]

# using lisaconstants:
MSUN_SI = lc.SOLAR_MASS  # 1.98848e30
MTSUN_SI = lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**3  # 4.925491025873693e-06
MRSUN_SI = lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**2  # 1476.6250615036158
PC_SI = lc.PARSEC  # 3.0856775814913674e16
C_SI = lc.SPEED_OF_LIGHT  # 299792458.0
G_SI = lc.GRAVITATIONAL_CONSTANT  # 6.674080e-11

# year
YRSID_SI = lc.ASTRONOMICAL_YEAR

# Conversion factors
MPC_TO_M = 1e6 * PC_SI  # Megaparsec to meters
