
"""
time-frequency.py
================================

Time-frequency utilities for phentax.
"""


import numpy as np 
import math 


def setup_tf_grid(T,
                  dt, 
                  dT):
    '''
    Sets up time-frequency grid for time-frequency domain calculations. 

    Args:
        T (float): Total observation time (s) 
        dt (float): Sampling time (s)
        dT (float): Time segment duration (s)
    Returns:
        t_grid (numpy.array): Time grid (s)
        f_grid (numpy.array): Frequency grid (Hz)
    '''

    print("Setting up time-frequency grid...")
    print(f"Total observation time: {T} s")

    # int -> Rounding down 
    nT = int(T/dT) # length of each time chunk
    
    print(f"Number of time segments: {nT}")

    t_grid = np.arange(nT+1)*dT # nT+1 as we want nT time segments, which means nT+1 time points
    # Frequency resolution 
    dF =f_min= (1/dT)
    f_max = 1/(dt)/ 2 # Nyquist frequency 

    nF = int((f_max - f_min) / dF) + 1 # frequency bins per segment
    
    # Frequency grid (neglecting DC component)
    f_grid = np.arange(1,nF+1) * dF  # segment frequencies
    # Time grid
    print('NOTE: The time grid calculates the number of segments as int(T/dT), which means it rounds down the number of segments.')
    return(t_grid,f_grid)
