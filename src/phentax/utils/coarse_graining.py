# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

"""
Coarse graining
============================

Utility functions for the creation of time grids.
"""

from functools import partial
from typing import Callable

import jax
jax.config.update("jax_enable_x64", True)  # Use double precision for time calculations
import jax.numpy as jnp
from jaxtyping import Array

SCALE_FACTOR = 12.0  # This is a tunable parameter that controls the overall density of the grid.
BUCKET_SIZE = 2000  # Number of steps per bucket for JIT cache friendliness.  Must be >= 1.


def leading_order_factor(eta: float | Array) -> float | Array:
    """
    Compute the leading-order (positive) factor C in the adaptive time step formula.
    
    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.

    Returns
    -------
    float | Array
        Leading-order factor C such that :math:`\\Delta t = C \cdot |t|^{3/8}` in the inspiral.
    """
    return (2.0 * jnp.pi * 4.0 / SCALE_FACTOR) * jnp.power(eta / 5.0, 3.0 / 8.0)

def leading_order_delta_t(eta: float | Array, t: float | Array) -> float | Array:
    """
    Compute adaptive time step at leading order in omega.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio.
    t : float | Array
        Time in units of total mass M.
    Returns
    -------
    float | Array
        Leading order time step delta_t = 1 / (10 * f_LO), where f_LO is the leading
        order GW frequency at time t.
    """

    C = leading_order_factor(eta)
    return C * jnp.power(jnp.abs(t), 3.0 / 8.0)


def estimate_adaptive_steps(
    eta: float | Array, tmin: float | Array, tmax: float | Array
) -> int:
    """
    Estimate the number of adaptive grid steps needed across a batch of binaries.

    Uses the analytical solution of the leading-order ODE to predict the
    total number of grid points, then returns a value rounded up to the
    nearest BUCKET_SIZE for JIT cache friendliness.

    The estimate accounts for the three-region grid structure:

    1. **Post-merger** (tmax → 0): coarse uniform with step ``C_post``.
    2. **Fine uniform** (0 → ≈ −1): step ``C``.
    3. **Adaptive** (≈ −1 → tmin): step ``C |t|^{3/8}``.

    Parameters
    ----------
    eta : float | Array
        Symmetric mass ratio(s).
    tmin : float | Array
        Minimum time(s) (start of the grid).
    tmax : float | Array
        Maximum time(s) (end of the grid).

    Returns
    -------
    int
        Estimated number of required steps (with safety margin), rounded
        up to the nearest BUCKET_SIZE.
    """
    eta = jnp.atleast_1d(jnp.asarray(eta, dtype=jnp.float64))
    tmin = jnp.atleast_1d(jnp.asarray(tmin, dtype=jnp.float64))
    tmax = jnp.atleast_1d(jnp.asarray(tmax, dtype=jnp.float64))

    C = leading_order_factor(eta)
    C_post = jnp.maximum(1.0, C)

    # Post-merger region: tmax → 0 with step C_post (+1 for the t=0 node)
    N_post = jnp.where(tmax > 0.0, jnp.ceil(tmax / C_post), 0.0) + 1.0

    # Fine uniform region: 0 → ≈ −1 with step C
    N_fine = jnp.ceil(1.0 / C)

    # Adaptive region: ODE solution from u_fine_end to |tmin|
    u_fine_end = N_fine * C
    N_adaptive = jnp.where(
        -tmin > u_fine_end,
        jnp.maximum(
            (jnp.power(-tmin, 5.0 / 8.0) - jnp.power(u_fine_end, 5.0 / 8.0))
            / (5.0 * C / 8.0),
            0.0,
        ),
        0.0,
    )

    # Take max across the batch, add safety margin, round up to nearest BUCKET_SIZE
    N_total = int(jnp.ceil(jnp.max(N_post + N_fine + N_adaptive) * 1.2)) + 200
    N_total = int(jnp.ceil(N_total / BUCKET_SIZE) * BUCKET_SIZE)
    return max(N_total, BUCKET_SIZE)  # at least BUCKET_SIZE


def estimate_adaptive_steps_from_T(T: float, delta_t: float = 15.0) -> int:
    """
    Estimate adaptive grid size from observation time and time step only.

    Uses worst-case symmetric mass ratio (eta = 0.25, equal mass) to
    guarantee the grid is large enough for any binary.  The result depends
    only on user-controlled quantities, so it can be computed once at
    init time without causing JIT recompilation.

    Parameters
    ----------
    T : float
        Total observation time in seconds.
    delta_t : float, default 15.0
        Time step in seconds.

    Returns
    -------
    int
        Estimated number of required steps, rounded up to the nearest
        BUCKET_SIZE for JIT-cache friendliness.
    """
    # Worst-case: equal mass eta=0.25 gives the densest adaptive grid.
    # tmin ~ -T/delta_t (rough conversion to mass-scaled time), tmax ~ 0
    # This is conservative because the actual mass-scaled time span is
    # almost always shorter.
    eta_worst = 0.25
    num_steps = T / delta_t
    # Use the same formula as estimate_adaptive_steps but with
    # conservative bounds derived from num_steps.
    # In mass-scaled units, the grid spans roughly [-num_steps * delta_t_M, 0]
    # where delta_t_M ~ delta_t / M_total_seconds.  For the purpose of
    # bounding, we use num_steps directly as a proxy for |tmin|.
    # The uniform grid would need num_steps points; the adaptive grid is
    # always sparser, so num_steps is a safe upper bound.
    return estimate_adaptive_steps(eta_worst, -num_steps, 0.0)


@partial(jax.jit, static_argnames=["max_steps"])
def _generate_adaptive_grid(
    eta: float, tmin: float, tmax: float, Mdelta_t: float, max_steps: int = 10000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate an adaptive time grid with t=0 (merger time) always included.

    The grid has three regions generated backwards from tmax:

    1. **Post-merger region** (tmax to 0): coarse uniform steps with
       step size ``C_post = max(C, Mdelta_t)`` so the ringdown is never
       sampled more densely than the user-specified time resolution.
    2. **Fine uniform region** (0 to t ≈ -1): constant step size ``C``
       for the late inspiral / merger.
    3. **Adaptive region** (t ≈ -1 to tmin): the step size grows as
       ``C * |t|^{3/8}`` according to the analytical ODE solution.

    The resulting grid is sorted in ascending order.  The merger time
    ``t = 0`` is always included as a grid point, and the first element
    of the returned grid is guaranteed to equal ``tmin``.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    tmin : float
        Minimum time (start of the grid, typically large and negative).
    tmax : float
        Maximum time (end of the grid, typically positive).
    Mdelta_t : float
        User-specified time step in units of total mass M.  Used as the
        lower bound for the post-merger step size so that the adaptive
        grid never over-samples the ringdown relative to the uniform grid.
    max_steps : int, optional
        Maximum number of steps in the grid, by default 10000.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - grid: Array of shape (max_steps,) containing time points.
          The first element is exactly ``tmin``; padding values (beyond
          the last valid point) are also set to ``tmin``.
        - mask: Boolean array of shape (max_steps,) indicating valid points.
          True means the point is part of the adaptive grid.
          False means it is a padding value (tmin).
    """
    # Inspiral step-size constant: dt = C * |t|^{3/8}, clamped to C for |t| < 1
    C = leading_order_factor(eta)
    # Post-merger step: at least as large as the user's time resolution so we
    # never over-sample the ringdown compared to the uniform grid.
    C_post = jnp.maximum(C, Mdelta_t)

    # --- Region sizes ---
    # Region 1 – post-merger: from tmax down to just above 0, step C_post
    N_post = jnp.where(tmax > 0.0, jnp.ceil(tmax / C_post), 0.0)

    # Region 2 – fine uniform: from 0 down to ≈ −1 with step C
    N_fine = jnp.ceil(1.0 / C)

    # Boundary of fine region in |t|
    u_fine_end = N_fine * C  # ≈ 1, exactly a multiple of C
    u_fine_pow = jnp.power(u_fine_end, 5.0 / 8.0)

    # --- Build grid (backwards from tmax) ---
    indices = jnp.arange(max_steps, dtype=jnp.float64)

    # Post-merger: t = tmax - i * C_post
    t_post = tmax - indices * C_post

    # Fine uniform (including t=0 at offset 0): t = -(i - N_post) * C
    fine_offset = jnp.maximum(indices - N_post, 0.0)
    t_fine = -fine_offset * C

    # Adaptive: ODE solution continuing from u_fine_end
    adapt_offset = jnp.maximum(indices - N_post - N_fine, 0.0)
    t_adapt = -jnp.power(u_fine_pow + (5.0 * C / 8.0) * adapt_offset, 8.0 / 5.0)

    # Select region for each index
    in_post = indices < N_post
    in_fine = (indices >= N_post) & (indices <= N_post + N_fine)

    t_grid = jnp.where(in_post, t_post, jnp.where(in_fine, t_fine, t_adapt))

    # Validity mask: point must lie within [tmin, tmax]
    mask = (t_grid >= tmin) & (t_grid <= tmax)

    # Pad invalid points with tmin
    grid = jnp.where(mask, t_grid, tmin)

    # Flip to ascending order
    grid_asc = jnp.flip(grid)
    mask_asc = jnp.flip(mask)

    # After the flip, valid points occupy the HIGH indices and padding (tmin)
    # occupies the LOW indices (the grid was built descending from tmax, so
    # the earliest times end up near the end of the pre-flip array and at
    # the beginning of the flipped array after the valid region starts).
    # The first valid entry is at jnp.argmax(mask_asc); force it to be
    # exactly tmin so the returned grid always starts at the requested
    # minimum time.
    first_valid = jnp.argmax(mask_asc)
    grid_asc = grid_asc.at[first_valid].set(tmin)

    return grid_asc, mask_asc


@partial(jax.jit, static_argnames=["max_steps"])
def generate_adaptive_grid(
    etas: float | Array,
    tmins: float | Array,
    tmaxs: float | Array,
    Mdelta_ts: float | Array,
    max_steps: int = 10000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batch version of generate_adaptive_grid.

    Parameters
    ----------
    etas : float | Array
        Symmetric mass ratios.
    tmins : float | Array
        Minimum times (start of the valid region).
    tmaxs : float | Array
        Maximum times (end of the grid).
    Mdelta_ts : float | Array
        User-specified time steps in units of total mass M.  Used to
        set the post-merger step size (see ``_generate_adaptive_grid``).
    max_steps : int, optional
        Maximum number of steps in the grid.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - grids: (batch, max_steps)
        - masks: (batch, max_steps)
    """
    etas = jnp.atleast_1d(etas)
    tmins = jnp.atleast_1d(tmins)
    tmaxs = jnp.atleast_1d(tmaxs)
    Mdelta_ts = jnp.atleast_1d(Mdelta_ts)
    return jax.vmap(partial(_generate_adaptive_grid, max_steps=max_steps))(
        etas, tmins, tmaxs, Mdelta_ts
    )


@partial(jax.jit, static_argnames=["max_steps"])
def _generate_uniform_grid(
    tmin: float, tmax: float, delta_t: float, max_steps: int = 10000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a uniform time grid with fixed step size.

    The grid is generated backwards from tmax: t[i] = tmax - i * delta_t.
    Points where t < tmin are masked out and padded with tmin.
    The resulting grid is sorted in ascending order.

    Parameters
    ----------
    tmin : float
        Minimum time (start of the valid region).
    tmax : float
        Maximum time (end of the grid).
    delta_t : float
        Time step size (must be positive).
    max_steps : int, optional
        Maximum number of steps in the grid.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - grid: Array of shape (max_steps,) containing time points.
        - mask: Boolean array of shape (max_steps,) indicating valid points.
    """
    # Generate indices [0, 1, ..., max_steps-1]
    indices = jnp.arange(max_steps)

    # Compute time points backwards from tmax
    # t_raw = [tmax, tmax-dt, tmax-2dt, ...]
    t_raw = tmax - indices * delta_t

    # Create mask for valid points
    # Valid if t >= tmin
    mask = t_raw >= tmin

    # Apply mask: replace invalid points with tmin (safe padding)
    grid = jnp.where(mask, t_raw, tmin)

    # Flip to get ascending order [tmin (padded), ..., tmin, ..., tmax]
    grid_asc = jnp.flip(grid)
    mask_asc = jnp.flip(mask)

    return grid_asc, mask_asc


@partial(jax.jit, static_argnames=["max_steps"])
def generate_uniform_grid(
    tmins: float | Array,
    tmaxs: float | Array,
    delta_ts: float | Array,
    max_steps: int = 10000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batch version of generate_uniform_grid.

    Parameters
    ----------
    tmins : float | Array
        Minimum times (start of the valid region).
    tmaxs : float | Array
        Maximum times (end of the grid).
    delta_ts : float | Array
        Time step sizes.
    max_steps : int, optional
        Maximum number of steps in the grid.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - grids: (batch, max_steps)
        - masks: (batch, max_steps)
    """
    tmins = jnp.atleast_1d(tmins)
    tmaxs = jnp.atleast_1d(tmaxs)
    delta_ts = jnp.atleast_1d(delta_ts)

    return jax.vmap(partial(_generate_uniform_grid, max_steps=max_steps))(
        tmins, tmaxs, delta_ts
    )


def masked_evaluate(
    time_grid: Array,
    mask: Array,
    func: Callable[[Array], Array],
    fill_value: float | complex = 0.0j,
) -> Array:
    """
    Evaluate a function on a grid only where the mask is True.

    This uses jax.lax.cond inside a vmap to avoid expensive computation
    on padded/invalid grid points. Since the grid is sorted (padded values
    are clustered), this is efficient on GPUs due to low warp divergence.

    Parameters
    ----------
    time_grid : Array
        Time points.
    mask : Array
        Boolean mask (True indicates valid points).
    func : Callable[[Array], Array]
        Function to evaluate. Must accept a scalar time and return a scalar/array.
    fill_value : float | complex, optional
        Value to return where mask is False, by default 0.0j.

    Returns
    -------
    Array
        Result of func(t) where mask is True, fill_value otherwise.
    """

    def _eval_point(t, m):
        return jax.lax.cond(
            m,
            lambda _: func(t),
            lambda _: fill_value,
            operand=None,
        )

    return jax.vmap(_eval_point)(time_grid, mask)


if __name__ == "__main__":
    # Simple test
    eta = 0.25
    tmin = -1000.0
    tmax = 500.0
    dt = 0.1

    grid, mask = _generate_adaptive_grid(eta, tmin, tmax, dt, max_steps=15000)

    ugrid, umask = _generate_uniform_grid(tmin, tmax, dt, max_steps=15000)

    etas = jnp.array([0.25, 0.2])
    tmins = jnp.array([-1000.0, -1500.0])
    tmaxs = jnp.array([500.0, 300.0])
    dts = jnp.array([0.1, 0.2])

    grids, masks = generate_adaptive_grid(etas, tmins, tmaxs, dts, max_steps=15000)
    ugrids, umasks = generate_uniform_grid(tmins, tmaxs, dts, max_steps=15000)