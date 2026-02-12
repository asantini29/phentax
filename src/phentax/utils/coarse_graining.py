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
import jax.numpy as jnp
from jaxtyping import Array


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

    omega_lo = 0.25 * jnp.power(-eta * t * 0.2, -0.375)
    return 1.0 / (omega_lo / (2.0 * jnp.pi)) / 12.0


def estimate_adaptive_steps(
    eta: float | Array, tmin: float | Array, tmax: float | Array
) -> int:
    """
    Estimate the number of adaptive grid steps needed across a batch of binaries.

    Uses the analytical solution of the leading-order ODE to predict the
    total number of grid points, then returns a value rounded up to the
    nearest 5000 for JIT cache friendliness.

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
        up to the nearest 5000.
    """
    eta = jnp.atleast_1d(jnp.asarray(eta, dtype=jnp.float64))
    tmin = jnp.atleast_1d(jnp.asarray(tmin, dtype=jnp.float64))
    tmax = jnp.atleast_1d(jnp.asarray(tmax, dtype=jnp.float64))

    C = (2.0 * jnp.pi / 3.0) * jnp.power(eta / 5.0, 3.0 / 8.0)

    # Uniform region: from tmax to t_thresh=-1 with step C
    N_uniform = jnp.maximum((tmax + 1.0) / C, 0.0)

    # Adaptive region: ODE solution from u_start to u_end = -tmin
    u_start = jnp.maximum(-tmax, 1.0)
    N_adaptive = jnp.maximum(
        (jnp.power(-tmin, 5.0 / 8.0) - jnp.power(u_start, 5.0 / 8.0))
        / (5.0 * C / 8.0),
        0.0,
    )

    # Take max across the batch, add safety margin, round up to nearest 5000
    N_total = int(jnp.ceil(jnp.max(N_uniform + N_adaptive) * 1.2)) + 200
    N_total = int(jnp.ceil(N_total / 5000.0) * 5000)
    return max(N_total, 5000)  # at least 5000


def estimate_adaptive_steps_from_T(
    T: float, delta_t: float = 15.0
) -> int:
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
        5000 for JIT-cache friendliness.
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
    return estimate_adaptive_steps(
        eta_worst, -num_steps, 0.0
    )


@partial(jax.jit, static_argnames=["max_steps"])
def _generate_adaptive_grid(
    eta: float, tmin: float, tmax: float, max_steps: int = 10000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate an adaptive time grid using vectorized operations.

    The grid has two regions generated backwards from tmax:

    1. **Uniform region** (tmax to t = -1): the leading-order frequency
       formula is clamped, giving a constant step size ``C``.
    2. **Adaptive region** (t = -1 to tmin): the step size grows as
       ``C * |t|^{3/8}`` according to the analytical ODE solution.

    The resulting grid is padded with tmin at the beginning (low indices)
    and sorted in ascending order.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    tmin : float
        Minimum time (start of the grid).
    tmax : float
        Maximum time (end of the grid).
    max_steps : int, optional
        Maximum number of steps in the grid, by default 10000.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - grid: Array of shape (max_steps,) containing time points.
        - mask: Boolean array of shape (max_steps,) indicating valid points.
          True means the point is part of the adaptive grid.
          False means it is a padding value (tmin).
    """
    # Step-size constant: dt = C * |t|^{3/8}, clamped to dt = C for |t| < 1
    C = (2.0 * jnp.pi / 3.0) * jnp.power(eta / 5.0, 3.0 / 8.0)

    # Threshold time: for t > t_thresh the LO formula is clamped
    t_thresh = -1.0

    # Phase 1  – uniform steps from tmax backwards with step C
    N_uniform = jnp.maximum((tmax - t_thresh) / C, 0.0)

    # Phase 2  – adaptive steps, ODE: du/dn = C*u^{3/8}, u = -t
    # Solution: u(n) = (u0^{5/8} + 5C/8 * n)^{8/5}
    u_start = jnp.maximum(-tmax, -t_thresh)  # max(-tmax, 1.0)
    u_start_pow = jnp.power(u_start, 5.0 / 8.0)

    # All indices at once – fully parallel
    indices = jnp.arange(max_steps, dtype=jnp.float64)

    # Uniform part: t = tmax - idx * C
    t_uniform = tmax - indices * C

    # Adaptive part: t = -(u_start^{5/8} + 5C/8 * a_idx)^{8/5}
    a_idx = jnp.maximum(indices - N_uniform, 0.0)
    t_adaptive = -jnp.power(u_start_pow + (5.0 * C / 8.0) * a_idx, 8.0 / 5.0)

    # Select: uniform for the first N_uniform steps, adaptive after
    in_uniform = indices < N_uniform
    t_grid = jnp.where(in_uniform, t_uniform, t_adaptive)

    # Validity mask
    mask = t_grid >= tmin

    # Pad invalid points with tmin
    grid = jnp.where(mask, t_grid, tmin)

    # Flip to ascending order
    grid_asc = jnp.flip(grid)
    mask_asc = jnp.flip(mask)

    return grid_asc, mask_asc


@partial(jax.jit, static_argnames=["max_steps"])
def generate_adaptive_grid(
    etas: float | Array,
    tmins: float | Array,
    tmaxs: float | Array,
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

    return jax.vmap(partial(_generate_adaptive_grid, max_steps=max_steps))(
        etas, tmins, tmaxs
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

    grid, mask = _generate_adaptive_grid(eta, tmin, tmax, max_steps=15000)

    ugrid, umask = _generate_uniform_grid(tmin, tmax, dt, max_steps=15000)

    etas = jnp.array([0.25, 0.2])
    tmins = jnp.array([-1000.0, -1500.0])
    tmaxs = jnp.array([500.0, 300.0])
    dts = jnp.array([0.1, 0.2])

    grids, masks = generate_adaptive_grid(etas, tmins, tmaxs, max_steps=15000)
    ugrids, umasks = generate_uniform_grid(tmins, tmaxs, dts, max_steps=15000)
    breakpoint()
