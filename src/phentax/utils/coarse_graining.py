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


@partial(jax.jit, static_argnames=["max_steps"])
def _generate_adaptive_grid(
    eta: float, tmin: float, tmax: float, max_steps: int = 10000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate an adaptive time grid using jax.lax.scan.

    The grid is generated backwards from tmax to tmin.
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

    # State: (current_time, is_finished)
    # We generate backwards from tmax to tmin
    init_val = (tmax, False)

    def scan_body(carry, _):
        t_curr, is_finished = carry

        # Calculate step size at current time
        # Use a safe time for the power law to avoid NaNs in padding region
        # (though we mask the result, the computation must be safe)
        safe_t = jnp.minimum(t_curr, -1.0)
        dt = leading_order_delta_t(eta, safe_t)

        # Calculate next time candidate
        t_next = t_curr - dt

        # Determine if this step finishes the grid (crosses tmin)
        # We are finished if we were already finished OR if we just crossed tmin
        just_finished = t_next <= tmin
        now_finished = is_finished | just_finished

        # Determine the output time for this step
        # If we were already finished, pad with tmin
        # If we just finished, clamp to tmin (this is the last valid point)
        # If we are still active, use t_next
        t_out = jnp.where(is_finished, tmin, jnp.where(just_finished, tmin, t_next))

        # Determine if this point is valid
        # It is valid if we were NOT finished at the start of this step
        # (This includes the point that hits tmin exactly/clamped)
        is_valid = ~is_finished

        new_carry = (t_out, now_finished)
        return new_carry, (t_out, is_valid)

    # Run scan
    # We generate max_steps-1 points (since tmax is the 0th point)
    _, (grid_body, mask_body) = jax.lax.scan(
        scan_body, init_val, None, length=max_steps - 1
    )

    # Construct full grid: [tmax, ...body...]
    full_grid = jnp.concatenate([jnp.array([tmax]), grid_body])
    full_mask = jnp.concatenate([jnp.array([True]), mask_body])

    # Flip to get ascending order: [tmin (padded), ..., tmin, ..., tmax]
    full_grid_asc = jnp.flip(full_grid)
    full_mask_asc = jnp.flip(full_mask)

    return full_grid_asc, full_mask_asc


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
