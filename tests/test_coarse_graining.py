# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

"""Tests for coarse_graining.py fixes:
1. Adaptive grid first point equals tmin exactly.
2. Post-merger step is max(C, Mdelta_t), never over-sampling relative to
   the user-specified time resolution.
3. Batched inputs work correctly.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from phentax.utils.coarse_graining import (
    _generate_adaptive_grid,
    generate_adaptive_grid,
)

ETA_EQUAL = 0.25  # equal mass
ETA_UNEQUAL = 0.15
TMIN = -1000.0
TMAX = 500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inspiral_C(eta):
    return float((2.0 * jnp.pi / 3.0) * jnp.power(eta / 5.0, 3.0 / 8.0))


# ---------------------------------------------------------------------------
# First-point guarantee
# ---------------------------------------------------------------------------


class TestFirstPoint:
    def test_first_point_equals_tmin_equal_mass(self):
        Mdelta_t = 3.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        assert (
            float(grid[mask][0]) == TMIN
        ), f"Expected first valid point == {TMIN}, got {float(grid[mask][0])}"

    def test_first_point_equals_tmin_unequal_mass(self):
        Mdelta_t = 5.0
        grid, mask = _generate_adaptive_grid(
            ETA_UNEQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        assert float(grid[mask][0]) == TMIN

    def test_first_point_equals_tmin_small_Mdelta_t(self):
        # Mdelta_t < C: C_post = C, first point should still be tmin
        C = _inspiral_C(ETA_EQUAL)
        Mdelta_t = C * 0.5  # smaller than C
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        assert float(grid[mask][0]) == TMIN

    def test_first_point_equals_tmin_large_Mdelta_t(self):
        # Mdelta_t >> C: C_post = Mdelta_t
        Mdelta_t = 50.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        assert float(grid[mask][0]) == TMIN


# ---------------------------------------------------------------------------
# Post-merger point count
# ---------------------------------------------------------------------------


class TestPostMergerSampling:
    def _count_post_merger(self, grid, mask):
        """Number of valid grid points with t > 0."""
        valid = grid[mask]
        return int(jnp.sum(valid > 0.0))

    def test_post_merger_uses_Mdelta_t_when_larger(self):
        """When Mdelta_t > C, post-merger count should match ceil(tmax/Mdelta_t)."""
        C = _inspiral_C(ETA_EQUAL)
        Mdelta_t = 3.0  # larger than C ≈ 0.42
        assert Mdelta_t > C

        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        n_post = self._count_post_merger(grid, mask)

        import math

        expected = math.ceil(TMAX / Mdelta_t)
        # Allow ±1 for rounding at the boundary
        assert (
            abs(n_post - expected) <= 1
        ), f"Expected ~{expected} post-merger points with Mdelta_t={Mdelta_t}, got {n_post}"

    def test_post_merger_uses_C_when_larger(self):
        """When C > Mdelta_t, post-merger count should match ceil(tmax/C)."""
        C = _inspiral_C(ETA_EQUAL)
        Mdelta_t = C * 0.1  # smaller than C
        assert C > Mdelta_t

        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        n_post = self._count_post_merger(grid, mask)

        import math

        expected = math.ceil(TMAX / C)
        assert (
            abs(n_post - expected) <= 1
        ), f"Expected ~{expected} post-merger points with C={C:.3f}, got {n_post}"

    def test_large_Mdelta_t_gives_fewer_post_merger_than_small(self):
        """Larger Mdelta_t should produce fewer post-merger points."""
        grid_fine, mask_fine = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, 1.0, max_steps=5000
        )
        grid_coarse, mask_coarse = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, 10.0, max_steps=5000
        )
        n_fine = self._count_post_merger(grid_fine, mask_fine)
        n_coarse = self._count_post_merger(grid_coarse, mask_coarse)
        assert n_coarse < n_fine, (
            f"Coarser Mdelta_t should yield fewer post-merger points: "
            f"fine={n_fine}, coarse={n_coarse}"
        )


# ---------------------------------------------------------------------------
# Grid validity
# ---------------------------------------------------------------------------


class TestGridValidity:
    def test_all_valid_points_within_bounds(self):
        Mdelta_t = 3.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        valid = grid[mask]
        assert float(jnp.min(valid)) >= TMIN
        assert float(jnp.max(valid)) <= TMAX

    def test_grid_ascending(self):
        Mdelta_t = 3.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        valid = grid[mask]
        diffs = jnp.diff(valid)
        assert bool(jnp.all(diffs >= 0.0)), "Grid must be non-decreasing"

    def test_merger_time_included(self):
        """t=0 (merger time) must appear in the grid."""
        Mdelta_t = 3.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        valid = grid[mask]
        assert bool(jnp.any(valid == 0.0)), "t=0 (merger) must be a grid point"

    def test_padding_equals_tmin(self):
        """Padded (invalid) entries must all equal tmin."""
        Mdelta_t = 3.0
        grid, mask = _generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, Mdelta_t, max_steps=5000
        )
        padded = grid[~mask]
        if padded.size > 0:
            assert bool(jnp.all(padded == TMIN)), "All padding values must equal tmin"


# ---------------------------------------------------------------------------
# Batched interface
# ---------------------------------------------------------------------------


class TestBatchedGrid:
    def test_batch_shapes(self):
        etas = jnp.array([ETA_EQUAL, ETA_UNEQUAL])
        tmins = jnp.array([TMIN, TMIN * 1.5])
        tmaxs = jnp.array([TMAX, TMAX])
        Mdelta_ts = jnp.array([3.0, 5.0])
        max_steps = 5000

        grids, masks = generate_adaptive_grid(
            etas, tmins, tmaxs, Mdelta_ts, max_steps=max_steps
        )
        assert grids.shape == (2, max_steps)
        assert masks.shape == (2, max_steps)

    def test_batch_first_points_equal_tmins(self):
        tmins = jnp.array([TMIN, -500.0])
        grids, masks = generate_adaptive_grid(
            jnp.array([ETA_EQUAL, ETA_UNEQUAL]),
            tmins,
            jnp.array([TMAX, TMAX]),
            jnp.array([3.0, 5.0]),
            max_steps=5000,
        )
        for i in range(2):
            assert float(grids[i][masks[i]][0]) == float(
                tmins[i]
            ), f"Batch {i}: grid[0]={float(grids[i][masks[i]][0])} != tmin={float(tmins[i])}"

    def test_scalar_inputs_broadcast(self):
        """Single (non-batched) scalar inputs should work via atleast_1d."""
        grids, masks = generate_adaptive_grid(
            ETA_EQUAL, TMIN, TMAX, 3.0, max_steps=5000
        )
        assert grids.shape == (1, 5000)
        assert float(grids[masks][0]) == TMIN
