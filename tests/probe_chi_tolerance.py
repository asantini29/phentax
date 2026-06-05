"""
WARN-01 chi spin tolerance probe script for Phentax.

Standalone diagnostic — NOT collected by pytest.
Run with: uv run python tests/probe_chi_tolerance.py

Probes chi1z and chi2z gradient correctness vs central finite-difference at the
WARN-01 spin-sensitive test point (m1=60, m2=10, chi1z=0.8, chi2z=0.6).

Uses the pinned-interior-index reduction (amplitudes[0, 0, mid_idx]) to avoid
mask-boundary confounds. The minimum rel_err across the h sweep determines
rtol_chi for use in test_gradient_correctness.py (Phase 3, D-03).
"""

import sys

from phentax.utils.config import configure_jax

configure_jax(platform="cpu", enable_x64=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: F401,E402

from phentax.waveform import IMRPhenomTHM  # noqa: E402

# ---------------------------------------------------------------------------
# WARN-01 spin-sensitive test point (module-level constants)
# ---------------------------------------------------------------------------
M1_SPIN = jnp.array(60.0)
M2_SPIN = jnp.array(10.0)
CHI1Z_SPIN = jnp.array(0.8)
CHI2Z_SPIN = jnp.array(0.6)
DISTANCE_SPIN = jnp.array(1.0)
PHI_REF_SPIN = jnp.array(1.0)  # NOT 0.0 — use 1.0 per OQ3 to avoid symmetry-zero
INCLINATION_SPIN = jnp.array(1.0)
PSI_SPIN = jnp.array(0.5)
DELTA_T_SPIN = 1.0 / 4096.0
T_SPIN = 10.0
F_MIN_SPIN = 20.0
F_REF_SPIN = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_mid_idx(model):
    """Return mask-interior time index stable across small chi perturbations.

    Calls compute_amp_phase at the WARN-01 base parameters and pins the midpoint
    of the True-indices as a static Python int (never a traced quantity).
    """
    _, _, mask, amplitudes, _ = model.compute_amp_phase(
        M1_SPIN,
        M2_SPIN,
        CHI1Z_SPIN,
        CHI2Z_SPIN,
        DISTANCE_SPIN,
        PHI_REF_SPIN,
        INCLINATION_SPIN,
        PSI_SPIN,
        delta_t=DELTA_T_SPIN,
        T=T_SPIN,
        f_min=F_MIN_SPIN,
        f_ref=F_REF_SPIN,
    )
    true_idx = jnp.where(mask[0])[0]
    n = len(true_idx)
    assert n > 0, (
        "_get_mid_idx: mask[0] is all-False — no valid time points found. "
        "Check that T_SPIN, F_MIN_SPIN, and mass parameters produce a non-empty waveform."
    )
    return int(true_idx[n // 2])  # midpoint of valid region — static Python int


def _central_fd(fn, x, h):
    """Central finite-difference gradient for scalar x."""
    x_arr = jnp.array(float(x))
    h_arr = jnp.array(float(h))
    return float((fn(x_arr + h_arr) - fn(x_arr - h_arr)) / (2.0 * h_arr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model)
    print(f"mid_idx = {mid_idx}")

    h_values = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]

    # ------------------------------------------------------------------
    # chi1z section
    # ------------------------------------------------------------------
    print()
    print("=== chi1z gradient probe ===")

    def fn_chi1z(chi1z):
        _, _, mask, amplitudes, _ = model.compute_amp_phase(
            M1_SPIN,
            M2_SPIN,
            chi1z,
            CHI2Z_SPIN,
            DISTANCE_SPIN,
            PHI_REF_SPIN,
            INCLINATION_SPIN,
            PSI_SPIN,
            delta_t=DELTA_T_SPIN,
            T=T_SPIN,
            f_min=F_MIN_SPIN,
            f_ref=F_REF_SPIN,
        )
        return amplitudes[0, 0, mid_idx]  # pinned interior index — NOT sum

    g_ad_chi1z = float(jax.grad(fn_chi1z)(CHI1Z_SPIN))
    print(f"AD gradient chi1z: {g_ad_chi1z:.6e}")
    if not jnp.isfinite(jnp.array(g_ad_chi1z)):
        print("FAIL: chi1z AD gradient is not finite")
        sys.exit(1)

    min_rel_err_chi1z = float("inf")
    best_h_chi1z = None
    for h in h_values:
        fd = _central_fd(fn_chi1z, CHI1Z_SPIN, h)
        rel_err = abs(g_ad_chi1z - fd) / (abs(fd) + abs(g_ad_chi1z) + 1e-50)
        print(f"  h={h:.1e}  fd={fd:.4e}  rel_err={rel_err:.3e}")
        if rel_err < min_rel_err_chi1z:
            min_rel_err_chi1z = rel_err
            best_h_chi1z = h

    print(f"  Best h={best_h_chi1z:.1e}  min_rel_err={min_rel_err_chi1z:.3e}")
    rtol_chi1z = max(min_rel_err_chi1z * 10.0, 1e-4)
    print(f"  Recommended rtol_chi1z = {rtol_chi1z:.1e}")

    # ------------------------------------------------------------------
    # chi2z section
    # ------------------------------------------------------------------
    print()
    print("=== chi2z gradient probe ===")

    def fn_chi2z(chi2z):
        _, _, mask, amplitudes, _ = model.compute_amp_phase(
            M1_SPIN,
            M2_SPIN,
            CHI1Z_SPIN,
            chi2z,
            DISTANCE_SPIN,
            PHI_REF_SPIN,
            INCLINATION_SPIN,
            PSI_SPIN,
            delta_t=DELTA_T_SPIN,
            T=T_SPIN,
            f_min=F_MIN_SPIN,
            f_ref=F_REF_SPIN,
        )
        return amplitudes[0, 0, mid_idx]  # pinned interior index — NOT sum

    g_ad_chi2z = float(jax.grad(fn_chi2z)(CHI2Z_SPIN))
    print(f"AD gradient chi2z: {g_ad_chi2z:.6e}")
    if not jnp.isfinite(jnp.array(g_ad_chi2z)):
        print("FAIL: chi2z AD gradient is not finite")
        sys.exit(1)

    min_rel_err_chi2z = float("inf")
    best_h_chi2z = None
    for h in h_values:
        fd = _central_fd(fn_chi2z, CHI2Z_SPIN, h)
        rel_err = abs(g_ad_chi2z - fd) / (abs(fd) + abs(g_ad_chi2z) + 1e-50)
        print(f"  h={h:.1e}  fd={fd:.4e}  rel_err={rel_err:.3e}")
        if rel_err < min_rel_err_chi2z:
            min_rel_err_chi2z = rel_err
            best_h_chi2z = h

    print(f"  Best h={best_h_chi2z:.1e}  min_rel_err={min_rel_err_chi2z:.3e}")
    rtol_chi2z = max(min_rel_err_chi2z * 10.0, 1e-4)
    print(f"  Recommended rtol_chi2z = {rtol_chi2z:.1e}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    rtol_chi = max(rtol_chi1z, rtol_chi2z)  # conservative: use the larger of the two
    print(f"Final rtol_chi (for test_gradient_correctness.py) = {rtol_chi:.1e}")

    any_fail = min_rel_err_chi1z > 1e-2 or min_rel_err_chi2z > 1e-2
    if any_fail:
        print(
            "WARNING: min rel_err > 1e-2 for at least one spin — chi gradient may be wrong"
        )
        sys.exit(1)
    else:
        print("ALL_PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
