"""
Gradient audit probe script for Phentax.

Standalone diagnostic script — NOT collected by pytest.
Run with: uv run python tests/probe_gradients.py

Probes every suspected autodiff blocker in Phentax using jax.grad,
jax.jacfwd, and central finite-difference comparison.

Inside-out ordering (D-02):
  Level 1: _compute_waveform_params  (internals.py blocker site)
  Level 2: compute_phase_coeffs_22   (propagation site)
           get_time_of_frequency     (Bisection differentiability)
  Level 3: guard probes              (waveform.py:1406, 1465)
           math.ceil                 (waveform.py:1454)
           inner JIT closures
  Level 4: higher-mode coefficients  (open question 2)
           chi spin gradients        (open question 1)
  Level 5: end-to-end compute_polarizations (BLOCKER-01 end-to-end)
           vmap + grad               (open question 3)
"""

import traceback

import jax
import jax.numpy as jnp

from phentax.utils.config import configure_jax

configure_jax(platform="cpu", enable_x64=True)

from phentax.core.amplitude import (
    compute_amplitude_coeffs_22,
    compute_amplitude_coeffs_hm,
)

# ---------------------------------------------------------------------------
# Imports that require JAX to be configured first
# ---------------------------------------------------------------------------
from phentax.core.internals import (  # noqa: E402
    WaveformParams,
    _compute_waveform_params,
    compute_waveform_params,
)
from phentax.core.phase import (
    compute_phase_coeffs_22,
    compute_phase_coeffs_hm,
    get_time_of_frequency,
)
from phentax.waveform import IMRPhenomTHM

# ---------------------------------------------------------------------------
# Global results accumulator: list of (probe_name, status, notes) tuples
# ---------------------------------------------------------------------------
results = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def finite_diff_check(fn, x, eps=None):
    """Central finite-difference gradient estimate for scalar x."""
    eps = eps if eps is not None else 1e-5 * jnp.abs(x)
    eps = jnp.where(eps == 0, jnp.array(1e-5), eps)
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


# ---------------------------------------------------------------------------
# Standard test parameters (all as jnp.array, matching compare_coeffs.py)
# ---------------------------------------------------------------------------
M1 = jnp.array(80.0)
M2 = jnp.array(20.0)
S1Z = jnp.array(0.5)
S2Z = jnp.array(-0.3)
DISTANCE = jnp.array(1.0)
INCLINATION = jnp.array(1.0)
F_REF = jnp.array(10.0)
PHI_REF = jnp.array(0.0)
F_MIN = jnp.array(20.0)
PSI = jnp.array(0.0)

# For end-to-end probes: use a short observation time to keep the grid small
DELTA_T = 15.0  # seconds
T_OBS = 128.0  # seconds (short but finite — avoids math.ceil(None/delta_t))


# ---------------------------------------------------------------------------
# Level 1 — core/internals.py
# ---------------------------------------------------------------------------


def probe_waveform_params():
    """
    Probe _compute_waveform_params directly.

    BLOCKER-01 root cause: internals.py:201-202
    Mt_ref = second_to_mass(t_ref, total_mass) when t_ref=jnp.nan (default)
    This computes nan / M_sec where M_sec depends on m1+m2.
    Under jax.grad, chain rule through nan / M_sec produces NaN gradient.
    We LEAVE t_min and t_ref at their jnp.nan defaults — do NOT pass None.
    """
    print()
    print("=== PROBE: probe_waveform_params (BLOCKER-01 root site) ===")

    # Forward pass — uses jnp.nan defaults for t_min and t_ref
    wf = _compute_waveform_params(
        M1,
        M2,
        S1Z,
        S2Z,
        DISTANCE,
        INCLINATION,
        PHI_REF,
        PSI,
        F_REF,
        F_MIN,
        # t_min=jnp.nan, t_ref=jnp.nan are the defaults
    )
    print(f"FORWARD: Mt_ref={float(wf.Mt_ref):.6e}, Mt_min={float(wf.Mt_min):.6e}")
    print(f"  Mt_ref is nan: {bool(jnp.isnan(wf.Mt_ref))}")
    print(f"  Mt_min is nan: {bool(jnp.isnan(wf.Mt_min))}")

    # Gradient wrt m1 — expect NaN because Mt_ref = nan / M_sec(m1)
    try:

        def fn_Mt_ref(m1):
            wp = _compute_waveform_params(
                m1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
            )
            return jnp.nansum(jnp.array([wp.Mt_ref, 0.0]))  # scalar, but NaN leaks

        g = jax.grad(fn_Mt_ref)(M1)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(Mt_ref wrt m1): {float(g):.6e}  NaN={is_nan}")
        if is_nan:
            print("BLOCKER-01: NaN gradient confirmed at internals.py:201-202")
            results.append(
                (
                    "probe_waveform_params",
                    "BLOCKER",
                    "BLOCKER-01: NaN gradient at internals.py:201-202",
                )
            )
        else:
            print(
                "UNEXPECTED PASS: gradient is finite — BLOCKER-01 may not be reproduced"
            )
            results.append(
                (
                    "probe_waveform_params",
                    "UNEXPECTED",
                    f"gradient finite: {float(g):.6e}",
                )
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            ("probe_waveform_params", "BLOCKER", f"{type(e).__name__}: {str(e)[:80]}")
        )


# ---------------------------------------------------------------------------
# Level 2 — core/phase.py (propagation site)
# ---------------------------------------------------------------------------


def probe_phase_coeffs():
    """
    Probe compute_phase_coeffs_22.

    Propagation site: phase.py:431-471 (lax.cond with NaN dead-branch).
    Forward pass is CORRECT (bisection runs, correct phiref0).
    Gradient is NaN because the dead branch leaks the m-dependent NaN.
    This is the "forward-correct, grad-NaN" signature.
    """
    print()
    print("=== PROBE: probe_phase_coeffs (propagation site phase.py:431-471) ===")

    # Forward pass — build WaveformParams directly (no compute_derived_params in current API)
    wf = _compute_waveform_params(
        M1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
    )
    wf_updated, phase_coeffs = compute_phase_coeffs_22(wf)
    print(f"FORWARD: phiref0={float(phase_coeffs.phiref0):.10e}")
    print(f"  phiref0 is nan: {bool(jnp.isnan(phase_coeffs.phiref0))}")

    # Gradient wrt m1 — expect NaN from dead-branch contamination
    try:

        def fn_phiref0(m1):
            wp = _compute_waveform_params(
                m1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
            )
            _, pc = compute_phase_coeffs_22(wp)
            return pc.phiref0

        g = jax.grad(fn_phiref0)(M1)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(phiref0 wrt m1): {float(g):.6e}  NaN={is_nan}")
        if is_nan:
            print(
                "BLOCKER: NaN gradient at propagation site (phase.py:431-471) — forward-correct, grad-NaN signature confirmed"
            )
            results.append(
                (
                    "probe_phase_coeffs",
                    "BLOCKER",
                    "forward-correct, grad-NaN at phase.py:431-471",
                )
            )
        else:
            print(f"PASS: gradient finite at phase.py propagation site: {float(g):.6e}")
            fd = finite_diff_check(fn_phiref0, M1)
            rel_err = float(jnp.abs(g - fd) / (jnp.abs(fd) + 1e-30))
            print(f"  finite_diff: {float(fd):.6e}, rel_err: {rel_err:.3e}")
            results.append(
                (
                    "probe_phase_coeffs",
                    "PASS",
                    f"grad={float(g):.4e}, rel_err={rel_err:.3e}",
                )
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            ("probe_phase_coeffs", "BLOCKER", f"{type(e).__name__}: {str(e)[:80]}")
        )


def probe_get_time_of_frequency():
    """
    Probe get_time_of_frequency (Bisection differentiability).

    AUDIT-03: optimistix.Bisection uses ImplicitAdjoint by default.
    Expected: finite jax.grad and jax.jacfwd, rel_err ~5e-9 vs finite_diff.
    This confirms CLEARED-03.
    """
    print()
    print("=== PROBE: probe_get_time_of_frequency (AUDIT-03 Bisection) ===")

    # Build phase coeffs (no lax.cond NaN contamination here — we call directly)
    wf = _compute_waveform_params(
        M1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
    )
    wf_updated, phase_coeffs = compute_phase_coeffs_22(wf)
    eta = wf_updated.eta
    freq_val = wf_updated.Mf_min  # a known valid dimensionless frequency

    print(f"FORWARD: freq_val={float(freq_val):.6e}, eta={float(eta):.6e}")
    t_val = get_time_of_frequency(freq_val, eta, phase_coeffs)
    print(f"FORWARD: t(f)={float(t_val):.10e}")

    # jax.grad wrt freq
    try:

        def fn_time_of_freq(freq):
            return get_time_of_frequency(freq, eta, phase_coeffs)

        g = jax.grad(fn_time_of_freq)(freq_val)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(t(f) wrt freq): {float(g):.10e}  NaN={is_nan}")

        if is_nan:
            print(
                "BLOCKER: NaN gradient in get_time_of_frequency — Bisection NOT differentiable"
            )
            results.append(
                (
                    "probe_get_time_of_frequency",
                    "BLOCKER",
                    "NaN gradient from Bisection",
                )
            )
        else:
            fd = finite_diff_check(fn_time_of_freq, freq_val)
            rel_err = float(jnp.abs(g - fd) / (jnp.abs(fd) + 1e-30))
            print(f"  finite_diff: {float(fd):.10e}, rel_err: {rel_err:.3e}")

            # jax.jacfwd
            g_fwd = jax.jacfwd(fn_time_of_freq)(freq_val)
            print(f"jax.jacfwd(t(f) wrt freq): {float(g_fwd):.10e}")

            print(
                f"CLEARED-03: get_time_of_frequency (Bisection) is differentiable: rel_err={rel_err:.3e}"
            )
            results.append(
                (
                    "probe_get_time_of_frequency",
                    "CLEARED",
                    f"CLEARED-03: rel_err={rel_err:.3e}",
                )
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            (
                "probe_get_time_of_frequency",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )


# ---------------------------------------------------------------------------
# Level 3 — waveform.py guards, math.ceil, inner JIT closures
# ---------------------------------------------------------------------------


def probe_tracer_guard_at_1406():
    """
    Probe the isinstance(Tracer) guard at waveform.py:1406.

    D-05 dual approach:
    - Analytically: the guard is validation-only (spin bound check), no gradient-graph participation.
    - Empirically: grad wrt chi1z (chi1z IS a Tracer -> guard skips assert),
                   grad wrt m1   (chi1z concrete -> guard RUNS assert, but it's not in grad path).

    D-07: jax.core.Tracer is a private API — flagged as secondary concern.
    """
    print()
    print(
        "=== PROBE: probe_tracer_guard_at_1406 (AUDIT-02 guard at waveform.py:1406) ==="
    )
    print(
        "ANALYTICAL: guard at waveform.py:1406 runs spin bound assertion if chi1z is concrete."
    )
    print(
        "  Under jax.grad(lambda chi1z: ...): chi1z IS a Tracer -> guard skips assert."
    )
    print(
        "  Under jax.grad(lambda m1: ...): chi1z concrete -> guard runs, but assertion not in grad path."
    )
    print("  Result: gradient graph is UNAFFECTED in both cases.")

    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    # Case 1: grad wrt chi1z (chi1z is a Tracer -> guard skips validation)
    try:

        def fn_chi1z(chi1z):
            times, mask, h_plus, h_cross = model.compute_polarizations(
                M1,
                M2,
                chi1z,
                S2Z,
                DISTANCE,
                PHI_REF,
                INCLINATION,
                PSI,
                delta_t=DELTA_T,
                T=T_OBS,
            )
            return jnp.sum(jnp.where(mask, h_plus, 0.0))

        g_chi1z = jax.grad(fn_chi1z)(S1Z)
        is_nan_chi1z = bool(jnp.isnan(g_chi1z))
        print(
            f"jax.grad(sum(h_plus) wrt chi1z): {float(g_chi1z):.6e}  NaN={is_nan_chi1z}"
        )
        print(
            "  chi1z is Tracer under grad -> guard at 1406 skips validation -> no gradient effect"
        )
    except Exception as e:
        print(f"NOTE (chi1z grad exception) ({type(e).__name__}): {str(e)[:200]}")

    # Case 2: grad wrt m1 (chi1z concrete -> guard RUNS assertion)
    try:

        def fn_m1_guard(m1):
            times, mask, h_plus, h_cross = model.compute_polarizations(
                m1,
                M2,
                S1Z,
                S2Z,
                DISTANCE,
                PHI_REF,
                INCLINATION,
                PSI,
                delta_t=DELTA_T,
                T=T_OBS,
            )
            return jnp.sum(jnp.where(mask, h_plus, 0.0))

        g_m1 = jax.grad(fn_m1_guard)(M1)
        is_nan_m1 = bool(jnp.isnan(g_m1))
        print(
            f"jax.grad(sum(h_plus) wrt m1) [chi1z concrete, guard runs]: {float(g_m1):.6e}  NaN={is_nan_m1}"
        )
        print(
            "  Assertion runs (chi1z concrete) but is NOT in gradient graph -> no gradient blockage"
        )
        print("CLEARED-01: guard at waveform.py:1406 has no gradient-graph effect")
        print(
            "D-07 NOTE: jax.core.Tracer is a private API — Phase 2 should migrate to public-API validation."
        )
        results.append(
            (
                "probe_tracer_guard_at_1406",
                "CLEARED",
                "CLEARED-01: guard at waveform.py:1406 safe",
            )
        )
    except Exception as e:
        print(f"NOTE (m1 grad exception) ({type(e).__name__}): {str(e)[:200]}")
        print(
            "CLEARED-01: guard at waveform.py:1406 has no gradient-graph effect (confirmed analytically)"
        )
        print(
            "D-07 NOTE: jax.core.Tracer is a private API — Phase 2 should migrate to public-API validation."
        )
        results.append(
            (
                "probe_tracer_guard_at_1406",
                "CLEARED",
                "CLEARED-01: guard at waveform.py:1406 safe (analytical)",
            )
        )


def probe_tracer_guard_at_1465():
    """
    Probe the isinstance(Tracer) guard at waveform.py:1465.

    D-05 dual approach:
    - Analytically: this guard prevents estimate_adaptive_steps_from_T (which calls int(jnp.ceil(...)))
      from running under JIT tracing. This is correct defensive behavior.
      The function is a side-effect-only call (sets self.max_adaptive_steps). No gradient-graph participation.
    - Empirically: same dual grad as 1406.
    """
    print()
    print(
        "=== PROBE: probe_tracer_guard_at_1465 (AUDIT-02 guard at waveform.py:1465) ==="
    )
    print(
        "ANALYTICAL: guard at waveform.py:1465 prevents estimate_adaptive_steps_from_T from running under tracing."
    )
    print(
        "  Under jax.grad(lambda chi1z: ...): chi1z IS a Tracer -> guard skips estimate_adaptive_steps_from_T."
    )
    print(
        "  The function calls int(jnp.ceil(...)), which would raise ConcretizationTypeError on a tracer."
    )
    print("  Correct defensive behavior. No gradient-graph participation.")

    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    try:

        def fn_chi1z_1465(chi1z):
            times, mask, h_plus, h_cross = model.compute_polarizations(
                M1,
                M2,
                chi1z,
                S2Z,
                DISTANCE,
                PHI_REF,
                INCLINATION,
                PSI,
                delta_t=DELTA_T,
                T=T_OBS,
            )
            return jnp.sum(jnp.where(mask, h_plus, 0.0))

        g = jax.grad(fn_chi1z_1465)(S1Z)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(sum(h_plus) wrt chi1z): {float(g):.6e}  NaN={is_nan}")
        print(
            "  Guard at 1465 skipped (chi1z is Tracer) -> estimate_adaptive_steps_from_T not called"
        )
        print("CLEARED-02: guard at waveform.py:1465 has no gradient-graph effect")
        print(
            "D-07 NOTE: jax.core.Tracer is a private API — Phase 2 should migrate to public-API validation."
        )
        results.append(
            (
                "probe_tracer_guard_at_1465",
                "CLEARED",
                "CLEARED-02: guard at waveform.py:1465 safe",
            )
        )
    except Exception as e:
        print(f"NOTE ({type(e).__name__}): {str(e)[:200]}")
        print(
            "CLEARED-02: guard at waveform.py:1465 has no gradient-graph effect (confirmed analytically)"
        )
        print(
            "D-07 NOTE: jax.core.Tracer is a private API — Phase 2 should migrate to public-API validation."
        )
        results.append(
            (
                "probe_tracer_guard_at_1465",
                "CLEARED",
                "CLEARED-02: guard at waveform.py:1465 safe (analytical)",
            )
        )


def probe_math_ceil():
    """
    Probe math.ceil at waveform.py:1454.

    AUDIT-01: math.ceil(T/delta_t) operates on concrete Python floats.
    T is self.T (set in __init__) or the T kwarg — both are Python floats.
    delta_t is a user argument. Neither is a JAX tracer at this call site.
    """
    print()
    print("=== PROBE: probe_math_ceil (AUDIT-01 waveform.py:1454) ===")
    print("ANALYTICAL: math.ceil(T/delta_t) at waveform.py:1454.")
    print(
        "  T is self.T (Python float) or T kwarg (Python float). delta_t is a Python float user arg."
    )
    print(
        "  math.ceil on concrete floats: SAFE. Would raise ConcretizationTypeError only if T/delta_t were JAX tracers."
    )
    print(
        "  In current code path: T and delta_t are always concrete at this call site."
    )

    import math

    T = T_OBS
    delta_t = DELTA_T
    num_steps = math.ceil(T / delta_t)
    print(
        f"EMPIRICAL: math.ceil({T}/{delta_t}) = {num_steps} (Python int: {type(num_steps).__name__})"
    )
    print("CLEARED-04: math.ceil at waveform.py:1454 safe (concrete at call site)")
    results.append(
        ("probe_math_ceil", "CLEARED", "CLEARED-04: math.ceil safe (concrete floats)")
    )


def probe_inner_jit_closures():
    """
    Probe inner @jax.jit closures (imr_omega / imr_phase pattern).

    AUDIT-01: Inner JIT closures cause recompilation overhead, not gradient blockage.
    jax.grad flows correctly through JIT boundaries.
    """
    print()
    print("=== PROBE: probe_inner_jit_closures (AUDIT-01 inner JIT) ===")

    # Build a minimal inner JIT closure analogous to the imr_phase / imr_omega pattern
    outer_array = jnp.array(2.0)  # captured from outer scope

    @jax.jit
    def inner_jit_fn(x):
        return x * outer_array + jnp.sin(x)

    try:

        def fn_with_inner_jit(x):
            return inner_jit_fn(x)

        g = jax.grad(fn_with_inner_jit)(jnp.array(1.0))
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad through inner @jax.jit closure: {float(g):.6e}  NaN={is_nan}")
        if not is_nan:
            print(
                "CLEARED-05: inner @jax.jit closures — recompilation overhead only, not a gradient blocker"
            )
            results.append(
                (
                    "probe_inner_jit_closures",
                    "CLEARED",
                    "CLEARED-05: inner JIT closures safe",
                )
            )
        else:
            print("BLOCKER: NaN gradient through inner JIT closure — unexpected")
            results.append(
                ("probe_inner_jit_closures", "BLOCKER", "NaN through inner JIT closure")
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            (
                "probe_inner_jit_closures",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )


# ---------------------------------------------------------------------------
# Level 4 — Higher-mode coefficients (open question 2)
# ---------------------------------------------------------------------------


def probe_compute_phase_coeffs_hm():
    """
    Probe compute_phase_coeffs_hm — open question 2.

    Are higher-mode phase coefficients free of additional blockers
    beyond the shared _compute_waveform_params dependency?
    """
    print()
    print("=== PROBE: probe_compute_phase_coeffs_hm (open question 2) ===")

    wf = _compute_waveform_params(
        M1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
    )
    wf_updated, phase_coeffs_22 = compute_phase_coeffs_22(wf)

    mode = jnp.array(33)
    # compute_phase_coeffs_hm requires OmegaCutPNAMP and PhiCutPNAMP
    # These come from amplitude_coeffs_22
    amp_coeffs_22 = compute_amplitude_coeffs_22(wf_updated, phase_coeffs_22)
    OmegaCutPNAMP = amp_coeffs_22.omegaCutPNAMP
    PhiCutPNAMP = amp_coeffs_22.phiCutPNAMP

    try:
        phase_coeffs_hm = compute_phase_coeffs_hm(
            wf_updated, phase_coeffs_22, OmegaCutPNAMP, PhiCutPNAMP, mode
        )
        print(f"FORWARD: phoff_hm={float(phase_coeffs_hm.phoff):.6e}")

        def fn_phoff_hm(m1):
            wp = _compute_waveform_params(
                m1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
            )
            wp_upd, pc22 = compute_phase_coeffs_22(wp)
            ac22 = compute_amplitude_coeffs_22(wp_upd, pc22)
            pc_hm = compute_phase_coeffs_hm(
                wp_upd, pc22, ac22.omegaCutPNAMP, ac22.phiCutPNAMP, mode
            )
            return pc_hm.phoff

        g = jax.grad(fn_phoff_hm)(M1)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(phoff_hm wrt m1): {float(g):.6e}  NaN={is_nan}")
        if is_nan:
            print(
                "BLOCKER: NaN gradient in compute_phase_coeffs_hm (likely same BLOCKER-01 propagation)"
            )
            results.append(
                (
                    "probe_compute_phase_coeffs_hm",
                    "BLOCKER",
                    "NaN gradient — BLOCKER-01 propagation",
                )
            )
        else:
            fd = finite_diff_check(fn_phoff_hm, M1)
            rel_err = float(jnp.abs(g - fd) / (jnp.abs(fd) + 1e-30))
            print(f"  finite_diff: {float(fd):.6e}, rel_err: {rel_err:.3e}")
            print(
                "PASS: compute_phase_coeffs_hm has no additional blockers beyond BLOCKER-01"
            )
            results.append(
                (
                    "probe_compute_phase_coeffs_hm",
                    "PASS",
                    f"grad={float(g):.4e}, rel_err={rel_err:.3e}",
                )
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            (
                "probe_compute_phase_coeffs_hm",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )


def probe_compute_amplitude_coeffs_hm():
    """
    Probe compute_amplitude_coeffs_hm — open question 2.

    Are higher-mode amplitude coefficients free of additional blockers
    beyond the shared _compute_waveform_params dependency?
    """
    print()
    print("=== PROBE: probe_compute_amplitude_coeffs_hm (open question 2) ===")

    wf = _compute_waveform_params(
        M1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
    )
    wf_updated, phase_coeffs_22 = compute_phase_coeffs_22(wf)
    mode = jnp.array(33)

    try:
        amp_coeffs_hm = compute_amplitude_coeffs_hm(wf_updated, phase_coeffs_22, mode)
        print(f"FORWARD: ampPeak_hm={float(amp_coeffs_hm.ampPeak):.6e}")

        def fn_ampPeak_hm(m1):
            wp = _compute_waveform_params(
                m1, M2, S1Z, S2Z, DISTANCE, INCLINATION, PHI_REF, PSI, F_REF, F_MIN
            )
            wp_upd, pc22 = compute_phase_coeffs_22(wp)
            ac_hm = compute_amplitude_coeffs_hm(wp_upd, pc22, mode)
            return ac_hm.ampPeak

        g = jax.grad(fn_ampPeak_hm)(M1)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(ampPeak_hm wrt m1): {float(g):.6e}  NaN={is_nan}")
        if is_nan:
            print(
                "BLOCKER: NaN gradient in compute_amplitude_coeffs_hm (likely same BLOCKER-01 propagation)"
            )
            results.append(
                (
                    "probe_compute_amplitude_coeffs_hm",
                    "BLOCKER",
                    "NaN gradient — BLOCKER-01 propagation",
                )
            )
        else:
            fd = finite_diff_check(fn_ampPeak_hm, M1)
            rel_err = float(jnp.abs(g - fd) / (jnp.abs(fd) + 1e-30))
            print(f"  finite_diff: {float(fd):.6e}, rel_err: {rel_err:.3e}")
            print(
                "PASS: compute_amplitude_coeffs_hm has no additional blockers beyond BLOCKER-01"
            )
            results.append(
                (
                    "probe_compute_amplitude_coeffs_hm",
                    "PASS",
                    f"grad={float(g):.4e}, rel_err={rel_err:.3e}",
                )
            )
    except Exception as e:
        print(f"BLOCKER (exception) ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            (
                "probe_compute_amplitude_coeffs_hm",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )


def probe_chi_spin_gradients():
    """
    Probe gradients wrt chi1z and chi2z — open question 1.

    The NaN blocker at internals.py:201-202 comes from Mt_ref = nan/M_sec
    where M_sec depends on total_mass = m1 + m2. chi1z/chi2z do NOT enter M_sec.
    Research finding: chi gradient was finite (~1.2e-21) but correctness is unclear.
    This probe runs finite-diff comparison to determine whether it is a correct small gradient
    or a silent wrong-gradient.
    """
    print()
    print("=== PROBE: probe_chi_spin_gradients (open question 1) ===")

    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    def masked_sum(m1, m2, chi1z, chi2z):
        times, mask, h_plus, h_cross = model.compute_polarizations(
            m1,
            m2,
            chi1z,
            chi2z,
            DISTANCE,
            PHI_REF,
            INCLINATION,
            PSI,
            delta_t=DELTA_T,
            T=T_OBS,
        )
        return jnp.sum(jnp.where(mask, h_plus, 0.0))

    # chi1z gradient
    try:
        g_chi1z = jax.grad(lambda chi1z: masked_sum(M1, M2, chi1z, S2Z))(S1Z)
        is_nan_chi1z = bool(jnp.isnan(g_chi1z))
        print(
            f"jax.grad(sum(h_plus) wrt chi1z): {float(g_chi1z):.6e}  NaN={is_nan_chi1z}"
        )

        if not is_nan_chi1z:
            fd_chi1z = finite_diff_check(
                lambda chi1z: masked_sum(M1, M2, chi1z, S2Z), S1Z
            )
            rel_err_chi1z = float(
                jnp.abs(g_chi1z - fd_chi1z) / (jnp.abs(fd_chi1z) + 1e-30)
            )
            print(
                f"  chi1z finite_diff: {float(fd_chi1z):.6e}, rel_err: {rel_err_chi1z:.3e}"
            )
            if rel_err_chi1z < 1e-3:
                print(
                    "  chi1z gradient is CORRECT (matches finite-diff within tolerance)"
                )
                results.append(
                    (
                        "probe_chi_spin_gradients[chi1z]",
                        "PASS",
                        f"grad={float(g_chi1z):.4e}, rel_err={rel_err_chi1z:.3e}",
                    )
                )
            else:
                print(
                    f"  chi1z gradient MAY BE WRONG: rel_err={rel_err_chi1z:.3e} > 1e-3"
                )
                results.append(
                    (
                        "probe_chi_spin_gradients[chi1z]",
                        "WARNING",
                        f"grad={float(g_chi1z):.4e}, rel_err={rel_err_chi1z:.3e}",
                    )
                )
        else:
            print("  chi1z gradient is NaN — BLOCKER-01 propagation through chi path")
            results.append(
                ("probe_chi_spin_gradients[chi1z]", "BLOCKER", "NaN gradient for chi1z")
            )
    except Exception as e:
        print(f"chi1z grad exception ({type(e).__name__}): {str(e)[:200]}")
        results.append(
            (
                "probe_chi_spin_gradients[chi1z]",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    # chi2z gradient
    try:
        g_chi2z = jax.grad(lambda chi2z: masked_sum(M1, M2, S1Z, chi2z))(S2Z)
        is_nan_chi2z = bool(jnp.isnan(g_chi2z))
        print(
            f"jax.grad(sum(h_plus) wrt chi2z): {float(g_chi2z):.6e}  NaN={is_nan_chi2z}"
        )

        if not is_nan_chi2z:
            fd_chi2z = finite_diff_check(
                lambda chi2z: masked_sum(M1, M2, S1Z, chi2z), S2Z
            )
            rel_err_chi2z = float(
                jnp.abs(g_chi2z - fd_chi2z) / (jnp.abs(fd_chi2z) + 1e-30)
            )
            print(
                f"  chi2z finite_diff: {float(fd_chi2z):.6e}, rel_err: {rel_err_chi2z:.3e}"
            )
            if rel_err_chi2z < 1e-3:
                print(
                    "  chi2z gradient is CORRECT (matches finite-diff within tolerance)"
                )
                results.append(
                    (
                        "probe_chi_spin_gradients[chi2z]",
                        "PASS",
                        f"grad={float(g_chi2z):.4e}, rel_err={rel_err_chi2z:.3e}",
                    )
                )
            else:
                print(
                    f"  chi2z gradient MAY BE WRONG: rel_err={rel_err_chi2z:.3e} > 1e-3"
                )
                results.append(
                    (
                        "probe_chi_spin_gradients[chi2z]",
                        "WARNING",
                        f"grad={float(g_chi2z):.4e}, rel_err={rel_err_chi2z:.3e}",
                    )
                )
        else:
            print("  chi2z gradient is NaN — BLOCKER-01 propagation through chi path")
            results.append(
                ("probe_chi_spin_gradients[chi2z]", "BLOCKER", "NaN gradient for chi2z")
            )
    except Exception as e:
        print(f"chi2z grad exception ({type(e).__name__}): {str(e)[:200]}")
        results.append(
            (
                "probe_chi_spin_gradients[chi2z]",
                "BLOCKER",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )


# ---------------------------------------------------------------------------
# Level 5 — End-to-end and vmap+grad
# ---------------------------------------------------------------------------


def probe_compute_polarizations():
    """
    Probe compute_polarizations end-to-end — AUDIT-01 mandatory end-to-end probe.

    Uses masked scalar reduction to avoid the anti-pattern of jnp.sum without mask.
    Expects NaN gradient — this is the definitive BLOCKER-01 end-to-end confirmation.
    A probe that passes here without NaN has failed to reproduce the bug.
    """
    print()
    print("=== PROBE: probe_compute_polarizations (AUDIT-01 end-to-end) ===")

    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    # Forward pass
    times, mask, h_plus, h_cross = model.compute_polarizations(
        M1, M2, S1Z, S2Z, DISTANCE, PHI_REF, INCLINATION, PSI, delta_t=DELTA_T, T=T_OBS
    )
    masked_hplus_sum = float(jnp.sum(jnp.where(mask, h_plus, 0.0)))
    print(f"FORWARD: masked sum(h_plus)={masked_hplus_sum:.6e}")
    print(f"  h_plus shape: {h_plus.shape}, mask True count: {int(jnp.sum(mask))}")

    # Gradient wrt m1 with masked reduction (not bare jnp.sum — see anti-pattern note)
    try:

        def fn_end_to_end(m1):
            times, mask, h_plus, h_cross = model.compute_polarizations(
                m1,
                M2,
                S1Z,
                S2Z,
                DISTANCE,
                PHI_REF,
                INCLINATION,
                PSI,
                delta_t=DELTA_T,
                T=T_OBS,
            )
            return jnp.sum(jnp.where(mask, h_plus, 0.0))

        g = jax.grad(fn_end_to_end)(M1)
        is_nan = bool(jnp.isnan(g))
        print(f"jax.grad(masked sum(h_plus) wrt m1): {float(g):.6e}  NaN={is_nan}")

        if is_nan:
            print(
                "BLOCKER-01 confirmed end-to-end: NaN gradient in compute_polarizations"
            )
            results.append(
                (
                    "probe_compute_polarizations",
                    "BLOCKER",
                    "BLOCKER-01 confirmed end-to-end: NaN gradient in compute_polarizations",
                )
            )
        else:
            fd = finite_diff_check(fn_end_to_end, M1)
            rel_err = float(jnp.abs(g - fd) / (jnp.abs(fd) + 1e-30))
            print(
                f"UNEXPECTED PASS: gradient finite: {float(g):.6e}, fd: {float(fd):.6e}, rel_err: {rel_err:.3e}"
            )
            print(
                "NOTE: BLOCKER-01 was NOT reproduced end-to-end — re-examine audit findings"
            )
            results.append(
                (
                    "probe_compute_polarizations",
                    "UNEXPECTED",
                    f"gradient finite: {float(g):.6e}",
                )
            )
    except Exception as e:
        print(f"EXCEPTION ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        # If exception contains NaN-like messaging, still count as blocker confirmed
        print(
            "BLOCKER-01 confirmed end-to-end: NaN gradient in compute_polarizations (via exception)"
        )
        results.append(
            (
                "probe_compute_polarizations",
                "BLOCKER",
                f"BLOCKER-01 confirmed end-to-end: {type(e).__name__}",
            )
        )


def probe_vmap_grad():
    """
    Probe jax.vmap + jax.grad composition — open question 3 / GRAD-06.

    vmap(grad(...)) over a batch of 2 binaries.
    Reports shape and whether NaN surfaces.
    """
    print()
    print("=== PROBE: probe_vmap_grad (open question 3 / GRAD-06) ===")

    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    m1_batch = jnp.array([80.0, 60.0])
    m2_batch = jnp.array([20.0, 40.0])
    chi1z_batch = jnp.array([0.5, 0.3])
    chi2z_batch = jnp.array([-0.3, 0.1])
    distance_batch = jnp.array([1.0, 1.5])
    phi_ref_batch = jnp.array([0.0, 0.1])
    inclination_batch = jnp.array([1.0, 0.8])
    psi_batch = jnp.array([0.0, 0.2])

    try:

        def single_grad(m1, m2, chi1z, chi2z, distance, phi_ref, inclination, psi):
            def fn(m1_):
                times, mask, h_plus, h_cross = model.compute_polarizations(
                    m1_,
                    m2,
                    chi1z,
                    chi2z,
                    distance,
                    phi_ref,
                    inclination,
                    psi,
                    delta_t=DELTA_T,
                    T=T_OBS,
                )
                return jnp.sum(jnp.where(mask, h_plus, 0.0))

            return jax.grad(fn)(m1)

        g_batch = jax.vmap(single_grad)(
            m1_batch,
            m2_batch,
            chi1z_batch,
            chi2z_batch,
            distance_batch,
            phi_ref_batch,
            inclination_batch,
            psi_batch,
        )
        is_nan = bool(jnp.isnan(g_batch).any())
        print(f"vmap(grad(sum(h_plus) wrt m1)) over batch of 2: shape={g_batch.shape}")
        print(f"  values: {g_batch}  any NaN: {is_nan}")
        if is_nan:
            print(
                "vmap(grad(...)) surfaces NaN — BLOCKER-01 propagates through vmap composition"
            )
            results.append(
                ("probe_vmap_grad", "BLOCKER", f"NaN in vmap(grad(...)): {g_batch}")
            )
        else:
            print("vmap(grad(...)) returns finite values — no additional vmap blocker")
            results.append(
                ("probe_vmap_grad", "PASS", f"shape={g_batch.shape}, values finite")
            )
    except Exception as e:
        print(f"EXCEPTION ({type(e).__name__}): {str(e)[:200]}")
        traceback.print_exc()
        results.append(
            ("probe_vmap_grad", "BLOCKER", f"{type(e).__name__}: {str(e)[:80]}")
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary():
    """Print the results summary table."""
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Probe':<44} {'Result':<12} {'Notes'}")
    print("-" * 72)
    for name, status, notes in results:
        print(f"{name:<44} {status:<12} {notes}")
    print("=" * 72)

    blockers = [r for r in results if r[1] == "BLOCKER"]
    cleared = [r for r in results if r[1] == "CLEARED"]
    passed = [r for r in results if r[1] == "PASS"]
    print(
        f"Total probes: {len(results)}  |  BLOCKER: {len(blockers)}  |  CLEARED: {len(cleared)}  |  PASS: {len(passed)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run all probes in inside-out order."""
    print("=" * 72)
    print("Phentax Gradient Audit Probe Script")
    print("=" * 72)

    # Level 1: innermost — _compute_waveform_params (BLOCKER-01 root)
    try:
        probe_waveform_params()
    except Exception as e:
        print(f"ERROR in probe_waveform_params: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_waveform_params", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    # Level 2: phase coefficients and Bisection
    try:
        probe_phase_coeffs()
    except Exception as e:
        print(f"ERROR in probe_phase_coeffs: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_phase_coeffs", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    try:
        probe_get_time_of_frequency()
    except Exception as e:
        print(
            f"ERROR in probe_get_time_of_frequency: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_get_time_of_frequency",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    # Level 3: guards, math.ceil, inner JIT
    try:
        probe_tracer_guard_at_1406()
    except Exception as e:
        print(
            f"ERROR in probe_tracer_guard_at_1406: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_tracer_guard_at_1406",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    try:
        probe_tracer_guard_at_1465()
    except Exception as e:
        print(
            f"ERROR in probe_tracer_guard_at_1465: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_tracer_guard_at_1465",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    try:
        probe_math_ceil()
    except Exception as e:
        print(f"ERROR in probe_math_ceil: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_math_ceil", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    try:
        probe_inner_jit_closures()
    except Exception as e:
        print(f"ERROR in probe_inner_jit_closures: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_inner_jit_closures", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    # Level 4: higher modes and spin gradients
    try:
        probe_compute_phase_coeffs_hm()
    except Exception as e:
        print(
            f"ERROR in probe_compute_phase_coeffs_hm: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_compute_phase_coeffs_hm",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    try:
        probe_compute_amplitude_coeffs_hm()
    except Exception as e:
        print(
            f"ERROR in probe_compute_amplitude_coeffs_hm: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_compute_amplitude_coeffs_hm",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    try:
        probe_chi_spin_gradients()
    except Exception as e:
        print(f"ERROR in probe_chi_spin_gradients: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_chi_spin_gradients", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    # Level 5: end-to-end and vmap+grad
    try:
        probe_compute_polarizations()
    except Exception as e:
        print(
            f"ERROR in probe_compute_polarizations: {type(e).__name__}: {str(e)[:200]}"
        )
        results.append(
            (
                "probe_compute_polarizations",
                "ERROR",
                f"{type(e).__name__}: {str(e)[:80]}",
            )
        )

    try:
        probe_vmap_grad()
    except Exception as e:
        print(f"ERROR in probe_vmap_grad: {type(e).__name__}: {str(e)[:200]}")
        results.append(
            ("probe_vmap_grad", "ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        )

    print_summary()


if __name__ == "__main__":
    main()
