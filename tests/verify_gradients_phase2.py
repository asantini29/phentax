"""
Phase 2 gradient finiteness verification script.

Standalone diagnostic — NOT collected by pytest.
Run with: uv run python tests/verify_gradients_phase2.py

Confirms GRAD-01 through GRAD-06 at the Phase 2 "finite, non-NaN" bar
after the BLOCKER-01 (02-01) and D-07 (02-02) fixes.

Correctness vs finite-differences is Phase 3's job (TEST-01).
This script only checks finiteness.
"""

import sys

import jax
import jax.numpy as jnp

from phentax.utils.config import configure_jax

configure_jax(platform="cpu", enable_x64=True)

from phentax.waveform import IMRPhenomTHM  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test parameters — match probe_gradients.py convention
# ---------------------------------------------------------------------------
M1 = jnp.array(80.0)
M2 = jnp.array(20.0)
S1Z = jnp.array(0.5)
S2Z = jnp.array(-0.3)
DISTANCE = jnp.array(1.0)
INCLINATION = jnp.array(1.0)
PHI_REF = jnp.array(0.0)
PSI = jnp.array(0.0)

DELTA_T = 15.0
T_OBS = 128.0

_fail_count = 0


def _check(name: str, value: float, *, nan: bool, inf: bool) -> None:
    global _fail_count
    finite = not nan and not inf
    status = "PASS" if finite else "FAIL"
    if not finite:
        _fail_count += 1
    print(f"{status}: {name}  val={value:.6e}  NaN={nan}  Inf={inf}")


# ---------------------------------------------------------------------------
# GRAD-01 — jax.grad wrt each of 8 physical params
# ---------------------------------------------------------------------------


def verify_grad_polarizations_all_params():
    print()
    print("=== verify_grad_polarizations_all_params (GRAD-01) ===")
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    def masked_sum(m1, m2, chi1z, chi2z, distance, phi_ref, inclination, psi):
        _, mask, h_plus, _ = model.compute_polarizations(
            m1,
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

    params = [
        (
            "m1",
            lambda: jax.grad(
                lambda x: masked_sum(
                    x, M2, S1Z, S2Z, DISTANCE, PHI_REF, INCLINATION, PSI
                )
            )(M1),
        ),
        (
            "m2",
            lambda: jax.grad(
                lambda x: masked_sum(
                    M1, x, S1Z, S2Z, DISTANCE, PHI_REF, INCLINATION, PSI
                )
            )(M2),
        ),
        (
            "chi1z",
            lambda: jax.grad(
                lambda x: masked_sum(
                    M1, M2, x, S2Z, DISTANCE, PHI_REF, INCLINATION, PSI
                )
            )(S1Z),
        ),
        (
            "chi2z",
            lambda: jax.grad(
                lambda x: masked_sum(
                    M1, M2, S1Z, x, DISTANCE, PHI_REF, INCLINATION, PSI
                )
            )(S2Z),
        ),
        (
            "distance",
            lambda: jax.grad(
                lambda x: masked_sum(M1, M2, S1Z, S2Z, x, PHI_REF, INCLINATION, PSI)
            )(DISTANCE),
        ),
        (
            "phi_ref",
            lambda: jax.grad(
                lambda x: masked_sum(M1, M2, S1Z, S2Z, DISTANCE, x, INCLINATION, PSI)
            )(PHI_REF),
        ),
        (
            "inclination",
            lambda: jax.grad(
                lambda x: masked_sum(M1, M2, S1Z, S2Z, DISTANCE, PHI_REF, x, PSI)
            )(INCLINATION),
        ),
        (
            "psi",
            lambda: jax.grad(
                lambda x: masked_sum(
                    M1, M2, S1Z, S2Z, DISTANCE, PHI_REF, INCLINATION, x
                )
            )(PSI),
        ),
    ]
    for name, fn in params:
        try:
            g = fn()
            v = float(g)
            _check(
                f"GRAD-01[{name}]", v, nan=bool(jnp.isnan(g)), inf=bool(jnp.isinf(g))
            )
        except Exception as e:
            global _fail_count
            _fail_count += 1
            print(f"FAIL: GRAD-01[{name}]  exception={type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# GRAD-02 — jax.jacfwd wrt all 8 params (packed vector)
# ---------------------------------------------------------------------------


def verify_jacfwd_polarizations():
    print()
    print("=== verify_jacfwd_polarizations (GRAD-02) ===")
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    # Pack 8 params into a single vector; unpack inside fn
    theta0 = jnp.array(
        [
            float(M1),
            float(M2),
            float(S1Z),
            float(S2Z),
            float(DISTANCE),
            float(PHI_REF),
            float(INCLINATION),
            float(PSI),
        ]
    )

    def scalar_fn(theta):
        m1, m2, c1, c2, dist, phi, inc, psi = (theta[i] for i in range(8))
        _, mask, h_plus, _ = model.compute_polarizations(
            m1,
            m2,
            c1,
            c2,
            dist,
            phi,
            inc,
            psi,
            delta_t=DELTA_T,
            T=T_OBS,
        )
        return jnp.sum(jnp.where(mask, h_plus, 0.0))

    try:
        jac = jax.jacfwd(scalar_fn)(theta0)
        has_nan = bool(jnp.isnan(jac).any())
        has_inf = bool(jnp.isinf(jac).any())
        print(f"  Jacobian shape: {jac.shape}  values: {jac}")
        _check(
            "GRAD-02[jacfwd all 8 params]",
            float(jnp.linalg.norm(jac)),
            nan=has_nan,
            inf=has_inf,
        )
    except Exception as e:
        global _fail_count
        _fail_count += 1
        print(f"FAIL: GRAD-02[jacfwd]  exception={type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# GRAD-03/04/05 — jax.grad through compute_amp_phase, compute_hlms, compute_strain_components
# ---------------------------------------------------------------------------


def verify_grad_through_methods():
    global _fail_count
    print()
    print("=== verify_grad_through_methods (GRAD-03, GRAD-04, GRAD-05) ===")
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    # GRAD-03: compute_amp_phase
    try:

        def fn_amp_phase(m1):
            _, _, mask, amplitudes, _ = model.compute_amp_phase(
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
            return jnp.sum(jnp.where(mask[:, None, :], amplitudes, 0.0))

        g = jax.grad(fn_amp_phase)(M1)
        v = float(g)
        _check(
            "GRAD-03[compute_amp_phase wrt m1]",
            v,
            nan=bool(jnp.isnan(g)),
            inf=bool(jnp.isinf(g)),
        )
    except Exception as e:
        global _fail_count
        _fail_count += 1
        print(
            f"FAIL: GRAD-03[compute_amp_phase]  exception={type(e).__name__}: {str(e)[:80]}"
        )

    # GRAD-04: compute_hlms — reduce via sum of real part
    try:

        def fn_hlms(m1):
            _, mask, h_lms = model.compute_hlms(
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
            return jnp.sum(jnp.where(mask[:, None, :], h_lms.real, 0.0))

        g = jax.grad(fn_hlms)(M1)
        v = float(g)
        _check(
            "GRAD-04[compute_hlms wrt m1]",
            v,
            nan=bool(jnp.isnan(g)),
            inf=bool(jnp.isinf(g)),
        )
    except Exception as e:
        _fail_count += 1
        print(
            f"FAIL: GRAD-04[compute_hlms]  exception={type(e).__name__}: {str(e)[:80]}"
        )

    # GRAD-05: compute_strain_components — reduce via sum of real part
    try:

        def fn_strain(m1):
            _, mask, strain = model.compute_strain_components(
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
            return jnp.sum(jnp.where(mask[:, None, :], strain.real, 0.0))

        g = jax.grad(fn_strain)(M1)
        v = float(g)
        _check(
            "GRAD-05[compute_strain_components wrt m1]",
            v,
            nan=bool(jnp.isnan(g)),
            inf=bool(jnp.isinf(g)),
        )
    except Exception as e:
        _fail_count += 1
        print(
            f"FAIL: GRAD-05[compute_strain_components]  exception={type(e).__name__}: {str(e)[:80]}"
        )


# ---------------------------------------------------------------------------
# GRAD-06 — vmap over batch of 4 composed with jax.grad
# ---------------------------------------------------------------------------


def verify_vmap_grad():
    global _fail_count
    print()
    print("=== verify_vmap_grad (GRAD-06) ===")
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    m1_batch = jnp.array([80.0, 60.0, 40.0, 30.0])
    m2_batch = jnp.array([20.0, 30.0, 20.0, 10.0])
    c1_batch = jnp.array([0.5, 0.3, 0.0, 0.6])
    c2_batch = jnp.array([-0.3, 0.1, -0.1, 0.4])
    d_batch = jnp.array([1.0, 1.5, 2.0, 0.8])
    phi_batch = jnp.array([0.0, 0.1, 0.2, 0.3])
    inc_batch = jnp.array([1.0, 0.8, 0.5, 1.2])
    psi_batch = jnp.array([0.0, 0.2, 0.4, 0.1])

    try:

        def single_grad(m1, m2, chi1z, chi2z, distance, phi_ref, inclination, psi):
            def fn(m1_):
                _, mask, h_plus, _ = model.compute_polarizations(
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
            c1_batch,
            c2_batch,
            d_batch,
            phi_batch,
            inc_batch,
            psi_batch,
        )
        has_nan = bool(jnp.isnan(g_batch).any())
        has_inf = bool(jnp.isinf(g_batch).any())
        print(f"  vmap(grad) shape={g_batch.shape}, values={g_batch}")
        _check(
            "GRAD-06[vmap+grad batch=4 shape=(4,)]",
            float(jnp.linalg.norm(g_batch)),
            nan=has_nan,
            inf=has_inf,
        )
    except Exception as e:
        _fail_count += 1
        print(f"FAIL: GRAD-06[vmap+grad]  exception={type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# WARN-01 — chi1z/chi2z finiteness at spin-sensitive point
# ---------------------------------------------------------------------------


def verify_warn01_chi_finiteness():
    global _fail_count
    print()
    print("=== verify_warn01_chi_finiteness (WARN-01) ===")
    print("  (finiteness only — correctness deferred to Phase 3 (TEST-01))")
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    # Spin-sensitive point: higher spins, asymmetric mass ratio
    m1_spin = jnp.array(60.0)
    m2_spin = jnp.array(10.0)
    chi1z_spin = jnp.array(0.8)
    chi2z_spin = jnp.array(0.6)

    def masked_sum_spin(chi1z, chi2z):
        _, mask, h_plus, _ = model.compute_polarizations(
            m1_spin,
            m2_spin,
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

    for name, fn, x0 in [
        ("chi1z", lambda x: masked_sum_spin(x, chi2z_spin), chi1z_spin),
        ("chi2z", lambda x: masked_sum_spin(chi1z_spin, x), chi2z_spin),
    ]:
        try:
            g = jax.grad(fn)(x0)
            v = float(g)
            is_nan = bool(jnp.isnan(g))
            is_inf = bool(jnp.isinf(g))
            finite = not is_nan and not is_inf
            status = "PASS" if finite else "FAIL"
            if not finite:
                _fail_count += 1
            print(
                f"{status}: WARN-01[{name}]  val={v:.6e}  NaN={is_nan}  Inf={is_inf}"
                f"  — correctness deferred to Phase 3 (TEST-01)"
            )
        except Exception as e:
            _fail_count += 1
            print(f"FAIL: WARN-01[{name}]  exception={type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Phase 2 Gradient Finiteness Verification")
    print("GRAD-01 through GRAD-06  |  WARN-01 chi finiteness")
    print("=" * 60)

    verify_grad_polarizations_all_params()
    verify_jacfwd_polarizations()
    verify_grad_through_methods()
    verify_vmap_grad()
    verify_warn01_chi_finiteness()

    print()
    print("=" * 60)
    if _fail_count == 0:
        print(f"RESULT: ALL PASS  (0 failures)")
    else:
        print(f"RESULT: {_fail_count} FAIL(s) detected")
    print("=" * 60)

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
