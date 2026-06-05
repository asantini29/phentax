"""
Permanent pytest-collected gradient correctness suite.

Proves that jax.grad / jax.jacfwd gradients match central finite-difference
estimates for all applicable (param, method) pairs across LIGO-like and
LISA-like mass regimes.

Covers TEST-01 through TEST-04 (Phase 3).

Run non-slow tests with:
    uv run pytest tests/test_gradient_correctness.py -m "not slow"

Run LISA (slow) tests with:
    uv run pytest tests/test_gradient_correctness.py -m slow
"""

from phentax.utils.config import configure_jax

configure_jax(platform="cpu", enable_x64=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from phentax.waveform import IMRPhenomTHM  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LIGO_PARAMS = {
    "m1": jnp.array(30.0),
    "m2": jnp.array(20.0),
    "chi1z": jnp.array(0.3),
    "chi2z": jnp.array(-0.1),
    "distance": jnp.array(100.0),
    "phi_ref": jnp.array(1.0),
    "inclination": jnp.array(1.0),
    "psi": jnp.array(0.5),
    "delta_t": 1.0 / 4096.0,
    "T": 10.0,
    "f_min": 20.0,
    "f_ref": 20.0,
}

LISA_PARAMS = {
    "m1": jnp.array(1e6),
    "m2": jnp.array(5e5),
    "chi1z": jnp.array(0.2),
    "chi2z": jnp.array(-0.1),
    "distance": jnp.array(1000.0),
    "phi_ref": jnp.array(1.0),
    "inclination": jnp.array(1.0),
    "psi": jnp.array(0.5),
    "delta_t": 2.5,
    "T": None,
    "f_min": 1e-4,
    "f_ref": 1e-4,
}

# ---------------------------------------------------------------------------
# Tolerance constants
# rtol_chi from 03-01-SUMMARY.md probe_chi_tolerance.py findings
# ---------------------------------------------------------------------------

_RTOL_CHI = 1e-4  # rtol_chi from 03-01-SUMMARY.md probe_chi_tolerance.py findings

RTOL_BY_PARAM = {
    "m1": 1e-4,
    "m2": 1e-4,
    "chi1z": _RTOL_CHI,
    "chi2z": _RTOL_CHI,
    "distance": 1e-4,
    "phi_ref": 1e-4,
    "inclination": 1e-4,
    "psi": 1e-4,
}

# ---------------------------------------------------------------------------
# Parameter applicability matrix (per D-05)
# Only applicable params per method — no vacuous zero-gradient tests.
# ---------------------------------------------------------------------------

AMP_PHASE_PARAMS = ["m1", "m2", "chi1z", "chi2z"]
# distance/phi_ref/inclination/psi EXCLUDED (not applicable to compute_amp_phase)

HLMS_PARAMS = ["m1", "m2", "chi1z", "chi2z", "distance"]
# phi_ref/inclination/psi EXCLUDED (not applicable to compute_hlms)

STRAIN_PARAMS = ["m1", "m2", "chi1z", "chi2z", "distance", "phi_ref", "inclination"]
# psi EXCLUDED (only h+/hx projection in compute_polarizations)

POLAR_PARAMS = [
    "m1",
    "m2",
    "chi1z",
    "chi2z",
    "distance",
    "phi_ref",
    "inclination",
    "psi",
]
# all 8 parameters applicable to compute_polarizations

# LISA test: 3 representative params for compute_polarizations
LISA_POLAR_PARAMS = ["m1", "distance", "chi1z"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_mid_idx(model, params):
    """Return a mask-interior time index stable across small parameter perturbations."""
    _, _, mask, _, _ = model.compute_amp_phase(
        params["m1"],
        params["m2"],
        params["chi1z"],
        params["chi2z"],
        params["distance"],
        params["phi_ref"],
        params["inclination"],
        params["psi"],
        delta_t=params["delta_t"],
        T=params["T"],
        f_min=params.get("f_min", 1e-4),
        f_ref=params.get("f_ref", 1e-4),
    )
    true_idx = jnp.where(mask[0])[0]
    n = len(true_idx)
    return int(true_idx[n // 2])


def _central_fd(fn, x, h):
    """Central finite-difference gradient for scalar x."""
    x_arr = jnp.array(float(x))
    h_arr = jnp.array(float(h))
    return float((fn(x_arr + h_arr) - fn(x_arr - h_arr)) / (2.0 * h_arr))


def _step_for(x):
    """Parameter-appropriate finite-difference step using a relative formula.

    Uses h = 1e-5 * max(|x|, 1.0), following the CONTEXT.md recommendation
    (h = 1e-5 * max(|x|, 1e-3)). This is scale-appropriate across both LIGO
    (m1~30) and LISA (m1~1e6) regimes: fixed steps (e.g. 0.3 M_sun) are too
    coarse for oscillating reductions (h_plus, h_lm.real) at the LIGO point,
    and too small (relative) for LISA mass scales.
    """
    return 1e-5 * max(abs(float(x)), 1.0)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_name", POLAR_PARAMS)
def test_grad_compute_polarizations(param_name):
    """AD gradient wrt param_name through compute_polarizations at LIGO point.

    Checks: finiteness (D-06) and correctness vs central FD (TEST-01).
    All 8 parameters are applicable to compute_polarizations.
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model, LIGO_PARAMS)

    def fn(val):
        kw = {**LIGO_PARAMS, param_name: val}
        _, _, h_plus, _ = model.compute_polarizations(
            kw["m1"],
            kw["m2"],
            kw["chi1z"],
            kw["chi2z"],
            kw["distance"],
            kw["phi_ref"],
            kw["inclination"],
            kw["psi"],
            delta_t=kw["delta_t"],
            T=kw["T"],
            f_min=kw["f_min"],
            f_ref=kw["f_ref"],
        )
        return h_plus[0, mid_idx]

    g_ad = jax.grad(fn)(LIGO_PARAMS[param_name])
    assert jnp.isfinite(g_ad), f"gradient wrt {param_name} is not finite: {g_ad}"

    h_step = _step_for(LIGO_PARAMS[param_name])
    fd = _central_fd(fn, LIGO_PARAMS[param_name], h_step)
    np.testing.assert_allclose(
        float(g_ad),
        fd,
        rtol=RTOL_BY_PARAM[param_name],
        err_msg=f"compute_polarizations grad wrt {param_name} mismatch at LIGO point",
    )


@pytest.mark.slow
@pytest.mark.parametrize("param_name", LISA_POLAR_PARAMS)
def test_grad_compute_polarizations_lisa(param_name):
    """AD gradient wrt param_name through compute_polarizations at LISA point.

    Marked @pytest.mark.slow — LISA waveforms take ~50s per grad call.
    3 representative params (m1, distance, chi1z) for mass-regime coverage.
    Mandatory execution gate enforced in 03-03.
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model, LISA_PARAMS)

    def fn(val):
        kw = {**LISA_PARAMS, param_name: val}
        _, _, h_plus, _ = model.compute_polarizations(
            kw["m1"],
            kw["m2"],
            kw["chi1z"],
            kw["chi2z"],
            kw["distance"],
            kw["phi_ref"],
            kw["inclination"],
            kw["psi"],
            delta_t=kw["delta_t"],
            T=kw["T"],
            f_min=kw["f_min"],
            f_ref=kw["f_ref"],
        )
        return h_plus[0, mid_idx]

    g_ad = jax.grad(fn)(LISA_PARAMS[param_name])
    assert jnp.isfinite(g_ad), f"gradient wrt {param_name} is not finite: {g_ad}"

    h_step = _step_for(LISA_PARAMS[param_name])
    fd = _central_fd(fn, LISA_PARAMS[param_name], h_step)
    np.testing.assert_allclose(
        float(g_ad),
        fd,
        rtol=RTOL_BY_PARAM[param_name],
        err_msg=f"compute_polarizations grad wrt {param_name} mismatch at LISA point",
    )


@pytest.mark.parametrize("param_name", AMP_PHASE_PARAMS)
def test_grad_compute_amp_phase(param_name):
    """AD gradient wrt param_name through compute_amp_phase at LIGO point.

    Checks: finiteness (D-06) and correctness vs central FD (TEST-01).
    Applicable params: m1, m2, chi1z, chi2z only
    (distance/phi_ref/inclination/psi are not applicable — enter later stages).
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model, LIGO_PARAMS)

    def fn(val):
        kw = {**LIGO_PARAMS, param_name: val}
        _, _, mask, amplitudes, _ = model.compute_amp_phase(
            kw["m1"],
            kw["m2"],
            kw["chi1z"],
            kw["chi2z"],
            kw["distance"],
            kw["phi_ref"],
            kw["inclination"],
            kw["psi"],
            delta_t=kw["delta_t"],
            T=kw["T"],
            f_min=kw["f_min"],
            f_ref=kw["f_ref"],
        )
        return amplitudes[0, 0, mid_idx]

    g_ad = jax.grad(fn)(LIGO_PARAMS[param_name])
    assert jnp.isfinite(g_ad), f"gradient wrt {param_name} is not finite: {g_ad}"

    h_step = _step_for(LIGO_PARAMS[param_name])
    fd = _central_fd(fn, LIGO_PARAMS[param_name], h_step)
    np.testing.assert_allclose(
        float(g_ad),
        fd,
        rtol=RTOL_BY_PARAM[param_name],
        err_msg=f"compute_amp_phase grad wrt {param_name} mismatch at LIGO point",
    )


@pytest.mark.parametrize("param_name", HLMS_PARAMS)
def test_grad_compute_hlms(param_name):
    """AD gradient wrt param_name through compute_hlms at LIGO point.

    Checks: finiteness (D-06) and correctness vs central FD (TEST-01).
    Applicable params: m1, m2, chi1z, chi2z, distance
    (phi_ref/inclination/psi not applicable — enter at Ylm/polarization stage).
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model, LIGO_PARAMS)

    def fn(val):
        kw = {**LIGO_PARAMS, param_name: val}
        _, _, h_lms = model.compute_hlms(
            kw["m1"],
            kw["m2"],
            kw["chi1z"],
            kw["chi2z"],
            kw["distance"],
            kw["phi_ref"],
            kw["inclination"],
            kw["psi"],
            delta_t=kw["delta_t"],
            T=kw["T"],
            f_min=kw["f_min"],
            f_ref=kw["f_ref"],
        )
        return h_lms[0, 0, mid_idx].real

    g_ad = jax.grad(fn)(LIGO_PARAMS[param_name])
    assert jnp.isfinite(g_ad), f"gradient wrt {param_name} is not finite: {g_ad}"

    h_step = _step_for(LIGO_PARAMS[param_name])
    fd = _central_fd(fn, LIGO_PARAMS[param_name], h_step)
    np.testing.assert_allclose(
        float(g_ad),
        fd,
        rtol=RTOL_BY_PARAM[param_name],
        err_msg=f"compute_hlms grad wrt {param_name} mismatch at LIGO point",
    )


@pytest.mark.parametrize("param_name", STRAIN_PARAMS)
def test_grad_compute_strain_components(param_name):
    """AD gradient wrt param_name through compute_strain_components at LIGO point.

    Checks: finiteness (D-06) and correctness vs central FD (TEST-01).
    Applicable params: m1, m2, chi1z, chi2z, distance, phi_ref, inclination
    (psi not applicable — only enters at the h+/hx projection in compute_polarizations).
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)
    mid_idx = _get_mid_idx(model, LIGO_PARAMS)

    def fn(val):
        kw = {**LIGO_PARAMS, param_name: val}
        _, _, strain_components = model.compute_strain_components(
            kw["m1"],
            kw["m2"],
            kw["chi1z"],
            kw["chi2z"],
            kw["distance"],
            kw["phi_ref"],
            kw["inclination"],
            kw["psi"],
            delta_t=kw["delta_t"],
            T=kw["T"],
            f_min=kw["f_min"],
            f_ref=kw["f_ref"],
        )
        return strain_components[0, 0, mid_idx].real

    g_ad = jax.grad(fn)(LIGO_PARAMS[param_name])
    assert jnp.isfinite(g_ad), f"gradient wrt {param_name} is not finite: {g_ad}"

    h_step = _step_for(LIGO_PARAMS[param_name])
    fd = _central_fd(fn, LIGO_PARAMS[param_name], h_step)
    np.testing.assert_allclose(
        float(g_ad),
        fd,
        rtol=RTOL_BY_PARAM[param_name],
        err_msg=f"compute_strain_components grad wrt {param_name} mismatch at LIGO point",
    )


def test_grad_vmap_composition():
    """vmap(grad wrt distance) over batch=4 equals per-sample grads to rtol=1e-10.

    TEST-04: Verifies that jax.vmap and jax.grad compose correctly.
    Uses distance gradient (mask-stable extrinsic param) with short grid for speed.
    Reduction: jnp.sum(jnp.where(mask, h_plus**2, 0.0)) — mask-stable.
    """
    model = IMRPhenomTHM(higher_modes=None, include_negative_modes=False)

    m1_b = jnp.array([80.0, 60.0, 40.0, 30.0])
    m2_b = jnp.array([20.0, 30.0, 20.0, 10.0])
    c1_b = jnp.array([0.5, 0.3, 0.0, 0.6])
    c2_b = jnp.array([-0.3, 0.1, -0.1, 0.4])
    d_b = jnp.array([1.0, 1.5, 2.0, 0.8])
    phi_b = jnp.array([0.0, 0.1, 0.2, 0.3])
    inc_b = jnp.array([1.0, 0.8, 0.5, 1.2])
    psi_b = jnp.array([0.0, 0.2, 0.4, 0.1])

    def single_grad(m1, m2, c1, c2, d, phi, inc, psi):
        def fn(d_):
            _, mask, h_plus, _ = model.compute_polarizations(
                m1, m2, c1, c2, d_, phi, inc, psi, delta_t=15.0, T=128.0
            )
            return jnp.sum(jnp.where(mask, h_plus**2, 0.0))

        return jax.grad(fn)(d)

    g_batch = jax.vmap(single_grad)(m1_b, m2_b, c1_b, c2_b, d_b, phi_b, inc_b, psi_b)
    g_per = jnp.array(
        [
            single_grad(
                m1_b[i], m2_b[i], c1_b[i], c2_b[i], d_b[i], phi_b[i], inc_b[i], psi_b[i]
            )
            for i in range(4)
        ]
    )

    assert jnp.isfinite(
        g_batch
    ).all(), f"vmap grad contains non-finite values: {g_batch}"
    np.testing.assert_allclose(
        g_batch,
        g_per,
        rtol=1e-10,
        err_msg="vmap(grad) result differs from per-sample grad",
    )
