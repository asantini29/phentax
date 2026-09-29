---
phase: 02-gradient-fixes
plan: "02"
subsystem: waveform
tags: [gradient, jax, public-api, robustness]
dependency_graph:
  requires: []
  provides: [_is_tracing helper, public-API tracing detection in waveform.py]
  affects: [src/phentax/waveform.py]
tech_stack:
  added: []
  patterns: [try/except float() for public-API tracing detection]
key_files:
  created: []
  modified:
    - src/phentax/waveform.py
decisions:
  - "Use try/except float() approach rather than isinstance(x, jax.Array) because in modern JAX tracers also satisfy isinstance(x, jax.Array)"
  - "Place _is_tracing at module level (not as a method) so it can be imported directly for testing"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-04"
requirements: [GRAD-01, GRAD-02]
---

# Phase 02 Plan 02: Replace Private-API Tracer Guards Summary

Public-API tracing-detection helper `_is_tracing` added to `waveform.py`, replacing both `isinstance(jnp.atleast_1d(chi1z), jax.core.Tracer)` guards with a JAX-release-stable equivalent.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Replace private-API Tracer guards with `_is_tracing` helper | 7b9037a | src/phentax/waveform.py |
| 2 | Verify grad path and concrete validation (no edit) | — | — |

## What Was Built

Added a module-level helper `_is_tracing(x) -> bool` in `waveform.py` that uses only JAX's public error API:
- Inside a `try` block: `float(jnp.atleast_1d(x).reshape(-1)[0])` then `return False` — a concrete array converts successfully, meaning we are not tracing.
- `except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError): return True` — an abstract tracer raises on `float()` conversion, meaning we are tracing.

Both `isinstance(jnp.atleast_1d(chi1z), jax.core.Tracer)` occurrences at lines 1406 and 1465 were replaced with `if not _is_tracing(chi1z):`. Bodies of both guards are unchanged.

## Verification Results (Task 2)

**_is_tracing unit check:**
- Concrete `jnp.array([0.3])`: `_is_tracing` returns `False` (correct)
- Traced under `jax.jit(lambda x: _is_tracing(x))`: returns `True` (correct)
- Output: `PASS: _is_tracing concrete=False traced=True`

**Gradient probes (probe_gradients.py):**
- Guard at 1406: `jax.grad(sum(h_plus) wrt chi1z): -4.198249e-14  NaN=False` -- matches Phase 1 CLEARED-01 value exactly
- Guard at 1465: `jax.grad(sum(h_plus) wrt chi1z): -4.198249e-14  NaN=False` -- matches Phase 1 CLEARED-02 value exactly
- Both guards still correctly skip under tracing; no ConcretizationTypeError

**Concrete out-of-bounds spin validation:**
- `chi1z=1.5` on concrete call: `PASS: concrete validation fired: Spin must be between -1 and 1`
- Validation guard migration did not disable eager spin-bound checking

**Full test suite:**
- `uv run pytest tests/`: 1038 passed, 28 warnings -- zero failures

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None -- this change replaces a guard implementation with an equivalent, adds no new network endpoints, auth paths, or file access patterns.

## Self-Check: PASSED

- `src/phentax/waveform.py` modified (27 insertions, 2 deletions)
- `grep -c "jax.core.Tracer" src/phentax/waveform.py` = 0 (private API fully removed)
- `grep -c "_is_tracing" src/phentax/waveform.py` = 3 (1 definition + 2 call sites)
- Commit 7b9037a exists on `worktree-agent-aad737822a4d5faf4`
- pytest: 1038 passed
