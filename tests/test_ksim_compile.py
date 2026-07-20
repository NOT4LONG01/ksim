"""Regression tests for the in-repo ZX compiler (`ksim.compile`).

Self-contained — no fixtures, no GPU/kokkos_sim, no bloqade-tsim. They compile
small circuits with `ksim.compile` and check the result against known-exact
analytic values via the CPU numpy reference sampler (`ksim.sample_flat`):

  * H·Tᵏ·H single-qubit P(1) = sin²(kπ/8) — exercises the non-Clifford cat5
    stabilizer-rank path and the ℤ[ω] term-family emission.
  * a small Clifford circuit compiles + samples end-to-end on CPU.

Definitions
-----------
test_single_t_pofone_exact
    H·T^k·H single-qubit P(1) = sin^2(k·pi/8), via the numpy reference sampler.

test_small_clifford_compiles_and_samples
    A small Clifford circuit compiles to a FlatProgram and samples on CPU.

test_compile_cache_reuses_across_noise_magnitude
    `CompileCache` compiles once for two circuits differing only in noise
    strength, and separately for a structurally different circuit.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import ksim
    from ksim import sample_flat
    HAS_KSIM = hasattr(ksim, "compile")
except Exception:
    HAS_KSIM = False

requires_ksim = pytest.mark.skipif(not HAS_KSIM, reason="ksim/pyzx_param not installed")


@requires_ksim
@pytest.mark.parametrize("k", [1, 2, 3])
def test_single_t_pofone_exact(k):
    flat, _channel_probs, _error_transform = ksim.compile(
        "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n",
        sample_detectors=True)
    shots = 200_000
    f = np.zeros((shots, 0), dtype=np.uint8)
    out = sample_flat(flat, f, np.random.default_rng(0))
    p1 = out[:, :flat.num_detectors].mean()
    exact = np.sin(k * np.pi / 8) ** 2
    assert abs(p1 - exact) < 0.01, f"k={k}: P(1)={p1:.4f} vs exact {exact:.4f}"


@requires_ksim
def test_small_clifford_compiles_and_samples():
    text = (
        "R 0 1 2\nH 0\nCNOT 0 1\nCNOT 1 2\nX_ERROR(0.01) 0 1 2\nTICK\n"
        "M 0 1 2\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    det, obs = ksim.sample_circuit(text, 4000, seed=0, backend="cpu")
    assert det.shape == (4000, 2)
    assert obs.shape == (4000, 1)
    assert det.mean() < 0.1


@requires_ksim
def test_compile_cache_reuses_across_noise_magnitude():
    cache = ksim.CompileCache()
    text = "R 0\nH 0\nX_ERROR({p}) 0\nM 0\nDETECTOR rec[-1]\n"

    flat_a, channel_probs_a, _ = ksim.compile(text.format(p=0.01), cache=cache)
    assert len(cache) == 1
    flat_b, channel_probs_b, _ = ksim.compile(text.format(p=0.2), cache=cache)
    assert len(cache) == 1
    assert flat_a is flat_b
    assert channel_probs_a[0][1] != channel_probs_b[0][1]

    flat_c, _, _ = ksim.compile("R 0\nH 0\nH 0\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]\n",
                                cache=cache)
    assert len(cache) == 2
    assert flat_c is not flat_a
