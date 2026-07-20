"""Non-Clifford sampler correctness against exact analytic values.

H·Tᵏ·H on one qubit has P(1) = sin²(kπ/8) in closed form, so it pins the
stabilizer-rank path (the cat5 decomposition + ℤ[ω] term-family emission)
against a known answer rather than against another simulator. This is the
gate that distinguishes ksim from a Clifford-only sampler — stim cannot run
these circuits at all.

Self-contained: builds its circuits from literal stim text, so it needs only
ksim (+ the optional kokkos_sim / tsim / clifft backends), never a code
registry or a decoder.

Run:
    OMP_PROC_BIND=false pytest tests/test_nonclifford.py -v

Definitions
-----------
TestNonClifford
    H·Tᵏ·H circuits, exact P(1) = sin²(kπ/8).

TestNonClifford.test_p1_matches_exact
    kokkos_sim (and tsim, when installed) P(1) within 5σ shot noise of exact.

TestNonClifford.test_p1_clifft
    clifft Schrodinger-VM P(1) must match sin²(kπ/8) within 5σ shot noise.

TestNonClifford.test_amplitude_precision
    kokkos_sim exact ℤ[ω] arithmetic must match the float64 NumPy reference
    (evaluate_flat) to 1e-9 — far tighter than a complex64 runtime achieves.
"""
import numpy as np
import pytest

try:
    import tsim
    HAS_TSIM = hasattr(tsim, "Circuit")
except Exception:
    HAS_TSIM = False
    tsim = None

try:
    import clifft as _clifft
    HAS_CLIFFT = True
except Exception:
    HAS_CLIFFT = False
    _clifft = None

try:
    import ksim
    HAS_ZXC = hasattr(ksim, "compile")
except ImportError:
    HAS_ZXC = False

try:
    import kokkos_sim
    HAS_KTSIM = True
except ImportError:
    HAS_KTSIM = False

requires_clifft = pytest.mark.skipif(
    not HAS_CLIFFT, reason="clifft not installed")
requires_ktsim = pytest.mark.skipif(
    not (HAS_ZXC and HAS_KTSIM),
    reason="kokkos_sim extension not built or ksim/pyzx_param not installed")


@requires_ktsim
class TestNonClifford:

    KS = [1, 2, 3]
    N_SHOTS = 20_000

    def _compile(self, k):
        import ksim
        from ksim import KokkosProgramSampler
        text = "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n"
        flat, channel_probs, error_transform = ksim.compile(
            text, sample_detectors=True)
        return error_transform, flat, KokkosProgramSampler(flat)

    def test_p1_matches_exact(self):
        for k in self.KS:
            exact = float(np.sin(k * np.pi / 8) ** 2)
            sigma = np.sqrt(exact * (1 - exact) / self.N_SHOTS)
            error_transform, flat, ks = self._compile(k)

            n_f = (error_transform.shape[0] if error_transform.size else 0)
            f = np.zeros((self.N_SHOTS, n_f), dtype=np.uint8)
            p1_kk = float(ks.sample(f, seed=11)[:, 0].mean())
            assert abs(p1_kk - exact) < 5 * sigma, (
                f"k={k}: kokkos_sim P(1)={p1_kk:.4f} vs exact {exact:.4f}")

            if HAS_TSIM:
                text = "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n"
                s = tsim.Circuit(text).compile_detector_sampler(seed=42)
                p1_t = float(np.asarray(s.sample(self.N_SHOTS)).mean())
                assert abs(p1_t - exact) < 5 * sigma, (
                    f"k={k}: tsim P(1)={p1_t:.4f} vs exact {exact:.4f}")

    @requires_clifft
    def test_p1_clifft(self):
        for k in self.KS:
            exact = float(np.sin(k * np.pi / 8) ** 2)
            sigma = np.sqrt(exact * (1 - exact) / self.N_SHOTS)
            text = "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n"
            prog = _clifft.compile(text)
            result = _clifft.sample(prog, self.N_SHOTS, seed=42)
            p1_c = float(np.asarray(result.detectors).mean())
            assert abs(p1_c - exact) < 5 * sigma, (
                f"k={k}: clifft P(1)={p1_c:.4f} vs exact {exact:.4f}")

    def test_amplitude_precision(self):
        import kokkos_sim as kt
        from ksim import evaluate_flat, _graph_to_dict
        rng = np.random.default_rng(11)
        for k in self.KS:
            _, flat, _ = self._compile(k)
            for c_flat in flat.components:
                F = int(np.asarray(c_flat.f_selection).shape[0])
                kc = kt.Component([_graph_to_dict(g) for g in c_flat.graphs], F)
                for gi, g_flat in enumerate(c_flat.graphs):
                    P = g_flat.np_params.shape[2]
                    pv = rng.integers(0, 2, size=(8, max(P, 1)))[:, :P].astype(np.uint8)
                    ref = evaluate_flat(g_flat, pv)
                    kk = np.asarray(kc.eval_abs(gi, np.ascontiguousarray(pv)))
                    if ref.size:
                        err = float(np.max(np.abs(kk - np.abs(ref))))
                        assert err < 1e-9, (
                            f"k={k} graph {gi}: amplitude error {err:.2e}")
