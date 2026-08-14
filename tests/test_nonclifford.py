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

TestNonClifford.test_p1_xtim
    xtim P(1) must match sin²(kπ/8) within 5σ shot noise. Every T here sits on
    one wire with no Clifford between, so the folded π/8 layer is rank-1 and
    the circuit is inside xtim's simulable class.

TestNonClifford.test_amplitude_precision
    kokkos_sim exact ℤ[ω] arithmetic must match the float64 NumPy reference
    (evaluate_flat) to 1e-9 — far tighter than a complex64 runtime achieves.

TestSimulableClass.test_xtim_refuses_t_depth_2
    The other side of the T-depth-1 restriction: H·T·H·T·H folds to two
    anticommuting π/8 axes, so xtim refuses it while the stabilizer-rank path
    samples it against the closed-form P(1) = 1/4. Neither engine dominates —
    this is the axis no µs/shot number shows.
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
    import xtim as _xtim
    HAS_XTIM = True
except Exception:
    HAS_XTIM = False
    _xtim = None

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
requires_xtim = pytest.mark.skipif(
    not HAS_XTIM, reason="xtim not installed")
requires_ktsim = pytest.mark.skipif(
    not (HAS_ZXC and HAS_KTSIM),
    reason="kokkos_sim extension not built or ksim/pyzx_param not installed")


def _htkh(k):
    """H·Tᵏ·H on one qubit, measured in Z — exact P(1) = sin²(kπ/8)."""
    return "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n"


@requires_ktsim
class TestNonClifford:

    KS = [1, 2, 3]
    N_SHOTS = 20_000

    def _compile(self, k):
        import ksim
        from ksim import KokkosProgramSampler
        flat, channel_probs, error_transform = ksim.compile(
            _htkh(k), sample_detectors=True)
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
                s = tsim.Circuit(_htkh(k)).compile_detector_sampler(seed=42)
                p1_t = float(np.asarray(s.sample(self.N_SHOTS)).mean())
                assert abs(p1_t - exact) < 5 * sigma, (
                    f"k={k}: tsim P(1)={p1_t:.4f} vs exact {exact:.4f}")

    @requires_clifft
    def test_p1_clifft(self):
        for k in self.KS:
            exact = float(np.sin(k * np.pi / 8) ** 2)
            sigma = np.sqrt(exact * (1 - exact) / self.N_SHOTS)
            prog = _clifft.compile(_htkh(k))
            result = _clifft.sample(prog, self.N_SHOTS, seed=42)
            p1_c = float(np.asarray(result.detectors).mean())
            assert abs(p1_c - exact) < 5 * sigma, (
                f"k={k}: clifft P(1)={p1_c:.4f} vs exact {exact:.4f}")

    @requires_xtim
    def test_p1_xtim(self):
        for k in self.KS:
            exact = float(np.sin(k * np.pi / 8) ** 2)
            sigma = np.sqrt(exact * (1 - exact) / self.N_SHOTS)
            sampler = _xtim.Circuit(_htkh(k)).compile_detector_sampler(seed=42)
            p1_x = float(np.asarray(sampler.sample(self.N_SHOTS)).mean())
            assert abs(p1_x - exact) < 5 * sigma, (
                f"k={k}: xtim P(1)={p1_x:.4f} vs exact {exact:.4f}")

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


@requires_ktsim
class TestSimulableClass:
    """Where xtim's generalized-T-depth-1 class ends and ours continues."""

    # |0> -H-T-H-T-H- M.  The middle H rotates the second T's axis to X against
    # the first's Z, so the folded π/8 layer has two anticommuting axes.
    T_DEPTH_2 = "R 0\nH 0\nT 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n"
    EXACT_P1 = 0.25
    N_SHOTS = 20_000

    @requires_xtim
    def test_xtim_refuses_t_depth_2(self):
        # The refusal is lazy: parsing and compile_detector_sampler both
        # succeed, and the class check lands when the reference is actually
        # built — XtimReferenceError from compile_reference(), XtimRejectError
        # from sample(). Both are XtimError.
        circuit = _xtim.Circuit(self.T_DEPTH_2)
        with pytest.raises(_xtim.XtimError):
            circuit.compile_detector_sampler(seed=42).sample(64)

    def test_kokkos_samples_t_depth_2(self):
        import ksim
        flat, _, error_transform = ksim.compile(
            self.T_DEPTH_2, sample_detectors=True)
        ks = ksim.KokkosProgramSampler(flat)
        n_f = error_transform.shape[0] if error_transform.size else 0
        f = np.zeros((self.N_SHOTS, n_f), dtype=np.uint8)
        p1 = float(np.asarray(ks.sample(f, seed=11))[:, 0].mean())
        sigma = np.sqrt(self.EXACT_P1 * (1 - self.EXACT_P1) / self.N_SHOTS)
        assert abs(p1 - self.EXACT_P1) < 5 * sigma, (
            f"kokkos_sim P(1)={p1:.4f} vs exact {self.EXACT_P1}")
