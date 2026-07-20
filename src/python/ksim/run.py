"""ksim.run — one-call circuit sampling: compile → noise → sample → (det, obs).

`sample_circuit` bundles the whole non-Clifford pipeline so callers don't have
to wire `compile` + `ChannelSampler` + `KokkosProgramSampler`/`sample_flat`
themselves:

    det, obs = ksim.sample_circuit(stim_circuit, shots)

`det` is `(shots, num_detectors)` uint8, `obs` is `(shots, num_observables)`
uint8 (the output columns after the detectors).

Definitions
-----------
sample_circuit
    Compile `circuit`, draw `shots` noise realisations, and sample it.

    Args:
        circuit: a ``stim.Circuit`` or stim program text (flattened here).
        shots: number of shots to draw.
        seed: RNG seed for both the noise channel sampler and the sampler.
        sample_detectors: sample detector+observable outputs (vs. raw
            measurement records).
        backend: ``"gpu"`` (kokkos_sim), ``"cpu"`` (numpy reference
            ``sample_flat``), or ``"auto"`` (gpu if kokkos_sim imports, else cpu).
        strategy: stabilizer-rank decomposition strategy (``"cat5"``).
        cache: ``ksim.CompileCache`` to reuse the compiled result across
            calls whose circuits share the same shape (e.g. a noise-strength
            sweep at fixed topology); defaults to the process-wide
            ``ksim.compile_cache``. Pass ``None`` to opt out.

    Returns:
        ``(det, obs)`` — ``det`` is ``(shots, num_detectors)`` uint8, ``obs`` is
        ``(shots, num_observables)`` uint8.
"""
from __future__ import annotations

import numpy as np
import stim

from .compile import ChannelSampler, compile_program, prepare_graph, shorthand_to_stim
from .compile.cache import CompileCache, default_cache
from .sample import sample_flat


def sample_circuit(
    circuit: "stim.Circuit | str",
    shots: int,
    *,
    seed: int = 42,
    sample_detectors: bool = True,
    backend: str = "gpu",
    strategy: str = "cat5",
    cache: "CompileCache | None" = default_cache,
) -> "tuple[np.ndarray, np.ndarray]":
    if isinstance(circuit, str):
        circuit = stim.Circuit(shorthand_to_stim(circuit))
    circuit = circuit.flattened()

    if cache is not None:
        flat, channel_probs, error_transform = cache.compile(
            circuit, sample_detectors=sample_detectors, mode="sequential", strategy=strategy)
    else:
        prepared = prepare_graph(circuit, sample_detectors=sample_detectors)
        flat = compile_program(prepared, mode="sequential", strategy=strategy)
        channel_probs, error_transform = prepared.channel_probs, prepared.error_transform

    if channel_probs:
        f = ChannelSampler(channel_probs, error_transform,
                           seed=seed).sample(shots).astype(np.uint8)
    else:
        n_f = (error_transform.shape[0] if error_transform.size else 0)
        f = np.zeros((shots, n_f), dtype=np.uint8)

    out = _run_sampler(flat, f, seed, backend)
    nd = flat.num_detectors
    return out[:, :nd].astype(np.uint8), out[:, nd:].astype(np.uint8)


def _run_sampler(flat, f, seed, backend):
    if backend == "auto":
        try:
            import kokkos_sim
            backend = "gpu"
        except Exception:
            backend = "cpu"
    if backend == "gpu":
        from .sample import KokkosProgramSampler
        return np.asarray(KokkosProgramSampler(flat).sample(f, seed=seed))
    if backend == "cpu":
        return np.asarray(sample_flat(flat, f, np.random.default_rng(seed)))
    raise ValueError(f"unknown backend {backend!r} (expected gpu/cpu/auto)")
