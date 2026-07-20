"""ksim.compile.cache — structural cache for the pyzx compile pipeline.

`prepare_graph`'s reduction (`zx.full_reduce` + `transform_error_basis`) and
`compile_program`'s stabilizer-rank recursion never look at the *magnitude*
of a noise probability, only at whether it is exactly zero: every noise
builder in `instructions.py` (`depolarize1`, `x_error`, `pauli_channel_1`,
`correlated_error`, ...) unconditionally emits the same error vertices
regardless of the numeric value; the one zero/nonzero branch in the whole
pipeline (`_m`'s `if p > 0`) is exactly reproduced by collapsing every
noisy-gate argument to a single nonzero sentinel below. So two circuits that
differ only in noise strength (e.g. a `p`-sweep at fixed topology) compile to
byte-identical `FlatProgram`s — `CompileCache` lets a caller that samples the
same circuit *shape* many times (same basis/preselect/layout, varying `p`,
or repeated calls) skip re-running that pipeline after the first hit.
`channel_probs` (the actual per-instruction probabilities) still come from a
fresh parse every call, since that parse is the cheap part.

`default_cache` is a process-wide `CompileCache` instance (exposed as
`ksim.compile_cache`) that `ksim.compile`/`ksim.sample_circuit` use
automatically — callers don't need to construct or thread a cache through to
benefit; pass `cache=None` to opt out for a call, or a fresh `CompileCache()`
for an isolated one.

Definitions
-----------
structural_signature
    Canonical string key for a stim circuit: identical for two circuits that
    differ only in the magnitude (not the zero/nonzero-ness) of noisy-gate
    arguments.
CompileCache
    Dict-backed cache from `structural_signature` (+ sample_detectors/mode/
    strategy) to `(FlatProgram, error_transform)`. `.compile(circuit, ...)`
    returns `(flat, channel_probs, error_transform)`, matching `ksim.compile`.
default_cache
    The `CompileCache` instance `ksim.compile`/`ksim.sample_circuit` use by
    default.
"""
from __future__ import annotations

import stim
import pyzx_param as zx
from pyzx_param.graph.scalar import Scalar

from .graph import build_sampling_graph, squash_graph, transform_error_basis
from .parse import parse_stim_circuit
from .pipeline import DecompositionMode, compile_program
from .types import SamplingGraph

__all__ = ["structural_signature", "CompileCache", "default_cache"]


def _canonical_args(name: str, args: tuple[float, ...]) -> tuple[float, ...]:
    try:
        is_noisy = stim.gate_data(name).is_noisy_gate
    except ValueError:
        is_noisy = False
    if not is_noisy:
        return args
    return tuple(0.0 if a == 0 else 1.0 for a in args)


def structural_signature(circuit: "stim.Circuit") -> str:
    lines = []
    for instr in circuit.flattened():
        args = _canonical_args(instr.name, tuple(instr.gate_args_copy()))
        targets = tuple(str(t) for t in instr.targets_copy())
        lines.append(f"{instr.name}|{instr.tag}|{args}|{targets}")
    return "\n".join(lines)


class CompileCache:
    """Reuses the pyzx compile pipeline across calls with the same circuit shape."""

    def __init__(self) -> None:
        self._store: dict[tuple, tuple] = {}

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def compile(
        self,
        circuit: "stim.Circuit",
        *,
        sample_detectors: bool = True,
        mode: DecompositionMode = "sequential",
        strategy: str = "cat5",
    ) -> "tuple":
        built = parse_stim_circuit(circuit)
        key = (structural_signature(circuit), sample_detectors, mode, strategy)

        cached = self._store.get(key)
        if cached is None:
            graph = build_sampling_graph(built, sample_detectors=sample_detectors)
            zx.full_reduce(graph, paramSafe=True)
            squash_graph(graph)
            graph, error_transform = transform_error_basis(graph, num_e=built.num_error_bits)
            graph.scalar = Scalar()
            prepared = SamplingGraph(
                graph=graph, error_transform=error_transform, channel_probs=[],
                num_outputs=len(graph.outputs()), num_detectors=len(built.detectors),
            )
            flat = compile_program(prepared, mode=mode, strategy=strategy)
            cached = (flat, error_transform)
            self._store[key] = cached

        flat, error_transform = cached
        return flat, built.channel_probs, error_transform


default_cache = CompileCache()
