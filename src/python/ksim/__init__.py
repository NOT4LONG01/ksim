"""ksim — ZX stabilizer-rank engine: symbolic **compile** + **sample**.

The whole non-Clifford simulation pipeline, on ``pyzx_param`` directly:

    stim.Circuit
      → ksim.compile(...)          # parse → double → reduce → e→f basis,
                                   #   components → plug → cat5 stabrank → emit
        → FlatProgram              # the flat numpy IR (program.py) — the contract
          → KokkosProgramSampler / sample_flat   # GPU / CPU sampling

Two subpackages joined by the ``program.py`` IR, with ``run.sample_circuit`` as
the one-call orchestrator (compile → noise → sample):

  compile/   stim circuit → FlatProgram
    graph / parse / instructions   stim → doubled, reduced ZX graph + noise
    stabrank                       cat5 stabilizer-rank recursion over pyzx_param
    pipeline                       components → plug → term-family emit → FlatProgram
    channels                       ChannelSampler + channel-prob builders
    program_text / types / _linalg shorthand→stim, SamplingGraph, GF(2) find_basis

  sample/    FlatProgram → outcomes
    evaluate      evaluate_flat / sample_flat (numpy reference, GPU-free)
    exact_scalar  exact ℤ[ω] arithmetic
    sampler       KokkosProgramSampler (kokkos_sim GPU runtime)

Some compile/ modules carry third-party attribution — see the ``NOTICE``
file at the repo root.

Definitions
-----------
compile
    One-call compile: stim circuit → ``(flat, channel_probs, error_transform)``.

    The returned ``flat`` is a ``FlatProgram`` ready for ``ksim.sample_flat`` /
    ``ksim.KokkosProgramSampler``; ``channel_probs`` and ``error_transform`` feed
    ``ksim.ChannelSampler`` to draw the noise ``f``-bits.

    ``circuit`` may be a ``stim.Circuit`` or stim program text; it is flattened
    here, so callers need not pre-flatten.

    Compiled results are reused automatically across calls whose circuits
    share the same shape (e.g. a noise-strength sweep at fixed topology) via
    the process-wide ``ksim.compile_cache`` — see ``ksim.compile.cache`` for
    why that's sound. Pass ``cache=None`` to opt out, or a fresh
    ``ksim.CompileCache()`` for an isolated one.
"""
from __future__ import annotations

import stim

from ksim.program import FlatProgram, FlatComponent, FlatScalarGraph
from ksim.program import flatten_program

from ksim.sample import evaluate_flat, sample_flat, KokkosProgramSampler, _graph_to_dict

from ksim.compile import (
    DecompositionMode, compile_program, prepare_graph,
    ChannelSampler, shorthand_to_stim, SamplingGraph,
    CompileCache, structural_signature, default_cache,
)

from ksim.run import sample_circuit

compile_cache = default_cache

__all__ = [
    "sample_circuit",
    "compile", "prepare_graph", "compile_program", "ChannelSampler",
    "SamplingGraph", "CompileCache", "structural_signature", "compile_cache",
    "FlatProgram", "FlatComponent", "FlatScalarGraph", "flatten_program",
    "evaluate_flat", "sample_flat", "KokkosProgramSampler", "_graph_to_dict",
]


def compile(
    circuit: "stim.Circuit | str",
    *,
    sample_detectors: bool = True,
    mode: DecompositionMode = "sequential",
    strategy: str = "cat5",
    cache: "CompileCache | None" = compile_cache,
) -> "tuple[FlatProgram, list, object]":
    if isinstance(circuit, str):
        circuit = stim.Circuit(shorthand_to_stim(circuit))
    circuit = circuit.flattened()
    if cache is not None:
        return cache.compile(circuit, sample_detectors=sample_detectors,
                              mode=mode, strategy=strategy)
    prepared = prepare_graph(circuit, sample_detectors=sample_detectors)
    flat = compile_program(prepared, mode=mode, strategy=strategy)
    return flat, prepared.channel_probs, prepared.error_transform
