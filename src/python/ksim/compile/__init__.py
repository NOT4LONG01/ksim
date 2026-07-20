"""ksim.compile — symbolic ZX compile: stim circuit → ksim.FlatProgram.

Parses a stim circuit into a doubled, reduced ZX graph (on ``pyzx_param``), runs
the cat5 stabilizer-rank decomposition, and emits the ``FlatProgram`` term-family
buffers as numpy. ``compile_program`` is the entry point; ``ksim.compile`` /
``ksim.sample_circuit`` wrap it. ``CompileCache`` lets callers that recompile
the same circuit shape repeatedly (e.g. a noise-strength sweep at fixed
topology) skip the ``full_reduce``/stabilizer-rank work after the first hit —
see ``ksim.compile.cache`` for why that's sound. Some modules carry
third-party attribution — see ``src/python/ksim/NOTICE``.
"""
from .pipeline import DecompositionMode, compile_program
from .graph import prepare_graph
from .channels import ChannelSampler
from .program_text import shorthand_to_stim
from .types import SamplingGraph
from .cache import CompileCache, structural_signature, default_cache

__all__ = [
    "compile_program", "DecompositionMode", "prepare_graph",
    "ChannelSampler", "shorthand_to_stim", "SamplingGraph",
    "CompileCache", "structural_signature", "default_cache",
]
