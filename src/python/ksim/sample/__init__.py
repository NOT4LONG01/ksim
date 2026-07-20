"""ksim.sample — evaluate/sample a compiled ksim.FlatProgram.

``KokkosProgramSampler`` drives the kokkos_sim GPU kernel; ``evaluate_flat`` /
``sample_flat`` are the GPU-free numpy reference (built on the exact ℤ[ω]
``exact_scalar`` arithmetic). Both consume the ``FlatProgram`` IR that
``ksim.compile`` emits.
"""
from .evaluate import evaluate_flat, sample_flat
from .sampler import KokkosProgramSampler, _graph_to_dict

__all__ = ["evaluate_flat", "sample_flat", "KokkosProgramSampler", "_graph_to_dict"]
