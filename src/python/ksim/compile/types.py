"""Core data type for the ksim graph-preparation phase.

Vendored from bloqade-tsim's ``tsim.core.types`` (Apache-2.0), trimmed to the
single JAX-free dataclass ksim needs: ``SamplingGraph``. tsim's
``CompiledComponent``/``CompiledProgram`` (equinox/JAX) are intentionally *not*
ported — ksim emits ``ksim.FlatProgram`` directly (see ``ksim.compile``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyzx_param.graph.base import BaseGraph


@dataclass(frozen=True)
class SamplingGraph:
    """Result of the graph preparation phase for sampling.

    The circuit has been parsed from stim, converted to a ZX graph, doubled
    (composed with adjoint), reduced via ``zx.full_reduce``, and had its error
    basis transformed (Gaussian elimination: e → f).

    Attributes:
        graph: The prepared ZX graph with f-parameters on vertices.
        error_transform: Binary matrix of shape (num_f, num_e) where entry
            [i, j] = 1 means f_i = XOR of e_j over the set bits.
        channel_probs: List of probability arrays for error channels.
        num_outputs: Number of output vertices (measurements or detectors).
        num_detectors: Number of detector vertices.
    """

    graph: "BaseGraph"
    error_transform: np.ndarray
    channel_probs: list[np.ndarray]
    num_outputs: int
    num_detectors: int
