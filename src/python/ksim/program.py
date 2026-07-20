"""ksim.program — flattened compiled-program IR.

Plain-numpy buffers that a tsim CompiledProgram is flattened into (by
ksim.program.flatten_program), and the exchange format the Kokkos
kernel consumes.

Layout per scalar graph (one graph per output-prefix step, each containing
num_graphs stabilizer terms):
  node_phases :  phases (G,T) u8 · params (G,T,P) u8 · counts (G,)
  halfpi      :  coeffs (G,T) u8 · params (G,T,P) u8
  pi_products :  psi/phi const (G,T) u8 · psi/phi params (G,T,P) u8
  phase_pairs :  alpha/beta (G,T) u8 · their params (G,T,P) u8 · counts (G,)
  prefactor   :  phase_indices (G,) u8 · floatfactor (G,4) i32 · power2 (G,) i32
                 · approx (G,) c64 · has_approx bool
All ragged dims are already padded by tsim's compiler.

Definitions
-----------
flatten_program
    Flatten a tsim CompiledProgram into numpy-only buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlatScalarGraph:
    np_phases: np.ndarray;  np_params: np.ndarray;  np_counts: np.ndarray
    hp_coeffs: np.ndarray;  hp_params: np.ndarray
    pp_psi_c: np.ndarray;   pp_psi_p: np.ndarray
    pp_phi_c: np.ndarray;   pp_phi_p: np.ndarray
    ph_alpha: np.ndarray;   ph_alpha_p: np.ndarray
    ph_beta: np.ndarray;    ph_beta_p: np.ndarray;  ph_counts: np.ndarray
    pf_phase_idx: np.ndarray; pf_floatfactor: np.ndarray
    pf_power2: np.ndarray;    pf_approx: np.ndarray; pf_has_approx: bool


@dataclass
class FlatComponent:
    output_indices: tuple
    f_selection: np.ndarray
    graphs: list = field(default_factory=list)


@dataclass
class FlatProgram:
    components: list
    output_order: np.ndarray
    num_outputs: int
    num_detectors: int


def _np(a):
    return np.asarray(a)


def flatten_program(program) -> FlatProgram:
    comps = []
    for c in program.components:
        fc = FlatComponent(tuple(c.output_indices), _np(c.f_selection))
        for g in c.compiled_scalar_graphs:
            fc.graphs.append(FlatScalarGraph(
                _np(g.node_phases.phases),  _np(g.node_phases.params),
                _np(g.node_phases.counts),
                _np(g.halfpi_phases.coeffs), _np(g.halfpi_phases.params),
                _np(g.pi_products.psi_const), _np(g.pi_products.psi_params),
                _np(g.pi_products.phi_const), _np(g.pi_products.phi_params),
                _np(g.phase_pairs.alpha),  _np(g.phase_pairs.alpha_params),
                _np(g.phase_pairs.beta),   _np(g.phase_pairs.beta_params),
                _np(g.phase_pairs.counts),
                _np(g.prefactor.phase_indices),
                _np(g.prefactor.floatfactor),
                _np(g.prefactor.power2),
                _np(g.prefactor.approximate_floatfactors),
                bool(g.prefactor.has_approximate_floatfactors)))
        comps.append(fc)
    return FlatProgram(comps, _np(program.output_order),
                       program.num_outputs, program.num_detectors)
