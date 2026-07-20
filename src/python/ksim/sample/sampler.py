"""ksim.sampler — Kokkos GPU runtime for a flattened tsim program (Phase 2).

KokkosProgramSampler drives the kokkos_sim GPU kernel over the FlatProgram IR;
it is a drop-in for the numpy reference ksim.evaluate.sample_flat.

Definitions
-----------
_graph_to_dict
    Pack one FlatScalarGraph for kokkos_sim.Component.

KokkosProgramSampler
    GPU runtime for a flattened tsim program (drop-in for sample_flat).
"""
from __future__ import annotations

import numpy as np

from ..program import FlatScalarGraph, FlatProgram


def _graph_to_dict(g: FlatScalarGraph) -> dict:
    apx = np.array(g.pf_approx, dtype=np.complex64, order='C')
    return dict(
        np_phases=np.array(g.np_phases, dtype=np.uint8, order='C'),
        np_params=np.array(g.np_params, dtype=np.uint8, order='C'),
        np_counts=np.array(g.np_counts, dtype=np.int32, order='C'),
        hp_coeffs=np.array(g.hp_coeffs, dtype=np.uint8, order='C'),
        hp_params=np.array(g.hp_params, dtype=np.uint8, order='C'),
        pp_psi_c=np.array(g.pp_psi_c, dtype=np.uint8, order='C'),
        pp_psi_p=np.array(g.pp_psi_p, dtype=np.uint8, order='C'),
        pp_phi_c=np.array(g.pp_phi_c, dtype=np.uint8, order='C'),
        pp_phi_p=np.array(g.pp_phi_p, dtype=np.uint8, order='C'),
        ph_alpha=np.array(g.ph_alpha, dtype=np.uint8, order='C'),
        ph_alpha_p=np.array(g.ph_alpha_p, dtype=np.uint8, order='C'),
        ph_beta=np.array(g.ph_beta, dtype=np.uint8, order='C'),
        ph_beta_p=np.array(g.ph_beta_p, dtype=np.uint8, order='C'),
        ph_counts=np.array(g.ph_counts, dtype=np.int32, order='C'),
        pf_phase_idx=np.array(g.pf_phase_idx, dtype=np.uint8, order='C'),
        pf_floatfactor=np.array(g.pf_floatfactor, dtype=np.int32, order='C'),
        pf_power2=np.array(g.pf_power2, dtype=np.int32, order='C'),
        pf_approx_f32=apx.view(np.float32),
        pf_has_approx=bool(g.pf_has_approx),
    )


class KokkosProgramSampler:

    def __init__(self, prog: FlatProgram):
        import kokkos_sim
        self.prog = prog
        self.components = []
        for comp in prog.components:
            F = int(np.asarray(comp.f_selection).shape[0])
            kc = kokkos_sim.Component([_graph_to_dict(g) for g in comp.graphs], F)
            self.components.append((comp, kc))

    def sample(self, f_params: np.ndarray, seed: int) -> np.ndarray:
        B = f_params.shape[0]
        outs = []
        for k, (comp, kc) in enumerate(self.components):
            f_sel = np.ascontiguousarray(
                f_params[:, comp.f_selection], dtype=np.uint8)
            outs.append(np.asarray(kc.sample(f_sel, seed + 0x9E37 * k))
                        [:, :len(comp.graphs) - 1])
        combined = (np.concatenate(outs, axis=1) if outs
                    else np.zeros((B, 0), np.uint8))
        return combined[:, np.argsort(self.prog.output_order)]
