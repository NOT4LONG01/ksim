"""ksim.evaluate — pure-NumPy reference evaluator/sampler.

Mirrors tsim.compile.evaluate.evaluate and tsim.sampler.sample_program
(sequential mode) over the FlatProgram IR, validated against tsim's JAX runtime.
Used as the CPU reference; the GPU path lives in ksim.sampler.

Definitions
-----------
evaluate_flat
    NumPy mirror of tsim.compile.evaluate.evaluate → complex (B,).

sample_flat
    NumPy mirror of tsim.sampler.sample_program (sequential mode).
"""
from __future__ import annotations

import numpy as np

from ..program import FlatScalarGraph, FlatProgram
from .exact_scalar import (
    UNIT_PHASES, ONE_PLUS_PHASES, IDENTITY4,
    _mul4, _prod4, _to_complex, _gf2,
)


def evaluate_flat(g: FlatScalarGraph, param_vals: np.ndarray) -> np.ndarray:
    if g.pf_phase_idx.shape[0] == 0:
        return np.zeros(param_vals.shape[0], dtype=np.complex128)
    B = param_vals.shape[0]
    Gn = g.pf_phase_idx.shape[0]

    rs = _gf2(g.np_params, param_vals)
    idx = (4*rs + g.np_phases[None]) % 8
    tv = ONE_PLUS_PHASES[idx]
    mask = np.arange(g.np_phases.shape[1])[None, :] < g.np_counts[:, None]
    tv = np.where(mask[None, ..., None], tv, IDENTITY4)
    np_c, np_p = _prod4(tv, axis=2)

    rs = _gf2(g.hp_params, param_vals)
    tot = np.sum((rs * g.hp_coeffs[None]) % 8, axis=-1) % 8
    hp_c = UNIT_PHASES[tot]

    psi = (g.pp_psi_c[None] + _gf2(g.pp_psi_p, param_vals)) % 2
    phi = (g.pp_phi_c[None] + _gf2(g.pp_phi_p, param_vals)) % 2
    sgn = np.sum(psi * phi, axis=-1) % 2
    pp_c = (1 - 2*sgn)[..., None] * IDENTITY4

    ra = _gf2(g.ph_alpha_p, param_vals); rb = _gf2(g.ph_beta_p, param_vals)
    al = (g.ph_alpha[None] + 4*ra) % 8;  be = (g.ph_beta[None] + 4*rb) % 8
    tv = IDENTITY4 + UNIT_PHASES[al] + UNIT_PHASES[be] - UNIT_PHASES[(al+be) % 8]
    mask = np.arange(g.ph_alpha.shape[1])[None, :] < g.ph_counts[:, None]
    tv = np.where(mask[None, ..., None], tv, IDENTITY4)
    ph_c, ph_p = _prod4(tv, axis=2)

    static_c = UNIT_PHASES[g.pf_phase_idx]
    ff_c     = g.pf_floatfactor.astype(np.int64)

    coeffs = _mul4(_mul4(_mul4(np_c, hp_c), _mul4(pp_c, ph_c)),
                   _mul4(np.broadcast_to(static_c, (B, Gn, 4)),
                         np.broadcast_to(ff_c,     (B, Gn, 4))))
    power = np_p + ph_p

    z = _to_complex(coeffs, power)
    if g.pf_has_approx:
        return np.sum(z * g.pf_approx[None] * np.exp2(g.pf_power2)[None], axis=-1)
    return np.sum(z * np.exp2(g.pf_power2.astype(np.float64))[None], axis=-1)


def sample_flat(prog: FlatProgram, f_params: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    B = f_params.shape[0]
    outs = []
    for comp in prog.components:
        n_out = len(comp.graphs) - 1
        f_sel = f_params[:, comp.f_selection].astype(np.int64)
        m = np.zeros((B, n_out), dtype=np.int64)
        prev = np.abs(evaluate_flat(comp.graphs[0], f_sel))
        for i, g in enumerate(comp.graphs[1:]):
            ones = np.ones((B, 1), dtype=np.int64)
            params = np.hstack([f_sel, m[:, :i], ones])
            p1 = np.abs(evaluate_flat(g, params))
            bits = rng.random(B) < (p1 / prev)
            m[:, i] = bits
            prev = np.where(bits, p1, prev - p1)
        outs.append(m)
    combined = np.concatenate(outs, axis=1) if outs else np.zeros((B, 0), int)
    return combined[:, np.argsort(prog.output_order)]
