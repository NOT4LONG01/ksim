"""Compile a prepared ZX graph into ``ksim.FlatProgram`` buffers.

Derived from bloqade-tsim's ``tsim.compile.pipeline`` + ``tsim.compile.compile``
(Apache-2.0). The semantics are unchanged; the only differences vs. upstream:

* the stabilizer-rank decomposition (``ksim.stabrank.find_stab``) calls
  ``pyzx_param`` directly — the cat5 magic-state cascade is pyzx_param's, not
  tsim's;
* the four term families + prefactor are emitted as plain numpy into
  ``ksim.FlatScalarGraph`` / ``FlatComponent`` / ``FlatProgram`` directly,
  dropping tsim's equinox/JAX ``CompiledScalarGraphs`` and the
  ``ksim.flatten_program`` round-trip.

So ``compile_program(prepare_graph(...))`` here is a drop-in replacement for
tsim's ``flatten_program(compile_program(prepare_graph(...)))`` and is verified
to reproduce tsim's ``FlatProgram`` buffers byte-for-byte on the golden fixtures
in ``tests/fixtures/zx_compiler/``.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Literal

import numpy as np
import pyzx_param as zx
from pyzx_param.graph.base import BaseGraph
from pyzx_param.graph.scalar import DyadicNumber
from pyzx_param.simulate import DecompositionStrategy

from ..program import FlatComponent, FlatProgram, FlatScalarGraph
from .graph import ConnectedComponent, connected_components, get_params
from .stabrank import find_stab
from .types import SamplingGraph

DecompositionMode = Literal["sequential", "joint"]


# ---------------------------------------------------------------------------
# Pipeline: prepared graph -> FlatProgram
# ---------------------------------------------------------------------------
def compile_program(
    prepared: SamplingGraph,
    *,
    mode: DecompositionMode = "sequential",
    strategy: DecompositionStrategy = "cat5",
) -> FlatProgram:
    """Compile a prepared graph into a ``ksim.FlatProgram`` ready for sampling.

    1. Split the graph into connected components.
    2. Per component (sorted by #outputs): plug outputs, reduce each plugged
       graph, stabilizer-rank decompose, emit the term families.
    3. Assemble into a ``FlatProgram`` with the global output ordering.
    """
    components = connected_components(prepared.graph)
    f_indices_global = _get_f_indices(prepared.graph)
    num_outputs = prepared.num_outputs

    compiled_components: list[FlatComponent] = []
    output_order: list[int] = []

    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    for component in sorted_components:
        compiled_components.append(
            _compile_component(
                component=component,
                f_indices_global=f_indices_global,
                mode=mode,
                strategy=strategy,
            )
        )
        output_order.extend(component.output_indices)

    return FlatProgram(
        components=compiled_components,
        output_order=np.array(output_order, dtype=np.int32),
        num_outputs=num_outputs,
        num_detectors=prepared.num_detectors,
    )


def _get_f_indices(graph: BaseGraph) -> list[int]:
    """Extract numerically sorted list of f-parameter indices from the graph."""
    all_params = get_params(graph)
    return sorted(int(p[1:]) for p in all_params if p.startswith("f"))


def _remove_phase_terms(graph: BaseGraph) -> None:
    """Remove parametrized global-phase terms from the graph's scalar."""
    graph.scalar.phasevars_halfpi = {}
    graph.scalar.phasevars_pi_pair = []


def _compile_component(
    component: ConnectedComponent,
    f_indices_global: list[int],
    mode: DecompositionMode,
    strategy: DecompositionStrategy = "cat5",
) -> FlatComponent:
    """Compile a single connected component into a ``FlatComponent``."""
    graph = component.graph
    output_indices = component.output_indices
    num_component_outputs = len(graph.outputs())

    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    outputs_to_plug = (
        list(range(num_component_outputs + 1))
        if mode == "sequential"
        else [0, num_component_outputs]
    )

    compiled_graphs: list[FlatScalarGraph] = []

    component_m_chars = [f"m{i}" for i in output_indices]
    plugged_graphs = _plug_outputs(graph, component_m_chars, outputs_to_plug)

    power2_base: int | None = None

    for num_m_plugged, plugged_graph in zip(
        outputs_to_plug, plugged_graphs, strict=True
    ):
        g_copy = plugged_graph.copy()
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()

        if power2_base is None:
            power2_base = g_copy.scalar.power2
        g_copy.scalar.add_power(-power2_base)

        _remove_phase_terms(g_copy)

        param_names = [f"f{i}" for i in f_selection]
        param_names += [f"m{output_indices[j]}" for j in range(num_m_plugged)]

        g_list = find_stab(g_copy, strategy=strategy)

        if len(g_list) == 1:
            # Clifford graph: the global phase terms are unobservable.
            _remove_phase_terms(g_list[0])

        compiled_graphs.append(_compile_flat_scalar_graph(g_list, param_names))

    return FlatComponent(
        output_indices=tuple(output_indices),
        f_selection=np.array(f_selection, dtype=np.int32),
        graphs=compiled_graphs,
    )


def _plug_outputs(
    graph: BaseGraph,
    m_chars: list[str],
    outputs_to_plug: list[int],
) -> list[BaseGraph]:
    """Create graphs with the specified numbers of outputs plugged."""
    graphs: list[BaseGraph] = []
    num_outputs = len(graph.outputs())

    for num_plugged in outputs_to_plug:
        g = graph.copy()
        output_vertices = list(g.outputs())

        # '0' plugs (measures) an output; '+' traces (marginalizes) it.
        effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
        g.apply_effect(effect)
        for i, v in enumerate(output_vertices[:num_plugged]):
            g.set_phase(v, m_chars[i])  # type: ignore[arg-type]

        # Compensate power for trace of unplugged outputs.
        g.scalar.add_power(num_outputs - num_plugged)
        graphs.append(g)

    return graphs


# ---------------------------------------------------------------------------
# Term-family emission: scalar graphs -> FlatScalarGraph (numpy)
# ---------------------------------------------------------------------------
def _compile_flat_scalar_graph(
    g_list: list[BaseGraph], params: list[str]
) -> FlatScalarGraph:
    """Emit the four term families + prefactor of a scalar-graph list as numpy.

    Mirrors tsim's ``compile_scalar_graphs`` but writes directly into the
    ``ksim.FlatScalarGraph`` layout (no equinox/JAX intermediary).
    """
    for i, g in enumerate(g_list):
        n_vertices = len(list(g.vertices()))
        if n_vertices != 0:
            raise ValueError(
                f"Only scalar graphs can be compiled but graph {i} has "
                f"{n_vertices} vertices"
            )
        if g.scalar.phasevars_pi and not g.scalar.is_zero:
            raise NotImplementedError(
                f"compile does not support Scalar.phasevars_pi (graph {i} has "
                f"phasevars_pi={sorted(g.scalar.phasevars_pi)!r})"
            )

    g_list = [g for g in g_list if not g.scalar.is_zero]

    n_params = len(params)
    char_to_idx = {char: i for i, char in enumerate(params)}

    np_phases, np_params, np_counts = _node_phases(g_list, char_to_idx, n_params)
    hp_coeffs, hp_params = _halfpi_phases(g_list, char_to_idx, n_params)
    (pp_psi_c, pp_psi_p, pp_phi_c, pp_phi_p) = _pi_products(
        g_list, char_to_idx, n_params
    )
    (ph_alpha, ph_alpha_p, ph_beta, ph_beta_p, ph_counts) = _phase_pairs(
        g_list, char_to_idx, n_params
    )
    (pf_phase_idx, pf_floatfactor, pf_power2, pf_approx, pf_has_approx) = _prefactor(
        g_list
    )

    return FlatScalarGraph(
        np_phases=np_phases, np_params=np_params, np_counts=np_counts,
        hp_coeffs=hp_coeffs, hp_params=hp_params,
        pp_psi_c=pp_psi_c, pp_psi_p=pp_psi_p,
        pp_phi_c=pp_phi_c, pp_phi_p=pp_phi_p,
        ph_alpha=ph_alpha, ph_alpha_p=ph_alpha_p,
        ph_beta=ph_beta, ph_beta_p=ph_beta_p, ph_counts=ph_counts,
        pf_phase_idx=pf_phase_idx, pf_floatfactor=pf_floatfactor,
        pf_power2=pf_power2, pf_approx=pf_approx, pf_has_approx=pf_has_approx,
    )


def _node_phases(g_list, char_to_idx, n_params):
    """Π (1 + ω^(4·parity(params) + α)) terms — phases (G,T), params (G,T,P), counts (G,)."""
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int]]]] = [[] for _ in range(num_graphs)]

    for i, g_i in enumerate(g_list):
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            assert g_i.scalar.phasenodes[term].denominator in [1, 2, 4]
            const_term = int(g_i.scalar.phasenodes[term] * 4)
            terms_per_graph[i].append((const_term, bitstr))

    counts = np.array([len(terms) for terms in terms_per_graph], dtype=np.int32)
    max_terms = int(counts.max()) if counts.size else 0

    phases = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    params = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    for i, terms in enumerate(terms_per_graph):
        for j, (const_phase, param_bit) in enumerate(terms):
            phases[i, j] = const_phase % 8
            params[i, j] = param_bit
    return phases, params, counts


def _halfpi_phases(g_list, char_to_idx, n_params):
    """ω^(Σ coeffs·parity) terms — same-bitstring j-values combined mod 4."""
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int]]]] = [[] for _ in range(num_graphs)]

    for i, g_i in enumerate(g_list):
        assert set(g_i.scalar.phasevars_halfpi.keys()) <= {1, 3}
        bitstr_to_j: dict[tuple[int, ...], int] = defaultdict(int)
        for j in [1, 3]:
            if j not in g_i.scalar.phasevars_halfpi:
                continue
            for term in range(len(g_i.scalar.phasevars_halfpi[j])):
                bitstr = [0] * n_params
                for v in g_i.scalar.phasevars_halfpi[j][term]:
                    bitstr[char_to_idx[v]] = 1
                bitstr_key = tuple(bitstr)
                bitstr_to_j[bitstr_key] = (bitstr_to_j[bitstr_key] + j) % 4
        for bitstr_key, combined_j in bitstr_to_j.items():
            if combined_j == 0:
                continue
            terms_per_graph[i].append((combined_j * 2, list(bitstr_key)))

    max_terms = max((len(terms) for terms in terms_per_graph), default=0)
    coeffs = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    params = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    for i, terms in enumerate(terms_per_graph):
        for j, (coeff, param_bit) in enumerate(terms):
            coeffs[i, j] = coeff
            params[i, j] = param_bit
    return coeffs, params


def _pi_products(g_list, char_to_idx, n_params):
    """Π (-1)^(ψ·φ) terms — ψ/φ each = const bit XOR parameter parity."""
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int], int, list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i, graph in enumerate(g_list):
        for p_set in graph.scalar.phasevars_pi_pair:
            psi_const = 1 if "1" in p_set[0] else 0
            psi_params = [0] * n_params
            for p in p_set[0]:
                if p != "1":
                    psi_params[char_to_idx[p]] = 1
            phi_const = 1 if "1" in p_set[1] else 0
            phi_params = [0] * n_params
            for p in p_set[1]:
                if p != "1":
                    phi_params[char_to_idx[p]] = 1
            terms_per_graph[i].append((psi_const, psi_params, phi_const, phi_params))

    max_terms = max((len(terms) for terms in terms_per_graph), default=0)
    psi_const_arr = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    psi_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    phi_const_arr = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    phi_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    for i, terms in enumerate(terms_per_graph):
        for j, (psi_c, psi_p, phi_c, phi_p) in enumerate(terms):
            psi_const_arr[i, j] = psi_c
            psi_params_arr[i, j] = psi_p
            phi_const_arr[i, j] = phi_c
            phi_params_arr[i, j] = phi_p
    return psi_const_arr, psi_params_arr, phi_const_arr, phi_params_arr


def _phase_pairs(g_list, char_to_idx, n_params):
    """Π (1 + ω^α + ω^β − ω^(α+β)) terms."""
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, int, list[int], list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i, graph in enumerate(g_list):
        for pp in range(len(graph.scalar.phasepairs)):
            alpha_params = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsA:
                alpha_params[char_to_idx[v]] = 1
            beta_params = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsB:
                beta_params[char_to_idx[v]] = 1
            const_alpha = int(graph.scalar.phasepairs[pp].alpha)
            const_beta = int(graph.scalar.phasepairs[pp].beta)
            terms_per_graph[i].append(
                (const_alpha, const_beta, alpha_params, beta_params)
            )

    counts = np.array([len(terms) for terms in terms_per_graph], dtype=np.int32)
    max_terms = int(counts.max()) if counts.size else 0
    alpha = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    beta = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    alpha_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    beta_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    for i, terms in enumerate(terms_per_graph):
        for j, (ca, cb, pa, pb) in enumerate(terms):
            alpha[i, j] = ca % 8
            beta[i, j] = cb % 8
            alpha_params_arr[i, j] = pa
            beta_params_arr[i, j] = pb
    return alpha, alpha_params_arr, beta, beta_params_arr, counts


def _prefactor(g_list):
    """Per-graph static scalar: phase_idx (G,), floatfactor (G,4), power2 (G,),
    approx (G,) c64, has_approx bool."""
    for g in g_list:
        if g.scalar.phase.denominator not in [1, 2, 4]:
            g.scalar.approximate_floatfactor *= np.exp(1j * g.scalar.phase * np.pi)
            g.scalar.phase = Fraction(0, 1)

    has_approx = any(g.scalar.approximate_floatfactor != 1.0 for g in g_list)
    approx = np.array(
        [g.scalar.approximate_floatfactor for g in g_list], dtype=np.complex64
    )
    phase_indices = np.array(
        [int(float(g.scalar.phase) * 4) % 8 for g in g_list], dtype=np.uint8
    )

    exact_floatfactor = []
    power2 = []
    for g in g_list:
        dn = g.scalar.floatfactor.copy()
        p_sqrt2 = g.scalar.power2
        if p_sqrt2 % 2 != 0:
            p_sqrt2 -= 1
            dn *= DyadicNumber(k=0, a=0, b=1, c=0, d=1)
        assert p_sqrt2 % 2 == 0
        p_sqrt2 -= 2 * dn.k
        dn.k = 0
        power2.append(p_sqrt2 // 2)
        exact_floatfactor.append([dn.a, dn.b, dn.c, dn.d])

    return (
        phase_indices,
        np.array(exact_floatfactor, dtype=np.int32).reshape(-1, 4),
        np.array(power2, dtype=np.int32),
        approx,
        bool(has_approx),
    )
