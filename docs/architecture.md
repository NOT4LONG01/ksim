# Architecture: the ZX engine and the three Kokkos extensions

This repo holds one Python package (`ksim`) and three nanobind-bound Kokkos
extensions (`kokkos_sim`, `kokkos_decoder`, `kokkos_mcts`). They live together
because they share one CMake build and one Kokkos install; only `kokkos_sim` is
part of `ksim` itself. The consumer of the other two is the
`soft-info-code-switch` QEC pipeline.

- [`ksim` — the ZX stabilizer-rank engine](#ksim)
- [`kokkos_sim` — the sampling runtime](#kokkos_sim)
- [`kokkos_decoder` — three decoder types](#kokkos_decoder)
- [`kokkos_mcts` — MCTS schedule optimizer](#kokkos_mcts)
- [C++ conventions that apply to all three](#cpp)

---

<a name="ksim"></a>
## `ksim` — the ZX stabilizer-rank engine

### Meaning

`ksim` simulates Clifford **and** non-Clifford (T-gate) circuits by the ZX
stabilizer-rank route: a stim circuit becomes a doubled+reduced ZX diagram,
the diagram's non-Clifford content is decomposed into a sum of stabilizer
terms (cat5 recursion), and each term's amplitude is evaluated in exact ℤ[ω]
arithmetic. Cost scales with T-count (χ ≈ 2^(0.228t)), not with qubit count —
which is why it handles encoded circuits stim cannot touch at all.

The package splits along one seam, the `FlatProgram`:

```
stim.Circuit
  → ksim.compile(...)          # parse → double → reduce → e→f basis,
                               #   components → plug → cat5 stabrank → emit
    → FlatProgram              # flat numpy IR — the contract between compile and sample
      → KokkosProgramSampler   # Kokkos runtime (CUDA on GPU builds, OpenMP on CPU)
      → sample_flat            # numpy reference (GPU-free)
```

### Architecture

```
program.py       FlatProgram / FlatComponent / FlatScalarGraph
                   the flat numpy IR both halves agree on

compile/         stim circuit → FlatProgram
  graph.py         stim → doubled + reduced ZX graph
  parse.py         circuit text → instruction stream
  instructions.py  the instruction vocabulary
  stabrank.py      find_stab: cat5 stabilizer-rank recursion over pyzx_param
  pipeline.py      compile_program: components → plug → term-family emit → FlatProgram
  channels.py      ChannelSampler + channel-probability builders
  cache.py         CompileCache / ksim.compile_cache — structural compile cache,
                     keyed on the zero/nonzero noise pattern, so a p-sweep at
                     fixed topology compiles once
  _linalg.py       GF(2) find_basis
  types.py         SamplingGraph
  program_text.py  shorthand → stim

sample/          FlatProgram → outcomes
  evaluate.py      evaluate_flat (amplitude) / sample_flat (samples) — numpy reference
  exact_scalar.py  ExactScalarArray: exact ℤ[ω] arithmetic (4 integer coefficients
                     + a power-of-2 exponent)
  sampler.py       KokkosProgramSampler — wraps one kokkos_sim.Component per
                     FlatComponent; drop-in replacement for sample_flat

run.py           sample_circuit: compile → draw noise → sample → (det, obs)
```

**The contract**: every field in `FlatProgram`/`FlatScalarGraph` (node_phases,
halfpi_phases, pi_products, phase_pairs, prefactor — all ℤ[ω]/power-of-2) has
its numeric meaning pinned by `sample/evaluate.py` + `sample/exact_scalar.py`
and mirrored exactly by `src/cpp/ksim/zx_eval.hpp`. That is what makes the
buffer format a *specification* rather than an implementation detail:
`compile/pipeline.py` emits exactly this shape, and the two evaluators pin
what every field means numerically.

**Why the same math is implemented twice** (numpy in `evaluate.py`, C++ in
`zx_eval.hpp`): the numpy path is reference/test-only, never on the production
sampling path. It exists so the GPU output can be checked against a
slow-but-obviously-correct implementation. Both are exact-arithmetic by
construction — numpy float64 lands at ~1e-16, `kokkos_sim` is int64-exact until
one final float64 conversion (~1e-16), while a complex64 float-tensor substrate
lands at ~1e-8. See [`benchmarks.md`](benchmarks.md) Part 1.

Usage example: [`README.md`](../README.md#use).

---

<a name="kokkos_sim"></a>
## `kokkos_sim` — ZX stabilizer-rank Kokkos runtime

The sampling backend behind `KokkosProgramSampler`: CUDA on GPU builds, OpenMP
on CPU-only builds. `zx_eval.hpp` carries the ZX-calculus math (the mirror of
`sample/evaluate.py`), `kokkos_sim.cpp` the Kokkos runtime and the nanobind
bindings. Callers reach it through `ksim`, not directly.

### Theoretical grounding: the stabilizer-rank sum

`kokkos_sim` and `tsim` ([QuEraComputing/tsim](https://github.com/QuEraComputing/tsim))
run the *same* algorithm: the circuit (composed with its adjoint) becomes a
**ZX-calculus diagram**, reduced by `pyzx`/`pyzx_param` rewrite rules. T gates
survive this translation as π/4 phase spiders. The reduced diagram is then
split by **stabilizer-rank decomposition**: each magic (π/4) spider group is
recursively replaced by a sum of stabilizer terms (≈2^{αt} terms for t
T-gates, α<1 with the cat-state strategies). Each output probability becomes
an exact closed-form sum over those terms — four symbolic "term families" per
term, with coefficients in the ring **ℤ[ω], ω = e^{iπ/4}** (integers
a+bω+ci+dω̄ scaled by powers of 2). Sampling is autoregressive:
P(bit_i = 1 | previous bits) = |amplitude with bit i plugged|/|previous
marginal|, one Bernoulli draw per output. Hardness is paid where it belongs —
exponentially in T-count, polynomially in everything else.

The two runtimes diverge only in *how* they evaluate that shared sum:

```
            symbolic (once per circuit, Python)                numeric (per shot)
 ┌────────────────────────────────────────────────────┐   ┌─────────────────────────┐
 | circuit             →   ZX diagram   →    stabilizer- |   |  autoregressive loop:   |
 |      (T†/T gates)       (pyzx reduce)    rank sum   | → |  P(bit=1 | prev bits) = |
 |                                          Σᵢ cᵢ·|sᵢ⟩  |   |  |amp(bit=1)|²/|prev|²  |
 |  T survives as π/4      magic spiders → ≈2^(αt)     |   |  one Bernoulli per bit  |
 |  phase spider           stabilizer terms, cᵢ ∈ ℤ[ω]  |   |                         |
 └────────────────────────────────────────────────────┘   └─────────────────────────┘
        identical for tsim and kokkos_sim                   tsim: JAX dispatch per
                                                            step, cᵢ in complex64
                                                            kokkos_sim: one fused
                                                            kernel, cᵢ in int64
```

### How the Kokkos translation works

The pipeline splits at a natural seam: everything **symbolic** is one-time
work per circuit, everything **numeric** repeats per shot.

| | stays in Python (`ksim`/`pyzx_param`) | moves to Kokkos (`kokkos_sim`) |
|---|---|---|
| ZX graph build + reduction | ✓ | |
| stabilizer-rank decomposition | ✓ | |
| term-family compilation | ✓ | |
| amplitude evaluation | (CPU numpy reference) | fused CUDA kernel |
| autoregressive sampling loop | (CPU numpy reference) | same kernel, per-shot |

`ksim.compile()` emits the flat numpy `FlatProgram` buffers directly
(bitmasks, phases, ℤ[ω] prefactors). `kokkos_sim.Component` uploads them
once; a single kernel launch then runs the *whole* autoregressive loop for a
batch — each GPU thread owns one shot, computes parities by walking the
bitmasks, multiplies ℤ[ω] coefficients in int64 with power-of-2
renormalisation, and draws output bits from its own RNG. The parameter
vector [error bits | sampled bits | trying bit] is never materialised; bits
are looked up on the fly. This removes the per-step JAX dispatch that tsim
pays on every shot.

### Why kokkos_sim wins on precision *and* speed — no trade-off involved

Everything left of the arrow above is shared: both runtimes evaluate the
*same exact* closed-form sum with the same terms and the same ℤ[ω]
coefficients. After compile there is no inherent floating-point math in this
algorithm at all — evaluating a term is GF(2) parity walks over bitmasks
plus exact integer ℤ[ω] multiplies. Both of tsim's costs are therefore
artifacts of its runtime substrate, not of the algorithm:

- it stores the exact integers in complex64 **because XLA wants float
  tensors**, rounding numbers that are exactly representable in int64;
- it drives the per-bit autoregressive loop from Python through JAX
  dispatch **because XLA wants whole-array ops**, wrapping trivial math in
  dispatch overhead.

kokkos_sim removes the substrate instead of optimizing within it: int64
ℤ[ω] coefficients with power-of-2 renormalisation (exact until one final
float64 conversion), and the whole loop fused into one CUDA kernel with a
thread per shot. Precision and speed improve together because neither was
being traded against the other — both were being paid to the same
middleman. Measured numbers: [`benchmarks.md`](benchmarks.md) Part 1.

### Precision design: why the arithmetic stays exact

- **The stabilizer-rank sum is exact** — unlike Pauli-propagation or
  tensor-network truncation there is no controllable-error knob; the only
  approximation anywhere is float rounding at the very end. (Exception:
  phases with denominators outside {1,2,4} fold into an "approximate
  floatfactor"; both runtimes inherit the same float32 constants there.)
- **The autoregressive recurrence `prev ← prev − p1` is the conditioning
  hazard**: when a conditional probability approaches 0, subtractive
  cancellation amplifies relative error. tsim monitors this (warns when the
  marginal norm deviates from 1 by >1e-5, fails near 1); the int64/float64
  Kokkos path pushes the cancellation floor far lower than a float32
  substrate can reach.
- **Coefficient growth is bounded** by the per-multiply renormalisation
  (divide by 2 whenever all four ℤ[ω] coefficients are even, tracking the
  exponent separately) — the same scheme as tsim, but int64 headroom (2⁶³)
  versus the ~2²⁴ exact-integer ceiling of float32 means deep products
  cannot silently lose low bits.

Measured amplitude deviations and the validation gates that check them:
[`benchmarks.md`](benchmarks.md) Part 1.

---

<a name="kokkos_decoder"></a>
## `kokkos_decoder` — three decoder types

```python
from kokkos_decoder import BpOsd0Decoder, RelayBpDecoder, BpLsdDecoder
dec = BpOsd0Decoder(H_dense_uint8, priors_float32, max_iter=30, max_batch=4096)
preds, converged = dec.decode_batch(syndromes_uint8)  # (B,nb), (B,)
```

`H` must be a dense `np.ndarray`, uint8, shape `(nc, nb)`. `converged` is
BP's own verdict per shot: where it is 0, OSD-0 (or LSD, or a relay's best
leg for `RelayBpDecoder`) supplied the prediction.

- `BpOsd0Decoder`: product-sum BP; non-converged shots get OSD-0 (bit-packed
  team-parallel Gauss–Jordan, columns sorted by signed posterior LLR ascending
  — most-likely-error first). Posteriors within `OSD_TIE_TOL = 1e-3` of each
  other are ordered by channel LLR ascending, then column index, which makes
  the output identical run to run despite the float atomics in the message
  sums. Predictions match `ldpc.BpOsdDecoder` on every shot where BP converges
  and on the OSD-0 shots up to that tie order (99.98 % of shots on triangular
  n=19, equal LER on tetrahedral n=15). Where a model has degenerate
  mechanisms — three columns on three detectors, any two composing to the
  third with a different observable, 156 such pairs in the triangular n=7
  graphlike model — OSD-0's choice among their tied posteriors decides the
  logical outcome, and ldpc's stable sort of its double-rounding noise picks
  differently from any deterministic rule; that is a property of OSD-0 on a
  tiny code, not a difference in the BP.
- `RelayBpDecoder` (arXiv:2506.01779): `pre_iter` iterations of BP (memory
  `gamma0` if non-zero), then `num_legs` disordered-memory legs
  (per-(shot, variable) γ ~ U[gamma_min, gamma_max], re-drawn each leg via
  on-device splitmix64 hash, applied as a prior↔posterior blend). Each leg
  restarts the messages from the priors and keeps only the previous posterior
  as memory — the relay. A shot decodes until it has produced `stop_nconv`
  converged solutions and returns the one of lowest prior weight (5 is the
  paper's Relay-BP-S; `<= 0` runs every leg); shots that never converge fall
  back to OSD-0. **Short legs win**: `leg_max_iter ≈ 8` with 10–20 legs beats
  `leg_max_iter = 30` on both LER and speed — the γ re-randomisation, not BP
  depth, is what escapes trapping sets. With the reference's own settings
  (`gamma0=0.1, num_legs=300, leg_max_iter=60, stop_nconv=5`) it lands on the
  reference implementation's LER (tetrahedral n=15, p=5e-3, 20 000 shots:
  4.5e-4 vs 4.0e-4), where `stop_nconv=1` plain-BP pre-phase gives 1.25e-3.
- `BpLsdDecoder`: GPU BP runs in batch; non-converged shots fall through to CPU
  cluster BFS + brute force (≤ `max_cluster_bits`) or OSD-0. The hybrid keeps
  mean throughput fast while improving accuracy on hard instances.

Source layout: `bp_decoders.hpp` (the API) + `bp_infra.cpp`, `bp_osd.cpp`,
`bp_lsd.cpp`, `relay_bp.cpp` + `kokkos_decoder.cpp` (nanobind entry).

How these compare to nv-qldpc and Roffe's `ldpc`, and how they got there:
[`benchmarks.md`](benchmarks.md) Part 2.

---

<a name="kokkos_mcts"></a>
## `kokkos_mcts` — MCTS schedule optimizer

```python
from kokkos_mcts import MctsScheduler, evaluate_schedule
sched = MctsScheduler(num_checks, num_bits, num_ticks, nshots=4096, max_bp_iter=30, seed=42)
best = sched.schedule(H_dense, priors, iters=500)  # returns a flat list of ints
```

Searches CNOT measurement schedules, scoring candidates by decoding them.
Source layout: `scheduler.hpp` (the API) + `mcts.cpp`, `schedule_eval.cpp`,
`mpi_search.cpp` + `kokkos_mcts.cpp` (nanobind entry). It reuses the decoder's
BP via `#include "../decoder/bp_decoders.hpp"` and links `decoder/bp_infra.cpp`
+ `decoder/bp_osd.cpp` into its own module, and it configures against MPI.

---

<a name="cpp"></a>
## C++ conventions that apply to all three

Files sit flat in each module directory (no `include/`/`src/` split), and a
single `src/cpp/CMakeLists.txt` builds all three modules. Naming: one header
per module carrying its API, named distinctly from the `.cpp` implementations,
and the nanobind entry file named after the module.

- **Never use sub-word (uint8/uint16) types for contended atomics in Kokkos
  CUDA kernels** — they fall back to desul CAS/lock emulation and can cost
  100×+ under contention; use `unsigned int`/`uint64_t` (native hardware
  atomics). This exact bug once made the decoder's BP ~200× slower (the
  `row_par` parity check).
- **CPU/GPU portability**: device memory goes through the `DeviceSpace` alias
  (`bp_decoders.hpp` / `zx_eval.hpp`, = `Kokkos::DefaultExecutionSpace::memory_space`)
  — never hardcode `Kokkos::CudaSpace`. OSD team size is 128 only under
  `KOKKOS_ENABLE_CUDA`, `Kokkos::AUTO` otherwise.
- **Rebuild all three extensions together, never just one.** Each links Kokkos
  statically, so a `.so` carries its own private copy of Kokkos's global
  runtime state. If two of them end up built against *different* Kokkos
  installs and both get loaded into one process, the second one's
  `Kokkos::initialize()` corrupts the first's device/threading state — a
  segfault (`rc=-11`) at the point the second extension is first *used*, not at
  import time, so it can look unrelated to whatever you rebuilt.
- **nanobind 2.12.0**: all modules use `nb::ndarray<nb::numpy, uint8_t>` +
  heap-owned `nb::capsule`.

Build recipes (CPU and CUDA): [`build.md`](build.md).
