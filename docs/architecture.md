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

### Use

```python
import ksim

# one call: compile → draw noise → sample → (det, obs)
det, obs = ksim.sample_circuit(stim_text, n_shots, seed=42, backend="auto")  # or "gpu"/"cpu"

# or the explicit pieces (e.g. to reuse the compiled program):
flat, channel_probs, error_transform = ksim.compile(stim_text, sample_detectors=True)
ks = ksim.KokkosProgramSampler(flat)       # GPU; or ksim.sample_flat(flat, f, rng) on CPU
f = ksim.ChannelSampler(channel_probs, error_transform,
                        seed=42).sample(n_shots).astype("uint8")   # noise bits, CPU
out = ks.sample(f, seed=42)                # (B, n_outputs); det = out[:, :flat.num_detectors]
```

Repeated compiles of the same circuit *shape* hit the process-wide
`ksim.compile_cache` automatically; pass `cache=None` to opt out.

`example/sample_demo.py` runs the whole path on a plain stim circuit, GPU if
`kokkos_sim` is built and the numpy reference otherwise.

---

<a name="kokkos_sim"></a>
## `kokkos_sim` — ZX stabilizer-rank Kokkos runtime

The sampling backend behind `KokkosProgramSampler`: CUDA on GPU builds, OpenMP
on CPU-only builds. `zx_eval.hpp` carries the ZX-calculus math (the mirror of
`sample/evaluate.py`), `kokkos_sim.cpp` the Kokkos runtime and the nanobind
bindings. Callers reach it through `ksim`, not directly.

---

<a name="kokkos_decoder"></a>
## `kokkos_decoder` — three decoder types

```python
from kokkos_decoder import BpOsd0Decoder, RelayBpDecoder, BpLsdDecoder
dec = BpOsd0Decoder(H_dense_uint8, priors_float32, max_iter=30, max_batch=4096)
preds, converged = dec.decode_batch(syndromes_uint8)  # (B,nb), (B,)
```

`H` must be a dense `np.ndarray`, uint8, shape `(nc, nb)`.

- `BpOsd0Decoder`: product-sum BP; non-converged shots get OSD-0 (bit-packed
  team-parallel Gauss–Jordan, columns sorted by signed posterior LLR ascending
  — most-likely-error first; predictions are bit-identical to
  `ldpc.BpOsdDecoder`).
- `RelayBpDecoder`: `pre_iter` standard BP, then `num_legs` disordered-memory
  legs (Relay-BP: per-(shot, variable) γ ~ U[gamma_min, gamma_max], re-drawn
  each leg via on-device splitmix64 hash, applied as a prior↔posterior blend),
  OSD-0 fallback on the rest. **Short legs win**: `leg_max_iter ≈ 8` with 10–20
  legs beats `leg_max_iter = 30` on both LER and speed — the γ re-randomisation,
  not BP depth, is what escapes trapping sets.
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
