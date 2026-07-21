# ksim

ZX stabilizer-rank engine: symbolic **compile** + **sample** for Clifford and
non-Clifford (T-gate) circuits, on `pyzx_param`.

```
stim.Circuit
  → ksim.compile(...)          # parse → double → reduce → e→f basis,
                               #   components → plug → cat5 stabrank → emit
    → FlatProgram              # flat numpy IR — the contract between compile and sample
      → KokkosProgramSampler   # Kokkos runtime (CUDA on GPU builds, OpenMP on CPU)
      → sample_flat            # numpy reference (GPU-free)
```

Exact ℤ[ω] int64 arithmetic on device — amplitude error ~1e-16 vs float64
reference. One engine for Clifford and non-Clifford circuits alike.

## How it works

**stim** rests on Gottesman–Knill: a Clifford state is a stabilizer tableau,
every gate a row update — but a T gate conjugates a Pauli into a non-Pauli, so
the tableau has nowhere to put it. `ksim` changes representation: the circuit
(composed with its adjoint) becomes a **ZX-calculus diagram** reduced by
`pyzx_param`; T gates survive as π/4 phase spiders, and **stabilizer-rank
decomposition** replaces each magic-spider group by a sum of ≈2^{αt} stabilizer
terms (α<1 with cat-state strategies). Each output probability becomes an exact
closed-form sum with coefficients in the ring **ℤ[ω], ω = e^{iπ/4}**. Sampling
is autoregressive — one Bernoulli draw per output. Hardness is paid
exponentially in T-count, polynomially in everything else.

`ksim.compile` does the symbolic work in Python (on `pyzx_param`) and emits the
flat numpy `FlatProgram`; `kokkos_sim` uploads it once and runs the whole
autoregressive loop on GPU in int64 ℤ[ω] with power-of-2 renormalisation (or
`ksim.sample_flat` runs the same on CPU, GPU-free). The precision and speed
analysis behind the ~1e-16 figure is in [`docs/benchmarks.md`](docs/benchmarks.md).

## Install

```bash
pip install -e .               # Python layer (numpy, stim, pyzx_param)
```

This repo also hosts the other two Kokkos/CUDA extensions used by the
`soft-info-code-switch` QEC pipeline — they share the same Kokkos build and
`src/cpp/` tree, so they live together here while their Python glue stays in
`soft-info-code-switch`:

| Extension | Imported as | Used by |
|-----------|-------------|---------|
| `kokkos_sim` | `import kokkos_sim` | `ksim.KokkosProgramSampler` (this package) |
| `kokkos_decoder` | `import kokkos_decoder` | `soft-info`'s `decode.decoders` (BP+OSD / Relay-BP / BP+LSD) |
| `kokkos_mcts` | `import kokkos_mcts` | `soft-info`'s `simulations/optimize_schedule.py` (needs MPI) |

The C++ backends are a separate CMake build (needs a Kokkos install). Without
them, `ksim.compile` and the numpy reference `ksim.sample_flat` still work; only
`KokkosProgramSampler` (and `sample_circuit(..., backend="gpu")`) require
`kokkos_sim`. All three `.so` install as top-level modules into `src/python/`,
so an editable install of this repo exposes them on the importer's path.

CPU and CUDA build recipes: [`docs/build.md`](docs/build.md). Rebuild all three
together, never just one: each statically links Kokkos, so mixing `.so` built
against different Kokkos installs in one process corrupts the shared runtime
state (segfault when the second extension is first used).

## Use

```python
import ksim

# one call: compile → draw noise → sample → (det, obs)
det, obs = ksim.sample_circuit(stim_text, n_shots, seed=42, backend="auto")

# or the explicit pieces (e.g. to reuse the compiled program):
flat, channel_probs, error_transform = ksim.compile(stim_text, sample_detectors=True)
ks = ksim.KokkosProgramSampler(flat)
f = ksim.ChannelSampler(channel_probs, error_transform, seed=42).sample(n_shots).astype("uint8")
out = ks.sample(f, seed=42)    # (B, n_outputs); det = out[:, :flat.num_detectors]
```

`example/sample_demo.py` runs this end to end (Clifford checked against stim, a
non-Clifford T circuit stim cannot touch, and the compile cache across a
p-sweep).

## Docs

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | The `FlatProgram` contract, the compile and sample halves, the `kokkos_decoder`/`kokkos_mcts` APIs, and the C++ conventions all three extensions share |
| [`docs/build.md`](docs/build.md) | CPU (OpenMP) and CUDA build recipes, Kokkos reinstall, smoke test |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Non-Clifford sampler deep dive (stim → tsim → kokkos_sim → clifft) and the decoder evolution log (kokkos vs nv-qldpc vs ldpc) |

Portions of the compile side are derived from Apache-2.0 third-party software —
see [`NOTICE`](NOTICE).
