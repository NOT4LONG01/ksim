# Decoder verification: kokkos_decoder against ldpc and Relay-BP

What `kokkos_decoder` is checked against, what the checks show, and the
defects those checks found and removed in September 2026. The reference
points are Roffe's `ldpc` (`BpOsdDecoder`, product-sum, OSD-0) for BP+OSD and
IBM's `relay-bp` package (github.com/trmue/relay, arXiv:2506.01779) for
Relay-BP. Every number below is 20 000 or 40 000 stim shots at p = 5e-3 on the
graphlike (`decompose_errors=True`) model, decoded on one H100, unless stated.

## BP+OSD-0 against ldpc

The BP is the same algorithm. A float64 numpy replica of product-sum BP
reproduces ldpc's posteriors to 1e-13 and its per-shot convergence verdicts
exactly; float32 with kokkos's message clamps changes nothing that matters.

| model | ldpc LER | kokkos LER | identical predictions | convergence flags agree |
|---|---|---|---|---|
| triangular n=19 | 0.1319 | 0.1319 | 99.98 % | 100 % |
| tetrahedral n=15 | 0.0241 | 0.0239 | 84 % (see below) | 94 % |
| triangular n=7 | 0.0662 | 0.0866 | 96.7 % | 100 % |

Where the two differ, BP did not converge and OSD-0 decided. OSD-0 orders the
columns by posterior LLR and takes the first independent set as pivots, so the
order *among tied posteriors* is part of the algorithm, and a non-converged
shot carries many ties: symmetric bits far from the syndrome receive identical
messages. On tetrahedral n=15 ldpc's doubles saturate to +inf on about half the
columns (kokkos clamps messages at ±15), so the two order that half
differently and still land on the same LER. On triangular n=7 the tie order
is a logical coin flip: the model holds 156 pairs of columns whose detector
patterns compose to a third column with a different observable, 43 % of ldpc's
pivots sit inside a tie group, and ldpc's stable sort of its own rounding noise
happens to pick the pair rather than the single mechanism. Adding 1e-13 of
noise to ldpc's own posteriors moves its 576 failures to 985; an 80-bit
replica gives 827. No deterministic rule reproduces that, and float32 cannot
see differences at that scale at all, so kokkos and ldpc disagree there by
construction. Compare the two on tetrahedral n=15 or triangular n=19, never on
triangular n=7 alone.

kokkos's rule is the one in `osd0_run_all`: posteriors quantised to
`OSD_TIE_TOL = 1e-3`, ties to the likelier channel prior, then the lower column
index. The quantisation is what makes the output identical run to run and
across batch sizes despite the float atomics in the message sums.

## Relay-BP against the reference implementation

`RelayBpDecoder` carries the paper's names (`R`, `T0`, `Tr`, `gamma0`,
`gamma_center`/`gamma_width`, `S`) and its defaults are the paper's gross-code
values. Tetrahedral n=15, p = 5e-3:

| setting | reference `relay-bp` | kokkos |
|---|---|---|
| `gamma0=0.1, R=301, Tr=60, S=1` | 0.00090 | 0.00090 |
| `gamma0=0.1, R=301, Tr=60, S=5` | 0.00040 | 0.00045, same ms/shot |
| plain-BP first leg, `R=21, Tr=8, S=1` | 0.00140 (`sets=20`) | 0.00080 |

The remaining difference is the BP kernel (the reference is min-sum, kokkos
product-sum) and the γ draw (per shot and variable here, one vector per leg
there). Two properties worth knowing: `S` is the cheap knob (at p = 5e-3,
`S=5` cuts the LER about three-fold at any leg setting, for no extra time,
since the legs run anyway; `S=9` adds little; at p = 1e-2 the gain needs the
paper's long legs), and `gamma0` is code-specific as the paper says (at `S=1`
on tetrahedral n=15 it swings the LER two-fold across {0, 0.05, 0.1, 0.125,
0.2}, with the gross-code 0.125 the worst; at `S=5` every value lands within
noise).

## Backends

The OpenMP and CUDA builds of the same source give the same answer. On 2048
surface d=3 shots (`stim` rotated memory, p = 0.01, 30 iterations), the
OpenMP `BpOsd0Decoder` reports converged on 91.31 % of shots, ldpc on
91.31 %, the flags agree on every shot and the predictions are identical; the
CUDA build reports the same. A kokkos build that reports converged on 100 % of
those shots is a build of the source before commit ed2d9bf, whatever names its
binding accepts: the three files a parameter rename touches
(`bp_decoders.hpp`, `kokkos_decoder.cpp`, `relay_bp.cpp`) are not the files
that fix the flag (`bp_osd.cpp`, `bp_lsd.cpp`, `bp_infra.cpp`), so a tree with
the first three copied over an old checkout compiles, accepts the new
arguments and still lies about convergence. The arbiter is one script:

```python
import numpy as np, stim
from ldpc.bposd_decoder import BpOsdDecoder
from kokkos_decoder import BpOsd0Decoder
circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3,
                                 rounds=3, after_clifford_depolarization=0.01)
dem = circuit.detector_error_model()
# H (nc x nb, uint8), p (nb,) from the DEM: soft-info-code-switch's
# circuit.stim_tools.dem_to_parity_check, or any equivalent
det, _ = circuit.compile_detector_sampler(seed=7).sample(2048, separate_observables=True)
det = det.astype(np.uint8)
ref = BpOsdDecoder(H, error_channel=list(p), max_iter=30, bp_method="product_sum",
                   osd_method="osd_0", osd_order=0)
conv_ref = np.array([(ref.decode(s), ref.converge)[1] for s in det])
_, conv = BpOsd0Decoder(H, p.astype(np.float32), 30, 512).decode_batch(det[:512])
print(conv.mean(), conv_ref[:512].mean())   # 0.9131 twice on a current build
```

## The defects the checks found

All in `src/cpp/decoder/`, fixed in commit ed2d9bf and the commit that carries
this document.

- **`converged` was always 1.** Every decoder overwrote the flag after OSD-0,
  LSD or the relay had run. It is now BP's own verdict per shot (for the relay:
  whether any leg converged); where it is 0 the fallback supplied the
  prediction. The kernel that clears it is one writer per shot, not one per
  (shot, check), so no two threads ever store to the same byte.
- **OSD-0's tie order was float-atomic noise.** The same batch decoded
  differently on 3 % of shots between two identical calls, and between batch
  sizes. The tie rule above replaced it.
- **`stop_nconv` was a batch-level early stop**, ending the legs once fewer
  than that many shots in the batch remained unconverged, and the relay kept
  the first converged solution. It is now the paper's per-shot `S`: collect
  `S` converged solutions, return the lowest prior weight. The relay also
  gained the paper's first-leg memory (`gamma0`) and restarts its messages
  from the priors every leg, keeping only the previous posterior as memory.
- **Parameter names** followed no source; they now follow the paper.

## What still differs from ldpc on purpose

ldpc's `BpOsdDecoder` in the consumer repo defaults to 100 iterations and an
order-20 combination sweep; `kokkos:bp_osd` is BP + OSD-0 with its own
defaults in the binding. A kokkos-vs-ldpc comparison has to hold both at
OSD-0 and the same `max_iter`, as the test in the consumer repo's
`tests/test_kokkos.py` does.
