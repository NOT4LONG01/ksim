#!/usr/bin/env python
"""benchmarks/bench_xtim.py — kokkos_sim vs xtim on non-Clifford circuits.

xtim (github.com/ikim-quantum/xtim, arXiv:2512.23799) is the fifth sampler in
`docs/benchmarks.md` Part 1. It is not another stabilizer-rank engine: it
exactly simulates any circuit that is Clifford-equivalent to **one
mutually-commuting layer of π/8 rotations** ("generalized T-depth 1"), and
prices a shot by the folded rank χ = 2^rank of that layer — not by T-count
(kokkos_sim/tsim) and not by active dimension (clifft). T-count is unlimited
inside the class; one step outside it and the circuit is refused outright.

So the head-to-head has two halves, and both are needed to read the numbers:

  * **inside xtim's class** — its own bundled protocol circuits (15-to-1
    distillation, d=3 cultivation, qRM↔Steane code switching). Both engines
    run them; compare µs/shot and detector fire rates.
  * **on the class boundary** — H·T·H·T·H, T-depth 2 after folding. xtim
    raises `XtimRejectError`; kokkos_sim samples it and is checked against the
    closed-form P(1). This is the axis that no throughput number captures.

Agreement is the gate `docs/benchmarks.md` puts ahead of speed: per-detector
fire rates from the two samplers must agree within shot noise, since both draw
from the noise model written into the same circuit text.

The one dialect edit: xtim's `PAULI_EXPECTATION` lines are dropped before
handing the text to `ksim.compile` (ksim samples detectors and observables; it
has no exact-expectation output). Everything else in the xtim dialect that
these circuits use is ordinary stim.

`--clifford` adds the Clifford memory rows of the throughput table, where xtim
is exercising the same tableau work as stim through a different engine and must
land at stim parity *and* reproduce stim's detector fire rates. Surface codes
come from `stim.Circuit.generated`; the colour codes need the consumer repo's
code registry on `PYTHONPATH` and are skipped without it.

Run (needs `pip install xtim`, plus the kokkos_sim build for --backend gpu):
    OMP_PROC_BIND=false python benchmarks/bench_xtim.py --shots 100000
    OMP_PROC_BIND=false python benchmarks/bench_xtim.py --json results.json
    PYTHONPATH=../soft-info-code-switch/src \
        OMP_PROC_BIND=false python benchmarks/bench_xtim.py --clifford

Definitions
-----------
gpu_name
    NVIDIA device name via nvidia-smi, or "unknown (no nvidia-smi)".

clifford_circuits
    The memory circuits of the throughput table — surface d=3/d=5 via
    `stim.Circuit.generated`, triangular/tetrahedral via the consumer registry —
    all at `rounds = d`, depolarizing p = 0.01.

bench_clifford
    stim vs xtim µs/shot on those circuits, plus xtim-vs-stim fire-rate
    agreement on a larger sample than the timing run uses.

strip_expectations
    Drop `PAULI_EXPECTATION` lines — the one xtim-only instruction in play.

run_xtim
    Build an xtim detector sampler, then time a warm `sample()` (build and the
    first reference-compiling call excluded), returning fire rates.

run_ksim
    Compile with `ksim.compile`, upload to `KokkosProgramSampler` (or the numpy
    `sample_flat` reference), time a warm `sample()`, return fire rates.

agreement
    Max per-detector fire-rate deviation between two runs, in units of the
    two-sample binomial sigma.

bench_class_boundary
    H·Tᵏ·H (accepted, exact P(1) = sin²(kπ/8)) and H·T·H·T·H (refused by xtim,
    sampled by kokkos_sim against a closed-form 2x2 reference).

bench_protocols
    Per-circuit head-to-head over xtim's bundled protocol examples.

main
    Parse args, run both halves, print the tables, optionally write JSON.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

# xtim's bundled protocols that ksim can also ingest. `ch_cultivation` is
# excluded on purpose: it is built on controlled-Hadamard, which xtim supports
# and ksim's gate table does not. `cube_ccz` is excluded because it declares no
# detectors at all — its whole output is the exact multi-magic expectation,
# which ksim has no counterpart for.
PROTOCOLS = [
    "miniature_oracle",
    "distillation_15_1_3",
    "cultivation_d3_rate",
    "cultivation_d3_faithful",
    "code_switching_rate",
    "code_switching_faithful",
]


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown (no nvidia-smi)"


def strip_expectations(text):
    return "\n".join(l for l in text.splitlines()
                     if not re.match(r"\s*PAULI_EXPECTATION", l)) + "\n"


def run_xtim(text, shots, seed=7):
    """Time an xtim detector sampler; raises if the circuit is out of class.

    Returns a dict of one-time cost, warm µs/shot, the engine `auto` picked
    (`twirl` is the fast path, `exact` the general one), the folded rank χ, and
    the per-detector fire rates.
    """
    import xtim
    circuit = xtim.Circuit(text)
    t0 = time.perf_counter()
    sampler = circuit.compile_detector_sampler(seed=seed)
    sampler.sample(min(shots, 4096))          # compiles + caches the reference
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    dets = np.asarray(sampler.sample(shots, separate_observables=True)[0])
    warm = (time.perf_counter() - t0) / shots * 1e6
    # the twirl engine can skip the reference cache, leaving reference_info()
    # without a chi — ask for the reference explicitly so the cost knob is
    # always reported.
    return {"build_s": build, "us_per_shot": warm,
            "engine": sampler.engine_report().get("engine"),
            "chi": circuit.compile_reference().chi,
            "n_detectors": dets.shape[1], "rates": dets.mean(axis=0)}


def run_ksim(text, shots, seed=42, backend="gpu"):
    """Time the ksim compile → sample path on the same circuit text."""
    import ksim
    t0 = time.perf_counter()
    flat, channel_probs, error_transform = ksim.compile(
        strip_expectations(text), sample_detectors=True, cache=None)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    if backend == "gpu":
        sampler = ksim.KokkosProgramSampler(flat)
        draw = lambda f, s: np.asarray(sampler.sample(f, seed=s))
    else:
        draw = lambda f, s: np.asarray(ksim.sample_flat(flat, f, seed=s))
    t_build = time.perf_counter() - t0

    channels = ksim.ChannelSampler(channel_probs, error_transform, seed=seed)
    f = channels.sample(min(shots, 4096)).astype("uint8")
    draw(f, seed)                              # warm the device / the caches
    f = channels.sample(shots).astype("uint8")
    t0 = time.perf_counter()
    out = draw(f, seed + 1)
    warm = (time.perf_counter() - t0) / shots * 1e6
    dets = out[:, :flat.num_detectors]
    return {"compile_s": t_compile, "build_s": t_build, "us_per_shot": warm,
            "graphs": sum(len(c.graphs) for c in flat.components),
            "n_detectors": flat.num_detectors, "rates": dets.mean(axis=0)}


def agreement(p_a, n_a, p_b, n_b):
    """Max |Δ fire rate| between two samplers, in two-sample sigmas."""
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    if p_a.shape != p_b.shape or p_a.size == 0:
        return float("nan"), float("nan")
    pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    sigma = np.sqrt(np.maximum(pool * (1 - pool), 1e-12) * (1 / n_a + 1 / n_b))
    d = np.abs(p_a - p_b)
    i = int(np.argmax(d / sigma))
    return float(d[i]), float(d[i] / sigma[i])


def _htkh(k):
    return "R 0\nH 0\n" + "T 0\n" * k + "H 0\nM 0\nDETECTOR rec[-1]\n"


def _exact_p1(text):
    """Closed-form P(M 0 = 1) for a single-qubit H/T word on |0>."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    psi = np.array([1, 0], dtype=complex)
    for line in text.splitlines():
        op = line.split()[0] if line.split() else ""
        if op == "H":
            psi = H @ psi
        elif op == "T":
            psi = T @ psi
    return float(abs(psi[1]) ** 2)


def bench_class_boundary(shots, backend):
    """H·Tᵏ·H inside the class; H·T·H·T·H outside it."""
    import xtim
    rows = []
    cases = [(f"H·T^{k}·H (in class)", _htkh(k)) for k in (1, 2, 3)]
    cases.append(("H·T·H·T·H (out of class)",
                  "R 0\nH 0\nT 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n"))

    for label, text in cases:
        exact = _exact_p1(text)
        row = {"circuit": label, "exact_p1": exact, "shots": shots}
        sigma = np.sqrt(max(exact * (1 - exact), 1e-12) / shots)

        try:
            r = run_xtim(text, shots)
            p1 = float(r["rates"][0])
            row["xtim"] = {"p1": p1, "us_per_shot": r["us_per_shot"],
                           "engine": r["engine"], "chi": r["chi"],
                           "sigmas": abs(p1 - exact) / sigma}
        except Exception as e:
            row["xtim"] = {"rejected": f"{type(e).__name__}: {str(e).splitlines()[0]}"}

        r = run_ksim(text, shots, backend=backend)
        p1 = float(r["rates"][0])
        row["ksim"] = {"p1": p1, "us_per_shot": r["us_per_shot"],
                       "sigmas": abs(p1 - exact) / sigma}
        rows.append(row)
    return rows


def bench_protocols(names, shots, backend):
    import xtim
    rows = []
    for name in names:
        text = xtim.load_example(name).text
        row = {"circuit": name, "shots": shots}
        try:
            r = run_xtim(text, shots)
            rates_x = r.pop("rates")
            row["xtim"] = r
        except Exception as e:
            row["xtim"] = {"rejected": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
            rates_x = None

        try:
            r = run_ksim(text, shots, backend=backend)
            rates_k = r.pop("rates")
            row["ksim"] = r
        except Exception as e:
            row["ksim"] = {"failed": f"{type(e).__name__}: {str(e)[:120]}"}
            rates_k = None

        if rates_x is not None and rates_k is not None:
            d, s = agreement(rates_x, shots, rates_k, shots)
            row["max_fire_rate_delta"] = d
            row["max_sigmas"] = s
        rows.append(row)
    return rows


CLIFFORD_CASES = [("surface d=3", "surface", 3), ("surface d=5", "surface", 5),
                  ("tri n=7", "triangular", 7), ("tri n=19", "triangular", 19),
                  ("tet n=15", "tetrahedral", 15)]
P_NOISE = 0.01


def clifford_circuits():
    """Yield (label, stim.Circuit) for the memory rows, at rounds = d, p = 0.01.

    Surface codes are self-contained; the colour codes come from the consumer
    repo's registry and are skipped (with a note) when it is not importable.
    """
    import stim
    for label, kind, size in CLIFFORD_CASES:
        if kind == "surface":
            yield label, stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=size, rounds=size,
                after_clifford_depolarization=P_NOISE,
                before_measure_flip_probability=P_NOISE,
                after_reset_flip_probability=P_NOISE)
            continue
        try:
            from circuit.registry import CodeRegistry
            from circuit.memory import generate_experiment_with_noise
        except ImportError:
            print(f"  {label}: skipped (needs the consumer repo's src/ on "
                  f"PYTHONPATH for the code registry)")
            continue
        code = CodeRegistry.load(kind, size)
        yield label, stim.Circuit(str(generate_experiment_with_noise(
            code.Hx, code.Hz, rounds=code.d, noise_model_name="depolarizing",
            noise_params={"p": P_NOISE})))


def bench_clifford(shots, agree_shots):
    import numpy as np
    import xtim
    rows = []
    for label, circ in clifford_circuits():
        stim_sampler = circ.compile_detector_sampler(seed=1)
        stim_sampler.sample(min(shots, 4096))
        t0 = time.perf_counter()
        stim_sampler.sample(shots)
        t_stim = (time.perf_counter() - t0) / shots * 1e6

        xtim_sampler = xtim.Circuit(str(circ)).compile_detector_sampler(seed=2)
        xtim_sampler.sample(min(shots, 4096))
        t0 = time.perf_counter()
        xtim_sampler.sample(shots, separate_observables=True)
        t_xtim = (time.perf_counter() - t0) / shots * 1e6

        p_s = np.asarray(stim_sampler.sample(agree_shots)).mean(axis=0)
        p_x = np.asarray(xtim_sampler.sample(
            agree_shots, separate_observables=True)[0]).mean(axis=0)
        _, sigmas = agreement(p_s, agree_shots, p_x, agree_shots)
        rows.append({"circuit": label, "qubits": circ.num_qubits,
                     "n_detectors": circ.num_detectors,
                     "stim_us_per_shot": t_stim, "xtim_us_per_shot": t_xtim,
                     "xtim_over_stim": t_xtim / t_stim,
                     "max_sigmas": sigmas,
                     "mean_abs_delta": float(np.abs(p_s - p_x).mean())})
    return rows


def _fmt(v, spec=".2f", missing="—"):
    return format(v, spec) if isinstance(v, (int, float)) else missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu",
                    help="gpu = KokkosProgramSampler, cpu = ksim.sample_flat")
    ap.add_argument("--circuits", nargs="*", default=PROTOCOLS)
    ap.add_argument("--skip-boundary", action="store_true")
    ap.add_argument("--clifford", action="store_true",
                    help="also run the Clifford memory rows against stim")
    ap.add_argument("--agree-shots", type=int, default=400_000,
                    help="shots for the xtim-vs-stim fire-rate check")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    try:
        import xtim
    except ImportError:
        print("xtim not installed: pip install xtim", file=sys.stderr)
        return 1

    result = {"when": datetime.now(timezone.utc).isoformat(),
              "gpu": gpu_name(), "backend": args.backend,
              "shots": args.shots, "xtim_version": xtim.__version__}

    if args.clifford:
        print(f"\nClifford memory rows — rounds = d, p = {P_NOISE}, "
              f"{args.shots} shots\n")
        rows = bench_clifford(args.shots, args.agree_shots)
        result["clifford"] = rows
        print(f"{'code':14s} {'qubits':>6s} {'dets':>5s} {'stim µs':>8s} "
              f"{'xtim µs':>8s} {'x/stim':>7s} {'max σ':>7s} {'mean |Δp|':>10s}")
        for r in rows:
            print(f"{r['circuit']:14s} {r['qubits']:6d} {r['n_detectors']:5d} "
                  f"{r['stim_us_per_shot']:8.2f} {r['xtim_us_per_shot']:8.2f} "
                  f"{r['xtim_over_stim']:7.1f} {r['max_sigmas']:7.1f} "
                  f"{r['mean_abs_delta']:10.5f}")

    if not args.skip_boundary:
        rows = bench_class_boundary(args.shots, args.backend)
        result["class_boundary"] = rows
        print(f"\nClass boundary — exact P(1), {args.shots} shots\n")
        print(f"{'circuit':24s} {'exact':>8s} {'xtim':>22s} {'kokkos_sim':>22s}")
        for r in rows:
            x = (r["xtim"].get("rejected", "")[:20] if "rejected" in r["xtim"]
                 else f"{r['xtim']['p1']:.5f} ({r['xtim']['sigmas']:.1f}σ)")
            k = f"{r['ksim']['p1']:.5f} ({r['ksim']['sigmas']:.1f}σ)"
            print(f"{r['circuit']:24s} {r['exact_p1']:8.5f} {x:>22s} {k:>22s}")

    rows = bench_protocols(args.circuits, args.shots, args.backend)
    result["protocols"] = rows
    print(f"\nxtim protocol circuits — {args.shots} shots, "
          f"{result['gpu']}, backend={args.backend}\n")
    print(f"{'circuit':26s} {'dets':>5s} {'χ':>3s} {'engine':>7s} {'xtim µs':>9s} "
          f"{'ksim µs':>9s} {'ratio':>7s} {'ksim cmp s':>11s} {'max σ':>7s}")
    for r in rows:
        x = r["xtim"].get("us_per_shot")
        k = r["ksim"].get("us_per_shot")
        ratio = k / x if isinstance(x, float) and isinstance(k, float) and x else None
        print(f"{r['circuit']:26s} "
              f"{r['ksim'].get('n_detectors', r['xtim'].get('n_detectors', 0)):5d} "
              f"{_fmt(r['xtim'].get('chi'), 'd'):>3s} "
              f"{str(r['xtim'].get('engine', '—')):>7s} "
              f"{_fmt(x, '.3f'):>9s} {_fmt(k, '.3f'):>9s} {_fmt(ratio, '.1f'):>7s} "
              f"{_fmt(r['ksim'].get('compile_s'), '.2f'):>11s} "
              f"{_fmt(r.get('max_sigmas'), '.1f'):>7s}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
