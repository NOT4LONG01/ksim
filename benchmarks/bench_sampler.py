#!/usr/bin/env python
"""benchmarks/bench_sampler.py — per-shot throughput of kokkos_sim, with an
optional apples-to-apples slot for an external simulator (e.g. SOFT).

The dev machine is CPU-only, so the fair GPU comparison against SOFT
(arXiv:2512.23037, generalized stabilizer tableau) cannot be run there. This
harness is what to run on a GPU node — it produces the numbers the benchmark
doc currently only cites.

Two metrics per circuit:
  * warm — sample-only µs/shot, compile excluded, swept over batch size. This is
           the engine's steady-state throughput and matches SOFT's Fig. 3
           (throughput vs batch_size on one GPU).
  * e2e  — end-to-end µs/shot for a fixed shot budget (compile + noise draw +
           sample). This matches SOFT's Table V "average time per shot", the
           only metric an external tool can be timed against from outside.

The external simulator is timed by wall-clock of a user-supplied command on the
*same* circuit and shot count, so no assumption is made about its output format.
Because that wall-clock includes process startup, use a large --external-shots
so startup amortizes; the harness prints the shot count it divided by.

Fairness checklist (enforce when reading results, not automatable here):
  * same GPU model for both tools (record from nvidia-smi, printed below);
  * same circuit file, same total shots, double precision on both;
  * warm kokkos_sim numbers exclude compile — SOFT's Table V does not, so
    compare SOFT against the e2e column, not warm.

Run:
    python benchmarks/bench_sampler.py --circuit msc_d3.stim --backend gpu \
        --batch-sizes 4096 65536 262144 524288 --e2e-shots 1000000 \
        --out results/msc_d3.json
    # add an external tool (SOFT), timed end-to-end on the same circuit:
    python benchmarks/bench_sampler.py --circuit msc_d3.stim --backend gpu \
        --external-shots 2000000 \
        --external-cmd 'soft-cli --circuit {circuit} --shots {shots} --gpu'

Definitions
-----------
gpu_name
    NVIDIA device name via nvidia-smi, or "unknown (no nvidia-smi)".

load_circuit
    Read a stim circuit file (or '-' for stdin) into text.

bench_kokkos_warm
    Sample-only µs/shot for kokkos_sim, per batch size (compile excluded).

bench_kokkos_e2e
    End-to-end µs/shot for kokkos_sim over a fixed shot budget (compile included).

bench_stim
    Reference: stim's native detector sampler µs/shot (Clifford circuits only).

bench_external
    Wall-clock µs/shot of an external simulator command on the same circuit.

main
    Parse args, run the requested benchmarks, print a table, write JSON.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True)
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "unknown (no nvidia-smi)"


def load_circuit(path):
    if path == "-":
        return sys.stdin.read()
    with open(path) as fh:
        return fh.read()


def bench_kokkos_warm(text, batch_sizes, repeats, seed, backend):
    import ksim
    t0 = time.perf_counter()
    flat, channel_probs, error_transform = ksim.compile(text, sample_detectors=True)
    compile_s = time.perf_counter() - t0
    ks = ksim.KokkosProgramSampler(flat, backend=backend) if _accepts_backend(ksim) \
        else ksim.KokkosProgramSampler(flat)

    rows = []
    for b in batch_sizes:
        f = ksim.ChannelSampler(channel_probs, error_transform,
                                seed=seed).sample(b).astype("uint8")
        ks.sample(f, seed=seed)  # warmup (JIT / first-launch cost off the clock)
        t0 = time.perf_counter()
        for _ in range(repeats):
            ks.sample(f, seed=seed)
        dt = (time.perf_counter() - t0) / repeats
        rows.append({"batch": b, "us_per_shot": dt / b * 1e6,
                     "shots_per_s": b / dt, "wall_s_per_call": dt})
    return {"compile_s": compile_s, "num_detectors": int(flat.num_detectors),
            "batches": rows}


def bench_kokkos_e2e(text, shots, seed, backend):
    import ksim
    ksim.sample_circuit(text, min(shots, 1024), seed=seed, backend=backend)  # warm compile cache
    t0 = time.perf_counter()
    det, obs = ksim.sample_circuit(text, shots, seed=seed, backend=backend)
    dt = time.perf_counter() - t0
    return {"shots": shots, "us_per_shot": dt / shots * 1e6,
            "shots_per_s": shots / dt, "wall_s": dt,
            "det_fire_rate": float(np.asarray(det).mean())}


def bench_stim(text, shots, seed):
    import stim
    circ = stim.Circuit(text)
    sampler = circ.compile_detector_sampler(seed=seed)
    sampler.sample(min(shots, 1024), separate_observables=True)  # warmup
    t0 = time.perf_counter()
    det, obs = sampler.sample(shots, separate_observables=True)
    dt = time.perf_counter() - t0
    return {"shots": shots, "us_per_shot": dt / shots * 1e6,
            "shots_per_s": shots / dt, "wall_s": dt}


def bench_external(cmd_template, circuit_path, shots):
    cmd = cmd_template.format(circuit=shlex.quote(circuit_path), shots=shots)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, shell=True)
    dt = time.perf_counter() - t0
    return {"cmd": cmd, "returncode": proc.returncode, "shots": shots,
            "wall_s": dt, "us_per_shot": dt / shots * 1e6,
            "shots_per_s": shots / dt,
            "note": "wall-clock incl. process startup; use large --external-shots"}


def _accepts_backend(ksim):
    import inspect
    try:
        return "backend" in inspect.signature(ksim.KokkosProgramSampler).parameters
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", required=True, help="stim circuit file, or '-' for stdin")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu", "auto"])
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[4096, 16384, 65536, 262144, 524288])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--e2e-shots", type=int, default=1_000_000)
    ap.add_argument("--stim-baseline", action="store_true",
                    help="also time stim's native sampler (Clifford circuits only)")
    ap.add_argument("--external-cmd", default=None,
                    help="shell template for an external simulator; "
                         "{circuit} and {shots} are substituted")
    ap.add_argument("--external-shots", type=int, default=2_000_000)
    ap.add_argument("--out", default=None, help="write full results as JSON here")
    args = ap.parse_args()

    text = load_circuit(args.circuit)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name(),
        "backend": args.backend,
        "circuit": args.circuit,
        "seed": args.seed,
    }

    print(f"GPU: {result['gpu']}   backend: {args.backend}   circuit: {args.circuit}\n")

    result["kokkos_warm"] = bench_kokkos_warm(
        text, args.batch_sizes, args.repeats, args.seed, args.backend)
    print(f"kokkos_sim (compile {result['kokkos_warm']['compile_s']*1e3:.1f} ms, "
          f"{result['kokkos_warm']['num_detectors']} detectors)")
    print(f"  {'batch':>8}  {'us/shot':>10}  {'shots/s':>12}")
    for r in result["kokkos_warm"]["batches"]:
        print(f"  {r['batch']:>8}  {r['us_per_shot']:>10.3f}  {r['shots_per_s']:>12,.0f}")

    result["kokkos_e2e"] = bench_kokkos_e2e(text, args.e2e_shots, args.seed, args.backend)
    e = result["kokkos_e2e"]
    print(f"\nkokkos_sim end-to-end ({e['shots']:,} shots): "
          f"{e['us_per_shot']:.3f} us/shot  [{e['shots_per_s']:,.0f} shots/s]")

    if args.stim_baseline:
        try:
            result["stim"] = bench_stim(text, args.e2e_shots, args.seed)
            print(f"stim baseline ({args.e2e_shots:,} shots): "
                  f"{result['stim']['us_per_shot']:.3f} us/shot")
        except Exception as exc:
            result["stim"] = {"error": str(exc)}
            print(f"stim baseline: skipped ({exc})")

    if args.external_cmd:
        result["external"] = bench_external(args.external_cmd, args.circuit,
                                            args.external_shots)
        x = result["external"]
        status = "ok" if x["returncode"] == 0 else f"FAILED rc={x['returncode']}"
        print(f"\nexternal ({x['shots']:,} shots, {status}): "
              f"{x['us_per_shot']:.3f} us/shot wall-clock  [{x['note']}]")
        if x["returncode"] == 0:
            print(f"  compare against kokkos_sim e2e {e['us_per_shot']:.3f} us/shot "
                  f"(same GPU + shots + precision required for this to be fair)")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
