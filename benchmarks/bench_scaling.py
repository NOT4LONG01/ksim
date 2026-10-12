#!/usr/bin/env python
"""benchmarks/bench_scaling.py — how the non-Clifford samplers scale with independent magic states.

`bench_xtim.py` compares the two engines one circuit at a time, and on every
circuit xtim can build it wins per shot. This asks the scaling question instead:
take one of xtim's bundled single-magic protocols and place k copies of it side
by side on disjoint qubits, so the circuit carries k independent magic states.

  * **xtim** folds the π/8 layer into one reference state of folded rank k, so
    χ = 2^k. Its per-shot "twirl" path stays cheap, but the reference it
    compiles once grows with χ, and from some k on it does not finish.
  * **kokkos_sim** splits the circuit into its disconnected components and
    compiles each on its own, so k copies cost k times one copy: compile time,
    GPU upload and µs/shot all grow linearly in k.

Every engine is timed the same way, in a child process so a build or a run
that passes `--limit` seconds is killed and reported as such: its one-time setup
(compile, reference, upload, JIT and the first small draw) and its warm µs/shot.
The shot count adapts per engine, enough for ~`--seconds` of warm sampling but no
more than `--shots`, so a 1000 µs/shot engine does not stall the scan:

    xtim     CPU, 1 core   folded π/8 reference, χ = 2^k
    ksim     GPU           `ksim.compile` + `KokkosProgramSampler`
    tsim     GPU (JAX)     the same stabilizer-rank sum as ksim, complex64, XLA JIT
    clifft   CPU           factored statevector over the active subspace
    merlin   CPU           phase-polynomial sampler for CNOT+T circuits

Where an engine returns detectors, its per-detector fire rates are checked
against ksim's (`bench_xtim.agreement`): a speed means nothing until the
samplers draw the same distribution. merlin returns the raw measurement record,
which stim cannot convert for a circuit with T gates, so it is timed only.

Run (`.benchenv`: the soft-info-code-switch env plus bloqade-tsim and CUDA jax):
    OMP_PROC_BIND=false XLA_PYTHON_CLIENT_PREALLOCATE=false \
        .benchenv/bin/python benchmarks/bench_scaling.py --json benchmarks/results/scaling.json
    ... --engines xtim,ksim --k 1,4,8,12,14 --limit 300

Definitions
-----------
target_count
    The largest qubit index a circuit line names, plus one: the stride by which
    each copy is shifted.

shift
    One circuit's text with every qubit target moved by `offset`. `rec[...]`
    references are left alone: each copy is appended after the last, so its
    records are still the most recent ones its own detectors look back to.

tiled
    k copies of a bundled xtim example, `PAULI_EXPECTATION` lines (which ksim
    does not accept) and comments (clifft parses ASCII only) stripped.

ENGINES, warm, TSIM_BATCH, REPEATS
    {name: one engine's point}, and the shared warm-timing loop: a probe draw
    sizes the timed one to ~`seconds` of sampling, capped at `shots`. tsim is
    timed in whole batches of TSIM_BATCH, the shape its setup already compiled:
    XLA recompiles for every new batch size, and a timed draw of a new size
    would charge that compile to the shots. The timed draw is repeated
    REPEATS times (`--repeats`) and the fastest kept, since load from other
    jobs on a shared card only ever slows a draw.

xtim_point / ksim_point / tsim_point / clifft_point / merlin_point
    One k on one engine, in a child process: setup seconds, warm µs/shot,
    shots timed, fire rates where it returns detectors.

timed
    Run a point in a child process with a wall-clock limit; None if killed.

main
    Parse args, scan k, print the table, optionally write JSON. An engine that
    is killed, fails, or passes `--limit` seconds of setup or `--max-us` µs/shot
    at some k is not run at any larger k: its cost only grows with k, and the
    point would be off the plots anyway. `dropped` in the JSON says where.
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_xtim import agreement, gpu_name, run_ksim, strip_expectations  # noqa: E402

SKIP = ("#", "DETECTOR", "OBSERVABLE", "SHIFT_COORDS", "TICK", "REPEAT", "}")


def target_count(text):
    top = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(SKIP):
            continue
        for tok in s.split()[1:]:
            if not tok.startswith("rec["):
                top = max([top] + [int(d) for d in re.findall(r"\d+", tok)])
    return top + 1


def shift(text, offset):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(SKIP):
            out.append(line)
            continue
        head, *toks = s.split()
        toks = [t if t.startswith("rec[") else
                re.sub(r"\d+", lambda m: str(int(m.group()) + offset), t) for t in toks]
        out.append(" ".join([head] + toks))
    return "\n".join(out)


def tiled(name, k):
    import xtim
    base = "\n".join(l for l in strip_expectations(xtim.load_example(name).text).splitlines()
                     if not l.strip().startswith("#")) + "\n"
    stride = target_count(base)
    return "\n".join(shift(base, i * stride) for i in range(k)) + "\n"


REPEATS = 1


def warm(draw, shots, seconds, batch=None):
    probe = batch or 1000
    t0 = time.perf_counter()
    draw(probe)
    per = (time.perf_counter() - t0) / probe
    n = int(max(probe, min(shots, seconds / max(per, 1e-9))))
    best = None
    for _ in range(REPEATS):        # the fastest draw: a shared card's load only slows one
        if batch is None:
            t0 = time.perf_counter()
            out = draw(n)
            us, m = (time.perf_counter() - t0) / n * 1e6, n
        else:
            reps = -(-n // batch)   # whole batches of the shape already compiled
            t0 = time.perf_counter()
            out = np.concatenate([draw(batch) for _ in range(reps)])
            us, m = (time.perf_counter() - t0) / (reps * batch) * 1e6, reps * batch
        if best is None or us < best[0]:
            best = (us, m, out)
    return best


def xtim_point(text, shots, seconds, q):
    import xtim
    t0 = time.perf_counter()
    circuit = xtim.Circuit(text)
    chi = circuit.compile_reference().chi
    sampler = circuit.compile_detector_sampler(seed=7)
    sampler.sample(256)
    setup = time.perf_counter() - t0
    us, n, dets = warm(lambda m: np.asarray(sampler.sample(m, separate_observables=True)[0]),
                       shots, seconds)
    q.put({"chi": chi, "setup_s": setup, "us_per_shot": us, "shots": n,
           "rates": dets.mean(axis=0).tolist()})


def ksim_point(text, shots, seconds, q):
    import ksim
    t0 = time.perf_counter()
    flat, probs, transform = ksim.compile(text, sample_detectors=True, cache=None)
    sampler = ksim.KokkosProgramSampler(flat)
    channels = ksim.ChannelSampler(probs, transform, seed=42)
    sampler.sample(channels.sample(256).astype("uint8"), seed=1)
    setup = time.perf_counter() - t0
    us, n, out = warm(lambda m: np.asarray(sampler.sample(channels.sample(m).astype("uint8"), seed=2)),
                      shots, seconds)
    q.put({"setup_s": setup, "us_per_shot": us, "shots": n,
           "rates": out[:, :flat.num_detectors].mean(axis=0).tolist()})


def tsim_point(text, shots, seconds, q):
    import tsim
    t0 = time.perf_counter()
    sampler = tsim.Circuit(text).compile_detector_sampler(seed=7)
    sampler.sample(TSIM_BATCH, separate_observables=True)     # XLA compiles this shape
    setup = time.perf_counter() - t0
    us, n, dets = warm(lambda m: np.asarray(sampler.sample(m, separate_observables=True)[0]),
                       shots, seconds, batch=TSIM_BATCH)
    q.put({"setup_s": setup, "us_per_shot": us, "shots": n,
           "rates": dets.mean(axis=0).tolist()})


def clifft_point(text, shots, seconds, q):
    import clifft
    t0 = time.perf_counter()
    prog = clifft.compile(text)
    clifft.sample(prog, 256, seed=1)
    setup = time.perf_counter() - t0
    us, n, res = warm(lambda m: clifft.sample(prog, m, seed=2), shots, seconds)
    q.put({"setup_s": setup, "us_per_shot": us, "shots": n,
           "rates": np.asarray(res.detectors).mean(axis=0).tolist()})


def merlin_point(text, shots, seconds, q):
    import merlin
    t0 = time.perf_counter()
    sampler = merlin.CircuitSampler(text, seed=7)
    sampler.sample(256)
    setup = time.perf_counter() - t0
    us, n, _ = warm(lambda m: sampler.sample(m), shots, seconds)
    q.put({"setup_s": setup, "us_per_shot": us, "shots": n})


TSIM_BATCH = 4096
ENGINES = {"xtim": xtim_point, "ksim": ksim_point, "tsim": tsim_point,
           "clifft": clifft_point, "merlin": merlin_point}


def _run(fn, repeats, *args):
    global REPEATS
    REPEATS = repeats
    fn(*args)


def timed(fn, text, shots, seconds, limit, repeats=1):
    q = mp.get_context("spawn").Queue()
    p = mp.get_context("spawn").Process(target=_run, args=(fn, repeats, text, shots, seconds, q))
    p.start()
    p.join(limit)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    try:
        return q.get(timeout=5)
    except Exception:
        return {"failed": f"exit code {p.exitcode}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", default="miniature_oracle",
                    help="bundled xtim example to tile (one magic state per copy)")
    ap.add_argument("--k", default="1,2,4,6,8,10,12,13,14,15,16")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--shots", type=int, default=100_000,
                    help="most shots an engine is timed on")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="warm sampling time an engine is given, within --shots")
    ap.add_argument("--limit", type=float, default=600.0,
                    help="seconds an engine may take on one k before it is killed")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed draws per point; the fastest is kept (a shared GPU)")
    ap.add_argument("--max-us", type=float, default=1000.0, dest="max_us",
                    help="µs/shot past which an engine leaves the scan")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    engines = [e for e in args.engines.split(",") if e]
    dropped = {}

    import xtim
    result = {"when": datetime.now(timezone.utc).isoformat(), "gpu": gpu_name(),
              "xtim_version": xtim.__version__, "circuit": args.circuit, "engines": engines,
              "shots": args.shots, "seconds": args.seconds, "limit_s": args.limit, "rows": []}
    print(f"{args.circuit} x k on disjoint qubits, {result['gpu']}, limit {args.limit:.0f} s\n")
    print(f"{'k':>3s} {'engine':>7s} {'setup s':>9s} {'µs/shot':>10s} {'shots':>8s} {'σ vs ksim':>9s}")
    for k in (int(x) for x in args.k.split(",")):
        text = tiled(args.circuit, k)
        row = {"k": k, "qubits": target_count(text)}
        for e in engines:
            if e in dropped:
                continue
            row[e] = timed(ENGINES[e], text, args.shots, args.seconds, args.limit, args.repeats)
            r = row[e]
            # an engine that left the plotted range at this k only gets slower with k
            if r is None:
                dropped[e] = f"killed at k = {k}"
            elif "failed" in r:
                dropped[e] = f"failed at k = {k}"
            elif r["setup_s"] > args.limit or r["us_per_shot"] > args.max_us:
                dropped[e] = f"left the range at k = {k}"
        ref = row.get("ksim")
        for e in engines:
            r = row.get(e)
            if r and e != "ksim" and ref and "rates" in r and "rates" in ref \
                    and len(r["rates"]) == len(ref["rates"]):
                _, r["sigmas_vs_ksim"] = agreement(np.asarray(r["rates"]), r["shots"],
                                                   np.asarray(ref["rates"]), ref["shots"])
        for e in engines:
            if e not in row:
                continue
            r = row[e]
            if r is None:
                print(f"{k:3d} {e:>7s} {f'> {args.limit:.0f} (killed)':>20s}", flush=True)
            elif "failed" in r:
                print(f"{k:3d} {e:>7s} failed: {r['failed']}", flush=True)
            else:
                sg = f"{r['sigmas_vs_ksim']:9.1f}" if "sigmas_vs_ksim" in r else f"{'—':>9s}"
                print(f"{k:3d} {e:>7s} {r['setup_s']:9.2f} {r['us_per_shot']:10.3f} "
                      f"{r['shots']:8d} {sg}" + (f"   χ={r['chi']}" if "chi" in r else ""), flush=True)
            if r:
                r.pop("rates", None)
        result["rows"].append(row)

    result["dropped"] = dropped
    for e, why in dropped.items():
        print(f"{e}: {why}, not run past it")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
