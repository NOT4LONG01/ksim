#!/usr/bin/env python
"""example/sample_demo.py — the ZX engine end to end, on plain stim circuits.

Compile a stim circuit to a FlatProgram, draw noise, sample, and report
detector/observable outcomes — first on a Clifford circuit (where stim itself
is the exact reference to check against), then on a non-Clifford one with a T
layer (where stim cannot run at all, which is the whole point of the engine).

The GPU path is used when the kokkos_sim extension is built (see
docs/build.md); otherwise ksim falls back to the numpy reference, so this file
runs anywhere.

Run
---
    python example/sample_demo.py

Definitions
-----------
clifford_demo
    Sample a repetition-code memory circuit with ksim and with stim, and
    compare detector-firing rates — they must agree to sampling noise.

nonclifford_demo
    Sample a circuit with a T layer in an observable position, which stim's
    Clifford sampler cannot simulate, and report the measured P(1).

compile_demo
    The explicit compile → noise → sample pieces, plus what the structural
    compile cache does across a physical-error-rate sweep.
"""
import time

import numpy as np
import stim

import ksim


def clifford_demo(distance=3, rounds=3, p=0.01, shots=20_000, seed=42):
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=distance, rounds=rounds,
        before_round_data_depolarization=p)

    det_k, obs_k = ksim.sample_circuit(str(circuit), shots, seed=seed, backend="auto")
    det_s, obs_s = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)

    print(f"\nrepetition_code d={distance}, {rounds} rounds, p={p}, {shots} shots")
    print(f"  detector firing rate   ksim {det_k.mean():.4f}   stim {det_s.mean():.4f}")
    print(f"  observable flip rate   ksim {obs_k.mean():.4f}   stim {obs_s.mean():.4f}")


def nonclifford_demo(shots=20_000, seed=42):
    # The T-gate shorthand dialect: sample_circuit runs shorthand_to_stim on
    # text input, so "T 0" becomes stim's S[T] 0.
    circuit = """
        H 0
        T 0
        DEPOLARIZE1(0.01) 0
        MX 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """

    t0 = time.perf_counter()
    det, obs = ksim.sample_circuit(circuit, shots, seed=seed, backend="auto")
    elapsed = time.perf_counter() - t0

    # T|+> measured in X: <X> = cos(pi/4), shrunk by (1 - 4p/3) by the depolarizer.
    expected = (1 - np.cos(np.pi / 4) * (1 - 4 * 0.01 / 3)) / 2

    print(f"\nnon-Clifford (T in an X-basis measurement), {shots} shots")
    print(f"  P(1) = {det.mean():.4f}   expected {expected:.4f}   "
          f"[{elapsed / shots * 1e6:.1f} us/shot]")
    print("  stim cannot sample this circuit at all: T is not Clifford")


def compile_demo(p_values=(1e-3, 5e-3, 1e-2), shots=4_000, seed=42):
    print("\ncompile once, sample per p (the structural compile cache at work)")
    for p in p_values:
        circuit = stim.Circuit.generated(
            "repetition_code:memory", distance=3, rounds=3,
            before_round_data_depolarization=p)

        t0 = time.perf_counter()
        flat, channel_probs, error_transform = ksim.compile(
            str(circuit), sample_detectors=True)
        t_compile = time.perf_counter() - t0

        f = ksim.ChannelSampler(channel_probs, error_transform,
                                seed=seed).sample(shots).astype("uint8")
        out = ksim.sample_flat(flat, f, np.random.default_rng(seed))
        det = np.asarray(out)[:, :flat.num_detectors]

        print(f"  p={p:<7g} compile {t_compile * 1e3:7.1f} ms   "
              f"detector rate {det.mean():.4f}")


if __name__ == "__main__":
    clifford_demo()
    nonclifford_demo()
    compile_demo()
