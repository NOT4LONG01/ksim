#!/usr/bin/env python
"""benchmarks/plot_scaling.py — the figures of `bench_scaling.py`'s JSON.

One large figure per question, every measured point a marker, one colour per
engine (tab10, fixed per engine):

    example/figure/scaling_setup.png     one-time cost against k. An engine killed
                                         at the limit is an open triangle at the
                                         limit: a lower bound, not a value.
    example/figure/scaling_per_shot.png  warm µs/shot against k.
    example/figure/scaling_total.png     wall time for N shots, setup included,
                                         against k, at N = 1e6 (solid) and 1e8
                                         (dotted).
    example/figure/scaling_wall_1e6.png  the same at N = 1e6 alone, every engine, ksim
                                         drawn heavier; a killed build is an open
                                         triangle at the limit (a lower bound).
    example/figure/scaling_pareto_k<k>.png
                                         at one k, every engine as a point in the
                                         (setup, µs/shot) plane; the Pareto front
                                         -- the engines no other beats on both --
                                         is joined by a line.

Run:
    python benchmarks/plot_scaling.py benchmarks/results/scaling_h100.json [--pareto 1,8,14,16]

Definitions
-----------
COLOURS, LABELS    Per engine: colour, and the label with its device and short reference.
SHOT_COUNTS        The N of the total-time figure.
WALL_SHOTS         The N of the single-budget wall-time figure.
frame              A large framed axes with title and labels, log y.
save               Legend outside, write at 300 dpi, close.
measured / killed  The (k, row) where an engine produced a point, or was killed.
pareto_front       The non-dominated points of a list of (setup, µs/shot, engine).
main               Read the JSON, draw every figure.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"xtim": "#1f77b4", "ksim": "#ff7f0e", "tsim": "#2ca02c",
           "clifft": "#d62728", "merlin": "#9467bd"}
LABELS = {"xtim": "xtim (CPU, 1 core) [Surti2025]",
          "ksim": "ksim (GPU) [ours]",
          "tsim": "tsim (GPU, JAX) [Haenel2026]",
          "clifft": "clifft (CPU) [Chase2026]",
          "merlin": "merlin (CPU) [Cercelescu2026]"}
# Surti2025       arXiv:2512.23799   Surti, Daguerre, Kim, "Efficient simulation of logical
#                                    magic state preparation protocols"
# Haenel2026      arXiv:2604.01059   Haenel, Luo, Zhao, "Tsim: Fast Universal Simulator for
#                                    Quantum Error Correction"
# Chase2026       arXiv:2604.27058   Chase, Labib, "Clifft: Fast Exact Simulation of
#                                    Near-Clifford Quantum Circuits"
# Cercelescu2026  arXiv:2610.06811   Cercelescu, Koch, Pesah, "Polynomial-time simulation of
#                                    non-Clifford quantum error correction"
SHOT_COUNTS = ((10**6, "-"), (10**8, ":"))
WALL_SHOTS = 10**6
POINT = dict(marker="o", ms=6, mec="black", mew=0.6, lw=1.5)
XLABEL = "k independent magic states (copies on disjoint qubits)"


def frame(title, xlabel, ylabel, logx=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(True, color="black", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.tick_params(labelsize=10)
    return fig, ax


def save(fig, ax, path):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10,
              frameon=True, fancybox=False, edgecolor="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def measured(rows, engine):
    return [(r["k"], r[engine]) for r in rows
            if r.get(engine) and "us_per_shot" in r[engine]]


def killed(rows, engine):
    return [r["k"] for r in rows if engine in r and r[engine] is None]


def pareto_front(points):
    front = []
    for s, u, e in sorted(points):
        if not front or u < front[-1][1]:
            front.append((s, u, e))
    return front


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("--pareto", default=None, help="k values for the Pareto figures")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.json).read_text())
    out = Path(__file__).resolve().parent.parent / "example" / "figure"
    rows, limit, engines = data["rows"], data["limit_s"], data["engines"]
    head = f"{data['circuit']} × k"

    def killed_marks(ax, e, y):
        ks = killed(rows, e)
        if ks:
            ax.plot(ks, [y] * len(ks), ls="none", marker="^", ms=9, mfc="white",
                    mec=COLOURS[e], mew=1.6, label=f"{e} killed at {limit:.0f} s")

    fig, ax = frame(f"One-time cost ({head})", XLABEL, "seconds")
    for e in engines:
        pts = measured(rows, e)
        if pts:
            ax.plot([k for k, _ in pts], [r["setup_s"] for _, r in pts],
                    color=COLOURS[e], label=LABELS[e], **POINT)
        killed_marks(ax, e, limit)
    save(fig, ax, out / "scaling_setup.png")

    fig, ax = frame(f"Per-shot cost, warm ({head})", XLABEL, "µs / shot")
    for e in engines:
        pts = measured(rows, e)
        if pts:
            ax.plot([k for k, _ in pts], [r["us_per_shot"] for _, r in pts],
                    color=COLOURS[e], label=LABELS[e], **POINT)
    save(fig, ax, out / "scaling_per_shot.png")

    fig, ax = frame(f"Wall time for N shots, setup included ({head})", XLABEL, "seconds")
    for n, ls in SHOT_COUNTS:
        for e in engines:
            pts = measured(rows, e)
            if pts:
                ax.plot([k for k, _ in pts],
                        [r["setup_s"] + n * r["us_per_shot"] * 1e-6 for _, r in pts],
                        color=COLOURS[e], ls=ls, label=f"{e}, N = {n:.0e}", **POINT)
    save(fig, ax, out / "scaling_total.png")

    n = WALL_SHOTS
    fig, ax = frame(f"Wall time for $10^{{{len(str(n)) - 1}}}$ shots, setup included ({head})", XLABEL, "seconds")
    for e in engines:
        pts = measured(rows, e)
        if pts:
            ax.plot([k for k, _ in pts],
                    [r["setup_s"] + n * r["us_per_shot"] * 1e-6 for _, r in pts],
                    color=COLOURS[e], label=LABELS[e],
                    **{**POINT, **({"lw": 2.5, "ms": 8, "zorder": 3} if e == "ksim" else {})})
        killed_marks(ax, e, limit)
    save(fig, ax, out / "scaling_wall_1e6.png")

    ks = ([int(x) for x in args.pareto.split(",")] if args.pareto
          else [r["k"] for r in rows][::max(1, len(rows) // 4)])
    for k in ks:
        row = next((r for r in rows if r["k"] == k), None)
        if row is None:
            continue
        pts = [(row[e]["setup_s"], row[e]["us_per_shot"], e) for e in engines
               if row.get(e) and "us_per_shot" in row[e]]
        if not pts:
            continue
        fig, ax = frame(f"Setup against per-shot cost, k = {k} ({row['qubits']} qubits)",
                        "one-time setup (s)", "µs / shot", logx=True)
        front = pareto_front(pts)
        ax.plot([s for s, _, _ in front], [u for _, u, _ in front], color="black",
                lw=1, ls="--", zorder=1, label="Pareto front")
        for s, u, e in pts:
            on = any(e == f for _, _, f in front)
            ax.plot([s], [u], ls="none", marker="o", ms=10 if on else 7, color=COLOURS[e],
                    mec="black", mew=1.2 if on else 0.6, label=LABELS[e] + ("" if on else " (dominated)"))
        for e in engines:
            if e in row and row[e] is None:
                ax.plot([limit], [min(u for _, u, _ in pts)], ls="none", marker=">", ms=9,
                        mfc="white", mec=COLOURS[e], mew=1.6,
                        label=f"{e}: setup > {limit:.0f} s (killed)")
        save(fig, ax, out / f"scaling_pareto_k{k}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
