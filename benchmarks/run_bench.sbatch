#!/bin/bash
#SBATCH --job-name=ksim-soft-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # site-specific; one GPU is enough (both tools are shot-parallel on one device)
#SBATCH --cpus-per-task=8
#SBATCH --output=bench-%j.log
# No #SBATCH --time here — set it to your cluster's policy if it requires one.
#
# ============================================================================
# BENCHMARK PLAN — kokkos_sim (ksim) vs SOFT on one GPU
# ============================================================================
# The dev cluster is CPU-only, so the fair GPU head-to-head against SOFT
# (arXiv:2512.23037) is documented as "cited, not re-run". This job produces
# the real numbers. Run it on a GPU node; it does not touch the docs.
#
# What it measures (see benchmarks/bench_sampler.py for the definitions):
#   * kokkos_sim warm  — sample-only us/shot swept over batch size (SOFT Fig. 3)
#   * kokkos_sim e2e   — end-to-end us/shot at a fixed shot budget (SOFT Table V)
#   * SOFT             — wall-clock us/shot of the SOFT binary on the SAME
#                        circuit + shot count (the only externally-timable metric)
#
# Prerequisites (do once, on this GPU node):
#   1. Build the three Kokkos extensions against CUDA and install ksim:
#        see docs/build.md (CUDA section). Verify: python -c "import kokkos_sim".
#   2. Obtain matched circuits as stim text. The head-to-head circuits are the
#        magic-state-cultivation (MSC) circuits SOFT reports on (d=3, d=5 / 42q).
#        Get them from SOFT's own repo (arXiv:2512.23037 links its GitHub), or
#        any Clifford+T stim file you also feed to SOFT. Set CIRCUIT_D3/CIRCUIT_D5.
#   3. Build SOFT per its repo and set SOFT_CMD to its run command. Use {circuit}
#        and {shots} placeholders — the harness substitutes them and times the
#        whole process by wall-clock (so make --external-shots large to amortize
#        startup). Leave SOFT_CMD empty to benchmark kokkos_sim alone.
#
# Fairness (enforced by you, not the script): same GPU model for both tools,
# same circuit file, same shot count, double precision on both. Compare SOFT's
# wall-clock against the kokkos_sim *e2e* column (both include compile/startup),
# not the warm column.
# ============================================================================

set -euo pipefail

# --- site config: edit these -------------------------------------------------
module load cuda/12 2>/dev/null || true          # names are site-specific
source "${KSIM_ENV:-$HOME/work/ksim/env}/bin/activate"

CIRCUIT_D3="${CIRCUIT_D3:-circuits/msc_cultivation_d3.stim}"
CIRCUIT_D5="${CIRCUIT_D5:-circuits/msc_cultivation_d5.stim}"
SOFT_CMD="${SOFT_CMD:-}"                          # e.g. 'soft-cli --circuit {circuit} --shots {shots} --gpu --fp64'
E2E_SHOTS="${E2E_SHOTS:-2000000}"
EXTERNAL_SHOTS="${EXTERNAL_SHOTS:-2000000}"
BATCHES="${BATCHES:-4096 16384 65536 262144 524288}"
# ----------------------------------------------------------------------------

RESULTS="results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS"
cd "$(dirname "$0")/.."                           # repo root

python -c "import kokkos_sim" \
  || { echo "kokkos_sim (CUDA) not importable — build it first (docs/build.md)"; exit 1; }

run_one() {
  local circuit="$1" tag="$2"
  [ -f "$circuit" ] || { echo "SKIP $tag: circuit $circuit not found"; return; }
  local ext=()
  [ -n "$SOFT_CMD" ] && ext=(--external-cmd "$SOFT_CMD" --external-shots "$EXTERNAL_SHOTS")
  python benchmarks/bench_sampler.py \
      --circuit "$circuit" --backend gpu \
      --batch-sizes $BATCHES --e2e-shots "$E2E_SHOTS" --stim-baseline \
      --out "$RESULTS/$tag.json" "${ext[@]}"
  echo
}

run_one "$CIRCUIT_D3" msc_d3
run_one "$CIRCUIT_D5" msc_d5

echo "results in $RESULTS/"
