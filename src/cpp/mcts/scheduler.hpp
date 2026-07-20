// scheduler — MCTS syndrome-extraction schedule optimizer (kokkos_mcts module API).
//
// One header for the module: GPU schedule evaluation (schedule_eval.cpp), the
// UCT-MCTS tree (mcts.cpp), and the MPI-distributed search (mpi_search.cpp). The
// GPU BP decoders (EdgeTable etc.) come from ../decoder/bp_decoders.hpp.
#pragma once
#include "../decoder/bp_decoders.hpp"

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <limits>
#include <cstdint>
#include <mpi.h>

namespace asy {

// ─── GPU-parallel schedule evaluation (schedule_eval.cpp) ────────────────────
// Joint circuit-level evaluation of a CSS syndrome-extraction schedule:
//
// 1. Sample nshots Pauli-frame simulations of one extraction round natively
//    (no external simulator).  The round follows circuit.py's structure:
//    X-check phase (ancilla-controlled CNOTs) then Z-check phase
//    (data-controlled CNOTs), with DEPOLARIZE2 after every CNOT,
//    DEPOLARIZE1 on data before the round and X-flips on measurements.
//    Hook errors — ancilla errors propagating onto data mid-extraction,
//    whose damage depends on the CNOT order — emerge naturally from the
//    frame propagation, and they couple the two blocks: the X-block
//    schedule shapes the error patterns the Z block must decode.
// 2. Decode each block's syndrome with Kokkos GPU BP (Hx for Z errors,
//    Hz for X errors).
// 3. Return P(either logical observable wrong).
//
// Same `seed` ⇒ same noise realisation, so candidate schedules are compared
// with common random numbers (variance reduction for the search).

// Joint scheduling problem over both check blocks.  Schedule rows
// [0, ncx) are X checks, [ncx, ncx+ncz) are Z checks; the flat schedule
// index is row * num_ticks + tick → data qubit (-1 = idle).
struct CssProblem {
    int nbits  = 0;
    int ncx    = 0;
    int ncz    = 0;
    int num_ticks = 0;
    std::vector<std::vector<int>> supp;   // per-row data-qubit support
    EdgeTable et_x;                       // Hx edges (decodes Z errors)
    EdgeTable et_z;                       // Hz edges (decodes X errors)
    std::vector<uint8_t> obs_x;           // logical-X support mask (nbits)
    std::vector<uint8_t> obs_z;           // logical-Z support mask (nbits)
    float p = 0.0f;                       // physical error rate
};

// Build the problem from dense H blocks.  BP priors are derived from p and
// each qubit's gate count (marginal flip probability per DEPOLARIZE2 plus
// the before-round DEPOLARIZE1).  num_ticks = max check weight + slack.
CssProblem make_css_problem(
    int nbits, int ncx, int ncz,
    const std::vector<uint8_t>& Hx_dense,   // ncx × nbits, row-major
    const std::vector<uint8_t>& Hz_dense,   // ncz × nbits, row-major
    const std::vector<uint8_t>& obs_x,
    const std::vector<uint8_t>& obs_z,
    float p,
    int tick_slack = 2);

// Full evaluate: sample, decode both blocks, return LER.
float evaluate_css_schedule(
    const CssProblem& prob,
    const std::vector<int>& schedule,
    int nshots,
    int max_bp_iter,
    int batch_size = 4096,
    uint64_t seed = 42);

// ─── UCT-MCTS schedule search (mcts.cpp) ─────────────────────────────────────
// Runs on CPU (one tree per MPI rank).  Leaf evaluation (Pauli-frame sample +
// GPU BP decode) is dispatched via evaluate_css_schedule above. The schedule
// jointly covers both check blocks because hook errors couple them; every
// H-edge (check, data qubit) is assigned exactly one tick and the search
// explores tick permutations, rooted at the natural-order baseline, with
// rollouts sharing one RNG seed (common random numbers).

// A schedule is a mapping from (row, tick) → data qubit (-1 = idle).
// Represented as a flat vector; index = row * num_ticks + tick.
using Schedule = std::vector<int>;

struct MctsNode {
    Schedule schedule;
    int visits   = 0;
    double value = 0.0;   // accumulated (1 - LER); higher = better
    std::vector<std::unique_ptr<MctsNode>> children;
    MctsNode* parent = nullptr;

    double uct(double C = 1.41421356) const {
        if (visits == 0) return std::numeric_limits<double>::infinity();
        double parent_v = parent ? (double)parent->visits : 1.0;
        return value / visits + C * std::sqrt(std::log(parent_v) / visits);
    }
};

class MctsTree {
public:
    MctsTree(const CssProblem& prob, int nshots, int max_bp_iter, uint64_t seed);

    // Run `iters` UCT iterations, return best schedule found so far.
    Schedule run(int iters);

    float best_score() const { return _best_score; }

    // Serialise best schedule as JSON array-of-arrays
    std::string best_schedule_json() const;

private:
    CssProblem _prob;
    int _nrows, _nt;
    int _nshots, _max_bp_iter;
    uint64_t _seed;
    std::mt19937_64 _rng;
    std::unique_ptr<MctsNode> _root;
    Schedule _best_sched;
    float _best_score = std::numeric_limits<float>::infinity();

    MctsNode* select(MctsNode* node);
    MctsNode* expand(MctsNode* node);
    float rollout(MctsNode* node);
    void backprop(MctsNode* node, float score);
    Schedule natural_schedule() const;
    std::vector<Schedule> neighbour_schedules(const Schedule& s);
};

// ─── MPI-distributed MCTS (mpi_search.cpp) ───────────────────────────────────
// Each rank runs an independent MCTS tree.  After each step, ranks share their
// best schedule via MPI_Allreduce and update their local best.

struct MpiSearchResult {
    Schedule best_schedule;
    float best_ler;
    int rank;
};

// Distributed search: run `steps` rounds of `iters_per_step` MCTS iterations,
// sharing the global best after each round. Returns the globally best schedule
// on all ranks.
MpiSearchResult mpi_mcts_search(
    MctsTree& local_tree,
    int steps,
    int iters_per_step,
    MPI_Comm comm = MPI_COMM_WORLD);

} // namespace asy
