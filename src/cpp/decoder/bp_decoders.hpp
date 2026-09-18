// bp_decoders — the kokkos_decoder module API: shared BP/OSD infrastructure +
// the BP-variant decoder entry points.
//
// One header for the module: the common GPU data structures (EdgeTable,
// BpWorkspace, OsdWorkspace) and helpers are implemented in bp_infra.cpp; each
// BP-variant decoder's entry points are implemented in its own .cpp:
//   bp_osd.cpp    — bp_run / bp_decode_batch / bp_osd0_decode_batch
//   relay_bp.cpp  — bp_run_memory / relay_bp_decode_batch
//   bp_lsd.cpp    — bp_lsd_decode_batch
#pragma once
#include <Kokkos_Core.hpp>
#include <vector>
#include <cstdint>

// CudaSpace on CUDA builds, HostSpace on CPU-only Kokkos builds
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

namespace asy {

// ─── Shared BP/OSD infrastructure ────────────────────────────────────────────
static constexpr float BP_EPS = 1e-6f;

struct EdgeTable {
    int num_checks;
    int num_bits;
    int num_edges;
    Kokkos::View<int*,   DeviceSpace> edge_row;
    Kokkos::View<int*,   DeviceSpace> edge_col;
    Kokkos::View<float*, DeviceSpace> channel_llr;

    static EdgeTable from_csc(
        int num_checks, int num_bits,
        const std::vector<int>& row_indices,
        const std::vector<int>& col_indices,
        const std::vector<float>& p_error);
};

struct BpResult {
    std::vector<uint8_t> predictions;  // flat row-major (B × num_bits)
    std::vector<uint8_t> converged;    // (B,)
};

struct BpWorkspace {
    int max_batch, nc, nb, E;

    Kokkos::View<float**,   DeviceSpace> syndrome_f;
    Kokkos::View<uint8_t**, DeviceSpace> syndrome_u;
    Kokkos::View<float**,   DeviceSpace> v2c;
    Kokkos::View<float**,   DeviceSpace> c2v;
    Kokkos::View<float**,   DeviceSpace> tlr_old;   // (B,nb) prev posterior (memory BP)
    Kokkos::View<float**,   DeviceSpace> gamma;     // (B,nb) per-variable memory strength
    Kokkos::View<float**,   DeviceSpace> total_llr;
    Kokkos::View<float**,   DeviceSpace> row_sum;
    Kokkos::View<float**,   DeviceSpace> sign_cnt;
    Kokkos::View<uint8_t**, DeviceSpace> pred;
    Kokkos::View<uint8_t*,  DeviceSpace> conv;
    // Relay-BP ensemble state: the lowest-prior-weight converged solution seen so
    // far, its weight, and how many converged solutions the shot has produced.
    Kokkos::View<uint8_t**, DeviceSpace> best_pred;
    Kokkos::View<float*,    DeviceSpace> best_w;
    Kokkos::View<int*,      DeviceSpace> nconv;
    // 32-bit (not uint8) so check_parity's atomic XOR uses native hardware
    // atomics; sub-word atomics fall back to CAS/lock emulation and serialize
    // badly on high-weight rows.
    Kokkos::View<unsigned int**, DeviceSpace> row_par;

    BpWorkspace() : max_batch(0), nc(0), nb(0), E(0) {}
    BpWorkspace(int B_, int nc_, int nb_, int E_);
};

struct OsdWorkspace {
    int osd_max_batch, nc, nb;
    int W;    // 64-bit words per bit-packed row: ceil(nb / 64)
    int n2;   // bitonic sort length: next power of two ≥ nb
    Kokkos::View<uint64_t***, DeviceSpace> Hw;    // (osd_B, nc, W) packed permuted H
    Kokkos::View<uint8_t**,   DeviceSpace> sw;    // (osd_B, nc)   permuted syndrome
    Kokkos::View<int**,       DeviceSpace> perm;  // (osd_B, n2)   column order
    Kokkos::View<int**,       DeviceSpace> piv;   // (osd_B, nc)   pivot column per row
    Kokkos::View<int*,        DeviceSpace> nc_idx;

    OsdWorkspace() : osd_max_batch(0), nc(0), nb(0), W(0), n2(0) {}
    OsdWorkspace(int osd_max_batch_, int nc_, int nb_, int bp_max_batch_);
};

Kokkos::View<uint8_t**, DeviceSpace> build_H_dense(const EdgeTable& et);

// Upload row-major uint8 syndrome array (B×nc) from host to workspace.
void upload_syndromes(const std::vector<uint8_t>& syn_host, int B, BpWorkspace& ws);

// Download workspace predictions and convergence flags to host BpResult.
BpResult download_result(int B, int nb, BpWorkspace& ws);

// OSD-0 on non-converged shots, in sub-batches of osd_ws.osd_max_batch.
// Columns are ordered by posterior LLR ascending; posteriors within OSD_TIE_TOL
// of each other are ordered by channel LLR ascending (likelier error first),
// then by column index, so the pivot set does not depend on float summation
// order.  Leaves ws.conv untouched: it still reports whether BP converged.
static constexpr float OSD_TIE_TOL = 1e-3f;
void osd0_run_all(
    int B, int nc, int nb,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_d,
    const Kokkos::View<float*, DeviceSpace>& channel_llr,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws);

// ─── BP + OSD-0  (bp_osd.cpp) ────────────────────────────────────────────────
// Standard product-sum BP.
void bp_run(const EdgeTable& et, int B, int max_iter, BpWorkspace& ws);

// BP only — uses pre-allocated workspace.
BpResult bp_decode_batch(
    const EdgeTable& et,
    const std::vector<uint8_t>& syndromes,
    int B,
    int max_iter,
    BpWorkspace& ws);

// BP + OSD-0.  OSD-0 runs only on non-converged shots via index-gather.
BpResult bp_osd0_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B,
    int max_bp_iter,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws);

// ─── Relay BP / disordered-memory BP  (relay_bp.cpp) ─────────────────────────
// Disordered-memory BP (Relay-BP style): each variable j blends its channel
// prior with its previous-iteration posterior using a per-(shot,variable)
// memory strength, ws.gamma(s,j):
//   prior_eff = (1-gamma)*lambda_j + gamma*M_j^(t-1)
//   M_j^(t)   = prior_eff + sum_c nu_{c->j}
// gamma=0 everywhere degenerates to standard product-sum BP.
// ws.gamma and ws.tlr_old must be populated before calling; tlr_old carries
// the posterior across legs (the "relay").
void bp_run_memory(
    const EdgeTable& et,
    int B,
    int max_iter,
    BpWorkspace& ws);

// Named as in arXiv:2506.01779: legs r = 0..R-1, leg 0 runs T0 iterations of
// BP with the uniform memory gamma0 (Mem-BP), every later leg Tr iterations
// with gamma_j ~ U[gamma_center - gamma_width/2, gamma_center + gamma_width/2]
// (DMem-BP); S is the number of solutions sought.  Defaults are the paper's
// gross-code values.
struct RelayConfig {
    int   R            = 301;     // maximum number of legs, the first included
    int   T0           = 80;      // iterations of leg 0
    int   Tr           = 60;      // iterations of every later leg
    float gamma0       = 0.125f;  // uniform memory strength of leg 0 (0 = plain BP)
    float gamma_center = 0.21f;   // the interval [-0.24, 0.66]
    float gamma_width  = 0.90f;
    int   S            = 1;       // solutions sought per shot; <= 0 runs every leg
    uint64_t seed      = 42;
};

// Relay BP (arXiv:2506.01779): leg 0 is T0 iterations of BP with memory gamma0,
// legs 1..R-1 are Tr iterations of disordered-memory BP.  Every leg restarts the
// messages from the channel priors and keeps only the previous posterior as its
// memory (the relay).  A shot keeps decoding until it has produced S converged
// solutions and returns the one of lowest prior weight (Relay-BP-S); a shot
// that never converges falls back to OSD-0.  converged reports whether any leg
// converged.
BpResult relay_bp_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B,
    const RelayConfig& cfg,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws);

// ─── BP + LSD  (bp_lsd.cpp) ──────────────────────────────────────────────────
struct LsdConfig {
    int max_cluster_bits = 20;   // clusters with > this many bits fall back to OSD-0
    int max_cluster_iters = 10;  // BFS expansion rounds (diameter bound)
};

// BP + LSD.
//   - BP step runs on GPU (Kokkos).
//   - LSD cluster finding and brute-force runs on CPU (after downloading non-converged shots).
//   - Clusters larger than max_cluster_bits fall back to OSD-0 (CPU Gaussian elimination).
//
// clusters_out (optional): if non-null, resized to B*nb and filled with each
// bit's local (per-shot) 1-based cluster id, 0 = not in any cluster (includes
// every bit on shots where BP converged -- LSD never runs there, so there are
// no clusters to report). Mirrors ldpc.BpLsdDecoder's individual_cluster_stats
// in a flat, Python-side-reducible form: pair with priors-derived bit_llrs and
// analysis.clusters.compute_cluster_stats(clusters[s], bit_llrs) per shot, the
// same reduction decode.sinter.CpuBpLsdDecoder.decode_one uses.
BpResult bp_lsd_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B,
    int max_bp_iter,
    const LsdConfig& lsd_cfg,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws,
    std::vector<int32_t>* clusters_out = nullptr);

} // namespace asy
