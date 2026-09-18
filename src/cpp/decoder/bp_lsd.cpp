#include "bp_decoders.hpp"
#include <Kokkos_Core.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <queue>
#include <limits>

namespace asy {

// ─── CPU-side LSD for a single non-converged shot ────────────────────────────
// H_host: dense (nc × nb) row-major parity-check matrix (on CPU)
// syndrome: nc-length syndrome for this shot
// pred_in:  current BP prediction (nb-length)
// pred_out: corrected prediction (nb-length, modified in place)
// llr:      posterior LLRs (nb-length, |llr[j]| = reliability of bit j)
// Returns true if all checks satisfied after LSD.
static bool lsd_single(
    int nc, int nb,
    const std::vector<uint8_t>& H_host,  // (nc × nb) row-major
    const std::vector<uint8_t>& syndrome,
    const std::vector<float>&   llr,
    const std::vector<uint8_t>& pred_in,
    std::vector<uint8_t>&       pred_out,
    const LsdConfig&            cfg,
    // adjacency built once per call from H_host
    const std::vector<std::vector<int>>& check_to_bits,
    const std::vector<std::vector<int>>& bit_to_checks,
    std::vector<int32_t>*       cluster_ids_out = nullptr)  // nb-length if non-null
{
    pred_out = pred_in;

    // Compute residual syndrome: which checks are still unsatisfied?
    std::vector<uint8_t> res_syn(nc);
    for (int c = 0; c < nc; ++c) {
        uint8_t par = 0;
        for (int b : check_to_bits[c]) par ^= pred_out[b];
        res_syn[c] = par ^ syndrome[c];  // 1 = unsatisfied
    }

    // BFS: expand each unsatisfied check into a cluster of (checks, bits)
    // visited_check[c] = cluster_id it belongs to (-1 = unvisited)
    // visited_bit[b]   = cluster_id it belongs to (-1 = unvisited)
    std::vector<int> visited_check(nc, -1);
    std::vector<int> visited_bit(nb, -1);

    // One pass: cluster = connected component of unsatisfied check + neighbors
    int num_clusters = 0;
    struct Cluster { std::vector<int> bits; std::vector<int> checks; };
    std::vector<Cluster> clusters;

    for (int c0 = 0; c0 < nc; ++c0) {
        if (!res_syn[c0] || visited_check[c0] >= 0) continue;

        // BFS from unsatisfied check c0
        int cid = num_clusters++;
        clusters.push_back({});
        Cluster& cl = clusters.back();

        std::queue<int> check_q;
        check_q.push(c0);
        visited_check[c0] = cid;
        cl.checks.push_back(c0);

        for (int _iter = 0; _iter < cfg.max_cluster_iters && !check_q.empty(); ++_iter) {
            // Expand checks → bits
            std::queue<int> bit_q;
            while (!check_q.empty()) {
                int c = check_q.front(); check_q.pop();
                for (int b : check_to_bits[c]) {
                    if (visited_bit[b] < 0) {
                        visited_bit[b] = cid;
                        cl.bits.push_back(b);
                        bit_q.push(b);
                    }
                }
            }
            // Expand bits → new unsatisfied checks
            while (!bit_q.empty()) {
                int b = bit_q.front(); bit_q.pop();
                for (int c : bit_to_checks[b]) {
                    if (visited_check[c] < 0) {
                        visited_check[c] = cid;
                        cl.checks.push_back(c);
                        if (res_syn[c]) check_q.push(c);
                    }
                }
            }
        }
    }

    if (cluster_ids_out) {
        std::fill(cluster_ids_out->begin(), cluster_ids_out->end(), 0);
        for (size_t cid = 0; cid < clusters.size(); ++cid)
            for (int b : clusters[cid].bits)
                (*cluster_ids_out)[b] = (int32_t)(cid + 1);  // 1-based, 0 = no cluster
    }

    // Process each cluster
    bool all_ok = true;
    for (auto& cl : clusters) {
        int kb = (int)cl.bits.size();
        int kc = (int)cl.checks.size();

        // Extract local syndrome (restricted to this cluster's checks)
        std::vector<uint8_t> local_syn(kc);
        for (int i = 0; i < kc; ++i) local_syn[i] = res_syn[cl.checks[i]];

        if (kb <= cfg.max_cluster_bits) {
            // Brute force: try all 2^kb flip patterns on top of current pred
            // Pick minimum-weight pattern that satisfies all local checks.
            // Note: "flip" means XOR with the cluster's local error pattern e.
            // A check c is satisfied by flip pattern e iff
            //   H_local_row(c) · e == local_syn(c)
            // where H_local is the nc×nb submatrix restricted to cluster checks/bits.

            // Build local H (kc × kb) in row-major
            std::vector<uint8_t> Hloc(kc * kb, 0);
            for (int i = 0; i < kc; ++i) {
                int c = cl.checks[i];
                for (int j = 0; j < kb; ++j) {
                    int b = cl.bits[j];
                    Hloc[i * kb + j] = H_host[c * nb + b];
                }
            }

            int best_weight = std::numeric_limits<int>::max();
            uint32_t best_pattern = 0;

            uint32_t total = 1u << kb;
            for (uint32_t pat = 0; pat < total; ++pat) {
                // Check if this pattern satisfies all local checks
                bool ok = true;
                for (int i = 0; i < kc && ok; ++i) {
                    uint8_t par = 0;
                    for (int j = 0; j < kb; ++j)
                        if ((pat >> j) & 1) par ^= Hloc[i * kb + j];
                    if (par != local_syn[i]) ok = false;
                }
                if (ok) {
                    int w = __builtin_popcount(pat);
                    if (w < best_weight) { best_weight = w; best_pattern = pat; }
                }
            }

            // Apply best pattern as XOR flip on pred_out
            if (best_weight < std::numeric_limits<int>::max()) {
                for (int j = 0; j < kb; ++j)
                    if ((best_pattern >> j) & 1) pred_out[cl.bits[j]] ^= 1;
            } else {
                all_ok = false;  // cluster couldn't be solved
            }
        } else {
            // Large cluster: OSD-0 on the local submatrix (CPU)
            // Sort bits by signed LLR ascending — most-likely-error first —
            // so GE pivots land on the bits BP considers most likely flipped.
            std::vector<int> ord(kb);
            for (int j = 0; j < kb; ++j) ord[j] = j;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                return llr[cl.bits[a]] < llr[cl.bits[b]];
            });

            // Build permuted local H (kc × kb)
            std::vector<uint8_t> Hloc(kc * kb, 0);
            for (int i = 0; i < kc; ++i) {
                int c = cl.checks[i];
                for (int j = 0; j < kb; ++j)
                    Hloc[i * kb + j] = H_host[c * nb + cl.bits[ord[j]]];
            }
            std::vector<uint8_t> s_local = local_syn;

            // Gaussian elimination (GF2) on (Hloc | s_local)
            std::vector<int> piv(kc, -1);
            int cur_row = 0;
            for (int j = 0; j < kb && cur_row < kc; ++j) {
                int pivot = -1;
                for (int r = cur_row; r < kc; ++r)
                    if (Hloc[r * kb + j]) { pivot = r; break; }
                if (pivot < 0) continue;
                if (pivot != cur_row) {
                    for (int k = 0; k < kb; ++k) std::swap(Hloc[cur_row*kb+k], Hloc[pivot*kb+k]);
                    std::swap(s_local[cur_row], s_local[pivot]);
                }
                for (int r = 0; r < kc; ++r) {
                    if (r != cur_row && Hloc[r*kb+j]) {
                        for (int k = 0; k < kb; ++k)
                            Hloc[r*kb+k] ^= Hloc[cur_row*kb+k];
                        s_local[r] ^= s_local[cur_row];
                    }
                }
                piv[cur_row] = j;
                ++cur_row;
            }
            int rank = cur_row;

            // OSD-0: non-pivot bits = 0, pivot bits = syndrome readout
            std::vector<uint8_t> x(kb, 0);
            for (int r = 0; r < rank; ++r)
                if (piv[r] >= 0) x[piv[r]] = s_local[r];

            // Apply to pred_out (XOR flip in original bit ordering)
            for (int j = 0; j < kb; ++j)
                if (x[j]) pred_out[cl.bits[ord[j]]] ^= 1;
        }
    }
    (void)all_ok;

    // Verify full syndrome
    for (int c = 0; c < nc; ++c) {
        uint8_t par = 0;
        for (int b : check_to_bits[c]) par ^= pred_out[b];
        if (par != syndrome[c]) return false;
    }
    return true;
}

// ─── Public: BP + LSD ─────────────────────────────────────────────────────────
BpResult bp_lsd_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B, int max_bp_iter,
    const LsdConfig& lsd_cfg,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws,
    std::vector<int32_t>* clusters_out)
{
    upload_syndromes(syndromes, B, ws);
    bp_run(et, B, max_bp_iter, ws);

    if (clusters_out) {
        clusters_out->assign((size_t)B * et.num_bits, 0);
    }

    // Check convergence
    int num_nc = 0;
    Kokkos::parallel_reduce("lsd_check_nc", B,
        KOKKOS_LAMBDA(int s, int& acc) { if (!ws.conv(s)) ++acc; },
        num_nc);

    if (num_nc == 0)
        return download_result(B, et.num_bits, ws);

    // Download workspace for non-converged shots
    auto pred_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.pred);
    auto conv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.conv);
    auto tlr_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.total_llr);
    auto syn_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.syndrome_u);

    int nc = et.num_checks, nb = et.num_bits;

    // Build adjacency lists on CPU from edge table
    std::vector<std::vector<int>> check_to_bits(nc), bit_to_checks(nb);
    {
        auto er_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, et.edge_row);
        auto ec_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, et.edge_col);
        for (int e = 0; e < et.num_edges; ++e) {
            check_to_bits[er_h(e)].push_back(ec_h(e));
            bit_to_checks[ec_h(e)].push_back(er_h(e));
        }
    }

    // Build dense H on CPU (downloaded from GPU once)
    auto H_host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, H_dev);
    std::vector<uint8_t> H_host(nc * nb);
    for (int r = 0; r < nc; ++r)
        for (int j = 0; j < nb; ++j)
            H_host[r * nb + j] = H_host_view(r, j);

    // LSD on each non-converged shot
    std::vector<uint8_t> lsd_corrections(B * nb);
    std::vector<uint8_t> pred_in(nb), pred_out(nb);
    std::vector<uint8_t> syndrome_shot(nc);
    std::vector<float>   llr_shot(nb);
    std::vector<int32_t> cluster_ids_shot(clusters_out ? nb : 0);

    for (int s = 0; s < B; ++s) {
        if (conv_h(s)) continue;

        for (int j = 0; j < nb; ++j) {
            pred_in[j] = pred_h(s, j);
            llr_shot[j] = tlr_h(s, j);
        }
        for (int c = 0; c < nc; ++c)
            syndrome_shot[c] = syn_h(s, c);

        lsd_single(nc, nb, H_host, syndrome_shot, llr_shot, pred_in, pred_out,
                   lsd_cfg, check_to_bits, bit_to_checks,
                   clusters_out ? &cluster_ids_shot : nullptr);

        for (int j = 0; j < nb; ++j)
            lsd_corrections[s * nb + j] = pred_out[j];
        if (clusters_out) {
            std::copy(cluster_ids_shot.begin(), cluster_ids_shot.end(),
                       clusters_out->begin() + (size_t)s * nb);
        }
    }

    // Upload LSD-corrected predictions for non-converged shots
    auto pred_upload = Kokkos::create_mirror_view(ws.pred);
    Kokkos::deep_copy(pred_upload, ws.pred);
    for (int s = 0; s < B; ++s) {
        if (conv_h(s)) continue;
        for (int j = 0; j < nb; ++j)
            pred_upload(s, j) = lsd_corrections[s * nb + j];
    }
    Kokkos::deep_copy(ws.pred, pred_upload);
    Kokkos::fence();

    return download_result(B, et.num_bits, ws);
}

} // namespace asy
