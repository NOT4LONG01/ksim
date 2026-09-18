#include "bp_decoders.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cfloat>
#include <climits>

namespace asy {

// ─── EdgeTable ────────────────────────────────────────────────────────────────
EdgeTable EdgeTable::from_csc(
    int num_checks, int num_bits,
    const std::vector<int>& row_idx,
    const std::vector<int>& col_idx,
    const std::vector<float>& p_error)
{
    EdgeTable et;
    et.num_checks = num_checks;
    et.num_bits   = num_bits;
    et.num_edges  = (int)row_idx.size();

    Kokkos::View<int*, Kokkos::HostSpace> h_row("h_row", et.num_edges);
    Kokkos::View<int*, Kokkos::HostSpace> h_col("h_col", et.num_edges);
    for (int i = 0; i < et.num_edges; ++i) { h_row(i) = row_idx[i]; h_col(i) = col_idx[i]; }
    et.edge_row = Kokkos::create_mirror_view_and_copy(DeviceSpace{}, h_row);
    et.edge_col = Kokkos::create_mirror_view_and_copy(DeviceSpace{}, h_col);

    Kokkos::View<float*, Kokkos::HostSpace> h_llr("h_llr", num_bits);
    for (int j = 0; j < num_bits; ++j) {
        float p = std::max(BP_EPS, std::min(1.0f - BP_EPS, p_error[j]));
        h_llr(j) = std::log((1.0f - p) / p);
    }
    et.channel_llr = Kokkos::create_mirror_view_and_copy(DeviceSpace{}, h_llr);
    return et;
}

// ─── BpWorkspace ──────────────────────────────────────────────────────────────
BpWorkspace::BpWorkspace(int B_, int nc_, int nb_, int E_)
    : max_batch(B_), nc(nc_), nb(nb_), E(E_)
{
    syndrome_f = Kokkos::View<float**,   DeviceSpace>("syn_f",     B_, nc_);
    syndrome_u = Kokkos::View<uint8_t**, DeviceSpace>("syn_u",     B_, nc_);
    v2c        = Kokkos::View<float**,   DeviceSpace>("v2c",       B_, E_);
    c2v        = Kokkos::View<float**,   DeviceSpace>("c2v",       B_, E_);
    tlr_old    = Kokkos::View<float**,   DeviceSpace>("tlr_old",   B_, nb_);
    gamma      = Kokkos::View<float**,   DeviceSpace>("gamma",     B_, nb_);
    total_llr  = Kokkos::View<float**,   DeviceSpace>("total_llr", B_, nb_);
    row_sum    = Kokkos::View<float**,   DeviceSpace>("row_sum",   B_, nc_);
    sign_cnt   = Kokkos::View<float**,   DeviceSpace>("sign_cnt",  B_, nc_);
    pred       = Kokkos::View<uint8_t**, DeviceSpace>("pred",      B_, nb_);
    conv       = Kokkos::View<uint8_t*,  DeviceSpace>("conv",      B_);
    best_pred  = Kokkos::View<uint8_t**, DeviceSpace>("best_pred", B_, nb_);
    best_w     = Kokkos::View<float*,    DeviceSpace>("best_w",    B_);
    nconv      = Kokkos::View<int*,      DeviceSpace>("nconv",     B_);
    row_par    = Kokkos::View<unsigned int**, DeviceSpace>("row_par",   B_, nc_);
}

// ─── OsdWorkspace ─────────────────────────────────────────────────────────────
OsdWorkspace::OsdWorkspace(int osd_B_, int nc_, int nb_, int bp_max_batch_)
    : osd_max_batch(osd_B_), nc(nc_), nb(nb_)
{
    W  = (nb_ + 63) / 64;
    n2 = 1;
    while (n2 < nb_) n2 <<= 1;
    Hw     = Kokkos::View<uint64_t***, DeviceSpace>("osd_Hw",    osd_B_, nc_, W);
    sw     = Kokkos::View<uint8_t**,   DeviceSpace>("osd_sw",    osd_B_, nc_);
    perm   = Kokkos::View<int**,       DeviceSpace>("osd_perm",  osd_B_, n2);
    piv    = Kokkos::View<int**,       DeviceSpace>("osd_piv",   osd_B_, nc_);
    nc_idx = Kokkos::View<int*,        DeviceSpace>("osd_nc_idx",bp_max_batch_);
}

// ─── Dense H builder ──────────────────────────────────────────────────────────
Kokkos::View<uint8_t**, DeviceSpace> build_H_dense(const EdgeTable& et) {
    int nc = et.num_checks, nb = et.num_bits, E = et.num_edges;
    Kokkos::View<uint8_t**, DeviceSpace> H_d("H_d", nc, nb);
    Kokkos::deep_copy(H_d, (uint8_t)0);
    auto row = et.edge_row, col = et.edge_col;
    Kokkos::parallel_for("fill_H", E, KOKKOS_LAMBDA(int e) {
        H_d(row(e), col(e)) = 1;
    });
    Kokkos::fence();
    return H_d;
}

// ─── Upload syndromes ─────────────────────────────────────────────────────────
void upload_syndromes(const std::vector<uint8_t>& syn_host, int B, BpWorkspace& ws) {
    int nc = ws.nc;
    auto syn_h = Kokkos::create_mirror_view(ws.syndrome_u);
    for (int s = 0; s < B; ++s)
        for (int c = 0; c < nc; ++c)
            syn_h(s, c) = syn_host[(size_t)s * nc + c];
    Kokkos::deep_copy(ws.syndrome_u, syn_h);
}

// ─── Download results ─────────────────────────────────────────────────────────
BpResult download_result(int B, int nb, BpWorkspace& ws) {
    auto pred_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.pred);
    auto conv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.conv);
    BpResult res;
    res.predictions.resize((size_t)B * nb);
    res.converged.resize(B);
    for (int s = 0; s < B; ++s) {
        res.converged[s] = conv_h(s);
        for (int j = 0; j < nb; ++j)
            res.predictions[(size_t)s * nb + j] = pred_h(s, j);
    }
    return res;
}

// ─── OSD-0 on non-converged shots ─────────────────────────────────────────────
// One team (CUDA block) per shot.  Columns are sorted by signed posterior LLR
// ascending — most-likely-error bits first — so the GE pivot set is the
// most-likely-error independent set (OSD-0 à la Roffe et al.).  Rows are
// bit-packed into 64-bit words; elimination is team-parallel over rows with
// word-wide XOR.
//
// The order among tied posteriors decides the pivot set, and BP leaves many
// exact ties on a non-converged shot (symmetric bits far from the syndrome).
// Ties go to the likelier channel prior, then the lower column index; the
// quantisation keeps that rule stable under float32 atomic summation noise.
KOKKOS_INLINE_FUNCTION bool osd_col_before(float ta, float la, int ia,
                                           float tb, float lb, int ib) {
    float qa = Kokkos::floor(ta / OSD_TIE_TOL);
    float qb = Kokkos::floor(tb / OSD_TIE_TOL);
    if (qa != qb) return qa < qb;
    if (la != lb) return la < lb;
    return ia < ib;
}

void osd0_run_all(
    int B, int nc, int nb,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_d,
    const Kokkos::View<float*, DeviceSpace>& channel_llr,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws)
{
    auto conv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ws.conv);
    std::vector<int> nc_host;
    nc_host.reserve(B);
    for (int s = 0; s < B; ++s)
        if (!conv_h(s)) nc_host.push_back(s);

    if (nc_host.empty()) return;

    auto tlr    = ws.total_llr;
    auto ch_llr = channel_llr;
    auto syn_u  = ws.syndrome_u;
    auto pred   = ws.pred;
    auto Hw     = osd_ws.Hw;
    auto sw     = osd_ws.sw;
    auto perm   = osd_ws.perm;
    auto piv    = osd_ws.piv;
    auto nc_idx = osd_ws.nc_idx;
    const int osd_B = osd_ws.osd_max_batch;
    const int W     = osd_ws.W;
    const int n2    = osd_ws.n2;

    int total_nc = (int)nc_host.size();
    auto nc_idx_h = Kokkos::create_mirror_view(osd_ws.nc_idx);

    using policy_t = Kokkos::TeamPolicy<>;
    using member_t = policy_t::member_type;

    for (int start = 0; start < total_nc; start += osd_B) {
        int end  = std::min(start + osd_B, total_nc);
        int Bosd = end - start;

        for (int i = 0; i < Bosd; ++i)
            nc_idx_h(i) = nc_host[start + i];
        Kokkos::deep_copy(osd_ws.nc_idx, nc_idx_h);

        // 128 threads/team exceeds the host backends' team-size limit
#ifdef KOKKOS_ENABLE_CUDA
        policy_t osd_policy(Bosd, 128);
#else
        policy_t osd_policy(Bosd, Kokkos::AUTO);
#endif
        Kokkos::parallel_for("osd0", osd_policy,
                             KOKKOS_LAMBDA(const member_t& tm) {
            const int i = tm.league_rank();
            const int s = nc_idx(i);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n2), [&](int j) {
                perm(i, j) = (j < nb) ? j : -1;
            });
            tm.team_barrier();

            // Bitonic sort of perm by osd_col_before; -1 padding → tail.
            for (int k = 2; k <= n2; k <<= 1) {
                for (int half = k >> 1; half > 0; half >>= 1) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n2), [&](int idx) {
                        int l = idx ^ half;
                        if (l > idx) {
                            int pa = perm(i, idx), pb = perm(i, l);
                            bool a_after_b;
                            if (pa < 0)      a_after_b = (pb >= 0);
                            else if (pb < 0) a_after_b = false;
                            else a_after_b = osd_col_before(tlr(s, pb), ch_llr(pb), pb,
                                                            tlr(s, pa), ch_llr(pa), pa);
                            bool up = ((idx & k) == 0);
                            if (a_after_b == up) {
                                perm(i, idx) = pb;
                                perm(i, l)   = pa;
                            }
                        }
                    });
                    tm.team_barrier();
                }
            }

            // Pack permuted H rows into 64-bit words; copy syndrome.
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc * W), [&](int rw) {
                int r = rw / W, w = rw % W;
                int base = w * 64;
                int lim  = (nb - base < 64) ? (nb - base) : 64;
                uint64_t word = 0;
                for (int t = 0; t < lim; ++t) {
                    int pcol = perm(i, base + t);
                    if (pcol >= 0 && pcol < nb && H_d(r, pcol)) word |= (1ULL << t);
                }
                Hw(i, r, w) = word;
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc), [&](int r) {
                sw(i, r)  = syn_u(s, r);
                piv(i, r) = -1;
            });
            tm.team_barrier();

            // Gauss–Jordan to RREF in the sorted column order.
            int cur_row = 0;
            for (int j = 0; j < nb && cur_row < nc; ++j) {
                const int      w   = j >> 6;
                const uint64_t bit = 1ULL << (j & 63);

                int pivot = INT_MAX;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, cur_row, nc),
                    [&](int r, int& m) {
                        if ((Hw(i, r, w) & bit) && r < m) m = r;
                    }, Kokkos::Min<int>(pivot));

                if (pivot == INT_MAX) continue;
                if (pivot != cur_row) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, W), [&](int t) {
                        uint64_t tmp = Hw(i, cur_row, t);
                        Hw(i, cur_row, t) = Hw(i, pivot, t);
                        Hw(i, pivot, t)   = tmp;
                    });
                    Kokkos::single(Kokkos::PerTeam(tm), [&]() {
                        uint8_t t8 = sw(i, cur_row);
                        sw(i, cur_row) = sw(i, pivot);
                        sw(i, pivot)   = t8;
                    });
                    tm.team_barrier();
                }
                Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc), [&](int r) {
                    if (r != cur_row && (Hw(i, r, w) & bit)) {
                        for (int t = 0; t < W; ++t)
                            Hw(i, r, t) ^= Hw(i, cur_row, t);
                        sw(i, r) ^= sw(i, cur_row);
                    }
                });
                Kokkos::single(Kokkos::PerTeam(tm), [&]() {
                    piv(i, cur_row) = j;
                });
                tm.team_barrier();
                ++cur_row;
            }

            // OSD-0 solve: non-pivot bits 0, pivot bits read off the syndrome.
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nb), [&](int j) {
                pred(s, perm(i, j)) = 0;
            });
            tm.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc), [&](int r) {
                int j = piv(i, r);
                if (j >= 0) pred(s, perm(i, j)) = sw(i, r);
            });
        });
        Kokkos::fence();
    }
}

} // namespace asy
