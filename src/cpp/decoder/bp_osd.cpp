#include "bp_decoders.hpp"
#include <Kokkos_Core.hpp>

namespace asy {

// ─── Standard product-sum BP ──────────────────────────────────────────────────
void bp_run(const EdgeTable& et, int B, int max_iter, BpWorkspace& ws) {
    int E  = et.num_edges;
    int nc = et.num_checks;
    int nb = et.num_bits;

    auto row    = et.edge_row;
    auto col    = et.edge_col;
    auto ch_llr = et.channel_llr;

    auto pair_B = Kokkos::make_pair(0, B);
    auto syn_u   = Kokkos::subview(ws.syndrome_u, pair_B, Kokkos::ALL());
    auto syn_f   = Kokkos::subview(ws.syndrome_f, pair_B, Kokkos::ALL());
    auto v2c     = Kokkos::subview(ws.v2c,        pair_B, Kokkos::ALL());
    auto c2v     = Kokkos::subview(ws.c2v,        pair_B, Kokkos::ALL());
    auto tlr     = Kokkos::subview(ws.total_llr,  pair_B, Kokkos::ALL());
    auto rs      = Kokkos::subview(ws.row_sum,    pair_B, Kokkos::ALL());
    auto sc      = Kokkos::subview(ws.sign_cnt,   pair_B, Kokkos::ALL());
    auto pred    = Kokkos::subview(ws.pred,       pair_B, Kokkos::ALL());
    auto conv    = Kokkos::subview(ws.conv,       pair_B);
    auto row_par = Kokkos::subview(ws.row_par,    pair_B, Kokkos::ALL());

    Kokkos::parallel_for("syn2f",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nc}),
        KOKKOS_LAMBDA(int s, int c) { syn_f(s,c) = (float)syn_u(s,c); });

    Kokkos::parallel_for("init_v2c",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
        KOKKOS_LAMBDA(int s, int e) {
            v2c(s,e) = ch_llr(col(e));
            c2v(s,e) = 0.0f;
        });

    Kokkos::deep_copy(ws.conv, (uint8_t)0);
    Kokkos::fence();

    Kokkos::DefaultExecutionSpace exec;
    for (int iter = 0; iter < max_iter; ++iter) {
        Kokkos::deep_copy(exec, rs, 0.0f);
        Kokkos::deep_copy(exec, sc, 0.0f);
        Kokkos::parallel_for("row_scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                float tv = Kokkos::tanh(v2c(s,e) * 0.5f);
                // Clamp magnitude away from both 0 and 1: away from 1 keeps atanh()
                // finite downstream, away from 0 keeps log(|tv|) from hitting -inf
                // (which otherwise cancels against itself in c2v_update's
                // rs(row) - lt and produces NaN that poisons total_llr).
                float mag = Kokkos::min(Kokkos::max(Kokkos::abs(tv), 1e-6f), 1.0f - BP_EPS);
                tv = (tv < 0.0f) ? -mag : mag;
                Kokkos::atomic_add(&rs(s, row(e)), Kokkos::log(mag));
                if (tv < 0.0f) Kokkos::atomic_add(&sc(s, row(e)), 1.0f);
            });

        Kokkos::parallel_for("c2v_update",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                float tv = Kokkos::tanh(v2c(s,e) * 0.5f);
                float mag = Kokkos::min(Kokkos::max(Kokkos::abs(tv), 1e-6f), 1.0f - BP_EPS);
                tv = (tv < 0.0f) ? -mag : mag;
                float lt       = Kokkos::log(mag);
                float extr_lt  = rs(s, row(e)) - lt;
                float extr_mag = Kokkos::min(Kokkos::exp(Kokkos::min(extr_lt, 10.0f)), 1.0f - BP_EPS);
                float own_neg  = (tv < 0.0f) ? 1.0f : 0.0f;
                float row_neg  = sc(s, row(e)) - own_neg;
                float esign    = (Kokkos::fmod(row_neg, 2.0f) != 0.0f) ? -1.0f : 1.0f;
                float ssign    = (syn_f(s, row(e)) > 0.5f) ? -1.0f : 1.0f;
                c2v(s,e) = 2.0f * Kokkos::atanh(esign * ssign * extr_mag);
            });

        Kokkos::deep_copy(exec, tlr, 0.0f);
        Kokkos::parallel_for("col_scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                Kokkos::atomic_add(&tlr(s, col(e)), c2v(s,e));
            });
        Kokkos::parallel_for("add_ch",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
            KOKKOS_LAMBDA(int s, int j) {
                if (conv(s)) return;
                tlr(s,j) += ch_llr(j);
                pred(s,j) = (tlr(s,j) < 0.0f) ? 1 : 0;
            });
        Kokkos::parallel_for("v2c_update",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                v2c(s,e) = tlr(s, col(e)) - c2v(s,e);
            });

        Kokkos::deep_copy(exec, conv, (uint8_t)1);
        Kokkos::deep_copy(exec, row_par, 0u);
        Kokkos::parallel_for("check_parity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                Kokkos::atomic_fetch_xor(&row_par(s, row(e)),
                                         (unsigned int)pred(s, col(e)));
            });
        Kokkos::parallel_for("check_conv",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nc}),
            KOKKOS_LAMBDA(int s, int r) {
                unsigned int want = (syn_f(s,r) > 0.5f) ? 1u : 0u;
                if (row_par(s,r) != want) conv(s) = 0;
            });
        // conv flags update on-device every iteration; the host-blocking
        // early-exit count only runs periodically to avoid a sync per iteration.
        if (iter % 8 == 7 || iter == max_iter - 1) {
            int num_nc = 0;
            Kokkos::parallel_reduce("count_nc", B,
                KOKKOS_LAMBDA(int s, int& acc) { if (!conv(s)) ++acc; },
                num_nc);
            if (num_nc == 0) break;
        }
    }
    Kokkos::fence();
}

// ─── Public decode functions ───────────────────────────────────────────────────
BpResult bp_decode_batch(
    const EdgeTable& et,
    const std::vector<uint8_t>& syndromes,
    int B, int max_iter, BpWorkspace& ws)
{
    upload_syndromes(syndromes, B, ws);
    bp_run(et, B, max_iter, ws);
    return download_result(B, et.num_bits, ws);
}

BpResult bp_osd0_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B, int max_bp_iter,
    BpWorkspace& ws, OsdWorkspace& osd_ws)
{
    upload_syndromes(syndromes, B, ws);
    bp_run(et, B, max_bp_iter, ws);
    osd0_run_all(B, et.num_checks, et.num_bits, H_dev, et.channel_llr, ws, osd_ws);
    return download_result(B, et.num_bits, ws);
}

} // namespace asy
