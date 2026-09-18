#include "bp_decoders.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cmath>

namespace asy {

// ─── Disordered-memory BP (Relay-BP) ─────────────────────────────────────────
// Each variable blends its channel prior with its previous-iteration posterior
// using per-(shot,variable) memory strength ws.gamma(s,j); ws.tlr_old carries
// posteriors across iterations and legs.
void bp_run_memory(
    const EdgeTable& et,
    int B, int max_iter,
    BpWorkspace& ws)
{
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
    auto tlr_old = Kokkos::subview(ws.tlr_old,    pair_B, Kokkos::ALL());
    auto gam     = Kokkos::subview(ws.gamma,      pair_B, Kokkos::ALL());
    auto c2v     = Kokkos::subview(ws.c2v,        pair_B, Kokkos::ALL());
    auto tlr     = Kokkos::subview(ws.total_llr,  pair_B, Kokkos::ALL());
    auto rs      = Kokkos::subview(ws.row_sum,     pair_B, Kokkos::ALL());
    auto sc      = Kokkos::subview(ws.sign_cnt,    pair_B, Kokkos::ALL());
    auto pred    = Kokkos::subview(ws.pred,        pair_B, Kokkos::ALL());
    auto conv    = Kokkos::subview(ws.conv,        pair_B);
    auto row_par = Kokkos::subview(ws.row_par,     pair_B, Kokkos::ALL());

    Kokkos::DefaultExecutionSpace exec;
    for (int iter = 0; iter < max_iter; ++iter) {
        // c2v update — same as standard BP
        Kokkos::deep_copy(exec, rs, 0.0f);
        Kokkos::deep_copy(exec, sc, 0.0f);
        Kokkos::parallel_for("mr_scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                float tv = Kokkos::tanh(v2c(s,e) * 0.5f);
                // Clamp magnitude away from both 0 and 1: away from 1 keeps atanh()
                // finite downstream, away from 0 keeps log(|tv|) from hitting -inf
                // (which otherwise cancels against itself in mc2v's rs(row) - lt
                // and produces NaN that poisons total_llr).
                float mag = Kokkos::min(Kokkos::max(Kokkos::abs(tv), 1e-6f), 1.0f - BP_EPS);
                tv = (tv < 0.0f) ? -mag : mag;
                Kokkos::atomic_add(&rs(s, row(e)), Kokkos::log(mag));
                if (tv < 0.0f) Kokkos::atomic_add(&sc(s, row(e)), 1.0f);
            });

        Kokkos::parallel_for("mc2v",
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

        // Variable update + memory v2c
        Kokkos::deep_copy(exec, tlr, 0.0f);
        Kokkos::parallel_for("mcol_scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                Kokkos::atomic_add(&tlr(s, col(e)), c2v(s,e));
            });
        // Posterior with memory-blended prior:
        //   tlr = (1-gamma)*lambda + gamma*tlr_old + sum_c c2v
        Kokkos::parallel_for("madd_ch",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
            KOKKOS_LAMBDA(int s, int j) {
                if (conv(s)) return;
                float g = gam(s,j);
                tlr(s,j) += (1.0f - g) * ch_llr(j) + g * tlr_old(s,j);
                pred(s,j) = (tlr(s,j) < 0.0f) ? 1 : 0;
            });
        Kokkos::parallel_for("mv2c",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (conv(s)) return;
                v2c(s,e) = tlr(s, col(e)) - c2v(s,e);
            });
        Kokkos::parallel_for("msave_tlr",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
            KOKKOS_LAMBDA(int s, int j) {
                if (conv(s)) return;
                tlr_old(s,j) = tlr(s,j);
            });

        // Convergence check
        Kokkos::deep_copy(exec, conv, (uint8_t)1);
        Kokkos::deep_copy(exec, row_par, 0u);
        Kokkos::parallel_for("mparity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                Kokkos::atomic_fetch_xor(&row_par(s, row(e)),
                                         (unsigned int)pred(s, col(e)));
            });
        Kokkos::parallel_for("mcheck_conv",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nc}),
            KOKKOS_LAMBDA(int s, int r) {
                unsigned int want = (syn_f(s,r) > 0.5f) ? 1u : 0u;
                if (row_par(s,r) != want) conv(s) = 0;
            });
        // Host-blocking early-exit count only every few iterations.
        if (iter % 8 == 7 || iter == max_iter - 1) {
            int num_nc = 0;
            Kokkos::parallel_reduce("mcount_nc", B,
                KOKKOS_LAMBDA(int s, int& acc) { if (!conv(s)) ++acc; },
                num_nc);
            if (num_nc == 0) break;
        }
    }
    Kokkos::fence();
}

// ─── Relay BP ─────────────────────────────────────────────────────────────────
// splitmix64 — stateless hash for per-(leg,shot,variable) gamma draws
KOKKOS_INLINE_FUNCTION uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// A shot is done once it holds stop_nconv converged solutions (never, if <= 0).
KOKKOS_INLINE_FUNCTION bool relay_done(int nconv, int stop) {
    return stop > 0 && nconv >= stop;
}

// Record the converged shots' solutions: keep the lowest prior weight seen.
static void relay_record(int B, int nb, const EdgeTable& et, int stop, BpWorkspace& ws) {
    auto conv = ws.conv;  auto nconv = ws.nconv;  auto pred = ws.pred;
    auto best = ws.best_pred;  auto best_w = ws.best_w;  auto ch = et.channel_llr;
    Kokkos::parallel_for("relay_record", B, KOKKOS_LAMBDA(int s) {
        if (!conv(s) || relay_done(nconv(s), stop)) return;
        float w = 0.0f;
        for (int j = 0; j < nb; ++j) if (pred(s, j)) w += ch(j);
        if (nconv(s) == 0 || w < best_w(s)) {
            best_w(s) = w;
            for (int j = 0; j < nb; ++j) best(s, j) = pred(s, j);
        }
        nconv(s) += 1;
    });
}

BpResult relay_bp_decode_batch(
    const EdgeTable& et,
    const Kokkos::View<uint8_t**, DeviceSpace>& H_dev,
    const std::vector<uint8_t>& syndromes,
    int B,
    const RelayConfig& cfg,
    BpWorkspace& ws,
    OsdWorkspace& osd_ws)
{
    upload_syndromes(syndromes, B, ws);

    const int nb = et.num_bits, E = et.num_edges;
    const int stop = cfg.stop_nconv;
    auto pair_B  = Kokkos::make_pair(0, B);
    auto tlr_sub = Kokkos::subview(ws.total_llr, pair_B, Kokkos::ALL());
    auto old_sub = Kokkos::subview(ws.tlr_old,   pair_B, Kokkos::ALL());
    auto gam_sub = Kokkos::subview(ws.gamma,     pair_B, Kokkos::ALL());
    auto nconv_sub = Kokkos::subview(ws.nconv,   pair_B);
    Kokkos::deep_copy(nconv_sub, 0);

    auto ch_llr = et.channel_llr;
    auto col    = et.edge_col;
    auto v2c    = ws.v2c;
    auto c2v    = ws.c2v;
    auto conv   = ws.conv;
    auto nconv  = ws.nconv;
    auto tlr_old = ws.tlr_old;

    // Phase 1: pre_iter iterations of BP, with uniform memory gamma0 if asked for.
    if (cfg.gamma0 != 0.0f) {
        bp_run(et, B, 0, ws);   // messages from the priors, conv cleared
        Kokkos::deep_copy(gam_sub, cfg.gamma0);
        Kokkos::parallel_for("relay_pre_memory",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
            KOKKOS_LAMBDA(int s, int j) { tlr_old(s,j) = ch_llr(j); });
        bp_run_memory(et, B, cfg.pre_iter, ws);
    } else {
        bp_run(et, B, cfg.pre_iter, ws);
    }
    relay_record(B, nb, et, stop, ws);

    // Phase 2: relay legs.  Each leg restarts the messages from the priors and
    // keeps only the previous posterior, blended in through a fresh gamma field.
    Kokkos::deep_copy(old_sub, tlr_sub);
    auto gam = ws.gamma;
    const uint64_t seed = cfg.seed ^ 0xdeadbeefcafeULL;
    const float gmin = cfg.gamma_min, grange = cfg.gamma_max - cfg.gamma_min;

    for (int leg = 0; leg < cfg.num_legs; ++leg) {
        int active = 0;
        Kokkos::parallel_reduce("relay_active", B,
            KOKKOS_LAMBDA(int s, int& acc) { if (!relay_done(nconv(s), stop)) ++acc; },
            active);
        if (active == 0) break;

        Kokkos::parallel_for("relay_gamma",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
            KOKKOS_LAMBDA(int s, int j) {
                uint64_t h = splitmix64(
                    seed + (uint64_t)leg * 0x100000001b3ULL
                         + (uint64_t)s   * 0x9e3779b97f4a7c15ULL
                         + (uint64_t)j);
                float u = (float)(h >> 40) * (1.0f / 16777216.0f);
                gam(s,j) = gmin + u * grange;
            });
        Kokkos::parallel_for("relay_leg_messages",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,E}),
            KOKKOS_LAMBDA(int s, int e) {
                if (relay_done(nconv(s), stop)) return;
                v2c(s,e) = ch_llr(col(e));
                c2v(s,e) = 0.0f;
            });
        Kokkos::parallel_for("relay_leg_reopen", B,
            KOKKOS_LAMBDA(int s) { if (!relay_done(nconv(s), stop)) conv(s) = 0; });

        bp_run_memory(et, B, cfg.leg_max_iter, ws);
        relay_record(B, nb, et, stop, ws);
    }

    // Phase 3: shots with a converged solution return their best one; the rest
    // go to OSD-0 on the last leg's posteriors.  converged = any leg converged.
    auto pred = ws.pred;  auto best = ws.best_pred;
    Kokkos::parallel_for("relay_best",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{B,nb}),
        KOKKOS_LAMBDA(int s, int j) { if (nconv(s) > 0) pred(s,j) = best(s,j); });
    Kokkos::parallel_for("relay_conv", B,
        KOKKOS_LAMBDA(int s) { conv(s) = (nconv(s) > 0) ? 1 : 0; });
    Kokkos::fence();
    osd0_run_all(B, et.num_checks, nb, H_dev, et.channel_llr, ws, osd_ws);
    return download_result(B, nb, ws);
}

} // namespace asy
