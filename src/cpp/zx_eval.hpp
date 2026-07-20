// zx_eval — ZX stabilizer-rank amplitude evaluation (the ZX-calculus part).
//
// Exact ℤ[ω] int64 coefficient arithmetic (ω = e^{iπ/4}) + the per-graph
// amplitude formula, mirroring tsim.core.exact_scalar bit-for-bit. This is the
// "what to compute"; the Kokkos runtime (GPU buffer management, fused sampling
// kernels, nanobind bindings) lives in kokkos_sim.cpp.
//
// Amplitude of one graph = Σ_g  ω^pf_phase · ff_g · 2^{power2_g}
//      · Π(1+ω^{4·par+φ})        (NodePhases)
//      · ω^{Σ c·par}             (HalfPiPhases)
//      · Π(−1)^{ψ·φ}             (PiProducts)
//      · Π(1+ω^α+ω^β−ω^{α+β})    (PhasePairs)
// with power-of-2 renormalisation; the final complex cast uses basis
// (1, ω, i, ω̄): re = a + (b+d)/√2, im = c + (b−d)/√2.
#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cmath>

// CudaSpace on CUDA builds, HostSpace on CPU-only Kokkos builds
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

// ─── ℤ[ω] arithmetic (ω = e^{iπ/4}); coeffs (a,b,c,d) = a+bω+cω²+dω³ ────────
struct Zw { long long a, b, c, d; };

KOKKOS_INLINE_FUNCTION Zw zw_mul(const Zw& x, const Zw& y) {
    return Zw{
        x.a*y.a + x.b*y.d - x.c*y.c + x.d*y.b,
        x.a*y.b + x.b*y.a + x.c*y.d + x.d*y.c,
        x.a*y.c + x.b*y.b + x.c*y.a - x.d*y.d,
        x.a*y.d - x.b*y.c - x.c*y.b + x.d*y.a};
}
// one renormalisation step: divide by 2 when all even (and not all zero)
KOKKOS_INLINE_FUNCTION void zw_reduce(Zw& x, long long& pow2) {
    if (((x.a | x.b | x.c | x.d) != 0) &&
        ((x.a & 1) == 0) && ((x.b & 1) == 0) &&
        ((x.c & 1) == 0) && ((x.d & 1) == 0)) {
        x.a >>= 1; x.b >>= 1; x.c >>= 1; x.d >>= 1; pow2 += 1;
    }
}
KOKKOS_INLINE_FUNCTION Zw zw_unit(int k) {           // ω^k, k in 0..7
    const long long t[8][4] = {
        {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,-1},
        {-1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,1}};
    return Zw{t[k][0], t[k][1], t[k][2], t[k][3]};
}

// ─── per-graph metadata (indices into the concatenated buffers) ──────────────
// meta layout per graph (int32 × NMETA):
enum {
    M_G = 0,        // number of stabilizer terms (graphs in tsim speak)
    M_P,            // parameter count of this step graph
    M_TNP, M_THP, M_TPP, M_TPH,          // max terms per family
    M_NP_PH, M_NP_PAR, M_NP_CNT,         // offsets: nodephase phases/params/counts
    M_HP_CO, M_HP_PAR,                   // halfpi coeffs/params
    M_PP_PSC, M_PP_PSP, M_PP_PHC, M_PP_PHP,
    M_PH_AL, M_PH_ALP, M_PH_BE, M_PH_BEP, M_PH_CNT,
    M_PF_IDX, M_PF_FF, M_PF_P2, M_PF_APX,  // prefactor offsets
    M_HAS_APX,
    NMETA
};

// ─── device-side component (POD view of a component's buffers) ───────────────
// param bit j: j<F → f(s,j); j<F+nm → m bit; j==F+nm → trying bit
struct DevComponent {
    int n_out, F, n_graphs;
    Kokkos::View<const int32_t*, DeviceSpace> meta;
    Kokkos::View<const uint8_t*, DeviceSpace> u8;
    Kokkos::View<const int32_t*, DeviceSpace> i32;
    Kokkos::View<const double*,  DeviceSpace> apx;
};

// ─── device evaluation of one graph for one shot ─────────────────────────────
template <class FV, class MV>
KOKKOS_INLINE_FUNCTION
void eval_graph(const DevComponent& C, int gi, int s, int nm, int trying,
                const FV& f, const MV& m, double& out_re, double& out_im)
{
    const int32_t* md = &C.meta(gi * NMETA);
    const int G = md[M_G], P = md[M_P];
    const int Tnp = md[M_TNP], Thp = md[M_THP], Tpp = md[M_TPP], Tph = md[M_TPH];
    const int F = C.F;

    // param bit lookup
    auto pbit = [&](int j) -> int {
        if (j < F)       return (int)f(s, j);
        if (j < F + nm)  return (int)m(s, j - F);
        return trying;
    };
    auto parity = [&](long long base, int t, int Tdim, int j_unused) -> int {
        (void)j_unused; (void)Tdim;
        int acc = 0;
        const uint8_t* row = &C.u8(base + (size_t)t * P);
        for (int j = 0; j < P; ++j) acc ^= (row[j] & pbit(j));
        return acc;
    };

    double sum_re = 0.0, sum_im = 0.0;
    const double SQ = 0.70710678118654752440;   // cos(π/4)

    for (int g = 0; g < G; ++g) {
        Zw acc{1, 0, 0, 0};
        long long pw = 0;

        // NodePhases: Π (1 + ω^{(4·par + φ) mod 8})
        {
            long long phb = md[M_NP_PH]  + (long long)g * Tnp;
            long long pab = md[M_NP_PAR] + (long long)g * Tnp * P;
            int cnt = C.i32(md[M_NP_CNT] + g);
            for (int t = 0; t < cnt; ++t) {
                int par = parity(pab, t, Tnp, 0);
                int k = (4 * par + C.u8(phb + t)) & 7;
                Zw u = zw_unit(k); u.a += 1;            // 1 + ω^k
                acc = zw_mul(acc, u); zw_reduce(acc, pw);
            }
        }
        // HalfPiPhases: ω^{Σ (par·coeff) mod 8}
        {
            long long cob = md[M_HP_CO]  + (long long)g * Thp;
            long long pab = md[M_HP_PAR] + (long long)g * Thp * P;
            int tot = 0;
            for (int t = 0; t < Thp; ++t) {
                int par = parity(pab, t, Thp, 0);
                tot += (par * C.u8(cob + t)) & 7;
            }
            acc = zw_mul(acc, zw_unit(tot & 7)); zw_reduce(acc, pw);
        }
        // PiProducts: (−1)^{Σ ψ·φ}
        {
            long long pscb = md[M_PP_PSC] + (long long)g * Tpp;
            long long pspb = md[M_PP_PSP] + (long long)g * Tpp * P;
            long long phcb = md[M_PP_PHC] + (long long)g * Tpp;
            long long phpb = md[M_PP_PHP] + (long long)g * Tpp * P;
            int e = 0;
            for (int t = 0; t < Tpp; ++t) {
                int psi = (C.u8(pscb + t) + parity(pspb, t, Tpp, 0)) & 1;
                int phi = (C.u8(phcb + t) + parity(phpb, t, Tpp, 0)) & 1;
                e ^= (psi & phi);
            }
            if (e) { acc.a = -acc.a; acc.b = -acc.b; acc.c = -acc.c; acc.d = -acc.d; }
        }
        // PhasePairs: Π (1 + ω^α + ω^β − ω^{α+β})
        {
            long long alb  = md[M_PH_AL]  + (long long)g * Tph;
            long long alpb = md[M_PH_ALP] + (long long)g * Tph * P;
            long long beb  = md[M_PH_BE]  + (long long)g * Tph;
            long long bepb = md[M_PH_BEP] + (long long)g * Tph * P;
            int cnt = C.i32(md[M_PH_CNT] + g);
            for (int t = 0; t < cnt; ++t) {
                int al = (C.u8(alb + t) + 4 * parity(alpb, t, Tph, 0)) & 7;
                int be = (C.u8(beb + t) + 4 * parity(bepb, t, Tph, 0)) & 7;
                Zw u = zw_unit(al), v = zw_unit(be), w = zw_unit((al + be) & 7);
                Zw term{1 + u.a + v.a - w.a, u.b + v.b - w.b,
                        u.c + v.c - w.c,     u.d + v.d - w.d};
                acc = zw_mul(acc, term); zw_reduce(acc, pw);
            }
        }
        // prefactor: ω^idx · floatfactor (ℤ[ω] int32×4)
        acc = zw_mul(acc, zw_unit(C.u8(md[M_PF_IDX] + g) & 7));
        {
            long long fb = md[M_PF_FF] + (long long)g * 4;
            Zw ff{C.i32(fb), C.i32(fb + 1), C.i32(fb + 2), C.i32(fb + 3)};
            acc = zw_mul(acc, ff); zw_reduce(acc, pw);
        }

        // → complex, × 2^{pw + power2} (× approx factor if present).
        double re = (double)acc.a + ((double)acc.b + (double)acc.d) * SQ;
        double im = (double)acc.c + ((double)acc.b - (double)acc.d) * SQ;
        double scale = exp2((double)(pw + C.i32(md[M_PF_P2] + g)));
        re *= scale; im *= scale;
        if (md[M_HAS_APX]) {
            double ar = C.apx(md[M_PF_APX] + 2 * g);
            double ai = C.apx(md[M_PF_APX] + 2 * g + 1);
            double r2 = re * ar - im * ai;
            im = re * ai + im * ar; re = r2;
        }
        sum_re += re; sum_im += im;
    }
    out_re = sum_re; out_im = sum_im;
}
