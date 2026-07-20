// kokkos_sim — Kokkos/CUDA runtime for tsim's compiled ZX stabilizer-rank
// programs (the Kokkos part: GPU buffer management, the fused per-component
// sampling kernel, and the nanobind bindings). The ZX-calculus amplitude math
// (exact ℤ[ω] arithmetic + eval_graph) lives in include/zx_eval.hpp.
//
// The symbolic compile (pyzx graph reduction + magic-state decomposition) stays
// in Python; this module replaces the JAX evaluator and autoregressive sampler
// with a single fused GPU kernel per component.
//
// Data contract (see src/python/ksim):
//   A component owns graphs[0..n_out]; graphs[0] is the normalisation, graph
//   i+1 has outputs 0..i plugged plus one "trying" bit.  Graph i's parameter
//   vector is [f_bits (F) | m_bits (i) | trying] — never materialised here;
//   bits are looked up on the fly inside the parity loops (see eval_graph).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "zx_eval.hpp"   // ZX-calculus: Zw arithmetic, meta enum,
                                    // DevComponent, eval_graph

namespace nb = nanobind;
using namespace nb::literals;

using DevU8  = Kokkos::View<uint8_t*,  DeviceSpace>;
using DevI32 = Kokkos::View<int32_t*,  DeviceSpace>;
using DevF64 = Kokkos::View<double*,   DeviceSpace>;

// ─── Kokkos guard ─────────────────────────────────────────────────────────────
static void ensure_kokkos() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        std::atexit([] { if (Kokkos::is_initialized()) Kokkos::finalize(); });
    }
}

KOKKOS_INLINE_FUNCTION double rng_uniform(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (double)(s >> 11) * (1.0 / (double)(1ULL << 53));
}

// ─── host-side component: owns the concatenated GPU buffers ──────────────────
struct Component {
    int n_out = 0;          // number of outputs (graphs.size() - 1)
    int F     = 0;          // f-parameter count
    int n_graphs = 0;
    DevI32 meta;            // (n_graphs × NMETA)
    DevU8  u8buf;           // all uint8 arrays concatenated
    DevI32 i32buf;          // counts / floatfactor / power2
    DevF64 apxbuf;          // approx floatfactors as interleaved re,im
};

// ─── host-side component builder + sampling kernels ──────────────────────────
using NpU8  = nb::ndarray<const uint8_t,  nb::c_contig>;
using NpI32 = nb::ndarray<const int32_t,  nb::c_contig>;

struct KokkosTsimComponent {
    Component comp;
    DevComponent dev;

    KokkosTsimComponent(nb::list graphs, int F) {
        ensure_kokkos();
        comp.F = F;
        comp.n_graphs = (int)graphs.size();
        comp.n_out = comp.n_graphs - 1;

        std::vector<int32_t> meta;
        std::vector<uint8_t> u8;
        std::vector<int32_t> i32;
        std::vector<double>  apx;

        auto push_u8 = [&](nb::handle a) -> int64_t {
            auto arr = nb::cast<NpU8>(a);
            int64_t off = (int64_t)u8.size();
            u8.insert(u8.end(), arr.data(), arr.data() + arr.size());
            return off;
        };
        auto push_i32 = [&](nb::handle a) -> int64_t {
            auto arr = nb::cast<NpI32>(a);
            int64_t off = (int64_t)i32.size();
            i32.insert(i32.end(), arr.data(), arr.data() + arr.size());
            return off;
        };

        for (auto gh : graphs) {
            nb::dict g = nb::cast<nb::dict>(gh);
            auto shape = [&](const char* key, int dim) -> int {
                auto arr = nb::cast<nb::ndarray<>>(g[key]);
                return dim < (int)arr.ndim() ? (int)arr.shape(dim) : 1;
            };
            int32_t m[NMETA] = {};
            m[M_G]   = shape("pf_phase_idx", 0);
            m[M_P]   = shape("np_params", 2);
            m[M_TNP] = shape("np_phases", 1);
            m[M_THP] = shape("hp_coeffs", 1);
            m[M_TPP] = shape("pp_psi_c", 1);
            m[M_TPH] = shape("ph_alpha", 1);
            m[M_NP_PH]  = (int32_t)push_u8(g["np_phases"]);
            m[M_NP_PAR] = (int32_t)push_u8(g["np_params"]);
            m[M_NP_CNT] = (int32_t)push_i32(g["np_counts"]);
            m[M_HP_CO]  = (int32_t)push_u8(g["hp_coeffs"]);
            m[M_HP_PAR] = (int32_t)push_u8(g["hp_params"]);
            m[M_PP_PSC] = (int32_t)push_u8(g["pp_psi_c"]);
            m[M_PP_PSP] = (int32_t)push_u8(g["pp_psi_p"]);
            m[M_PP_PHC] = (int32_t)push_u8(g["pp_phi_c"]);
            m[M_PP_PHP] = (int32_t)push_u8(g["pp_phi_p"]);
            m[M_PH_AL]  = (int32_t)push_u8(g["ph_alpha"]);
            m[M_PH_ALP] = (int32_t)push_u8(g["ph_alpha_p"]);
            m[M_PH_BE]  = (int32_t)push_u8(g["ph_beta"]);
            m[M_PH_BEP] = (int32_t)push_u8(g["ph_beta_p"]);
            m[M_PH_CNT] = (int32_t)push_i32(g["ph_counts"]);
            m[M_PF_IDX] = (int32_t)push_u8(g["pf_phase_idx"]);
            m[M_PF_FF]  = (int32_t)push_i32(g["pf_floatfactor"]);
            m[M_PF_P2]  = (int32_t)push_i32(g["pf_power2"]);
            {   // approx factors: complex64 viewed as float32 pairs (re, im)
                auto arr = nb::cast<nb::ndarray<const float, nb::c_contig>>(
                    g["pf_approx_f32"]);
                m[M_PF_APX] = (int32_t)apx.size();   // flat double offset
                for (size_t k = 0; k < arr.size(); ++k)
                    apx.push_back((double)arr.data()[k]);
            }
            m[M_HAS_APX] = nb::cast<bool>(g["pf_has_approx"]) ? 1 : 0;
            meta.insert(meta.end(), m, m + NMETA);
        }

        auto to_dev = [&](auto& host, auto tag) {
            using V = decltype(tag);
            Kokkos::View<typename V::value_type*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                h(host.data(), host.size());
            V d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "buf"),
                host.size());
            Kokkos::deep_copy(d, h);
            return d;
        };
        if (u8.empty())  u8.push_back(0);
        if (i32.empty()) i32.push_back(0);
        if (apx.empty()) apx.push_back(0.0);
        comp.meta   = to_dev(meta, DevI32{});
        comp.u8buf  = to_dev(u8,   DevU8{});
        comp.i32buf = to_dev(i32,  DevI32{});
        comp.apxbuf = to_dev(apx,  DevF64{});

        dev = DevComponent{comp.n_out, comp.F, comp.n_graphs,
                           comp.meta, comp.u8buf, comp.i32buf, comp.apxbuf};
    }

    // debug: evaluate graph gi on explicit params (B, P) → |amplitude| (B,)
    // params are packed as [f | m | trying]; we pass them through f_d with
    // F temporarily widened, nm=0, trying ignored by making P<=F.
    nb::ndarray<nb::numpy, double> eval_abs(
        int gi, nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig> params)
    {
        int B = (int)params.shape(0), P = (int)params.shape(1);
        Kokkos::View<uint8_t**, Kokkos::LayoutRight, DeviceSpace>
            f_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "fp"), B,
                std::max(P, 1));
        Kokkos::View<const uint8_t**, Kokkos::LayoutRight, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h(params.data(), B, P);
        if (P > 0)
            Kokkos::deep_copy(Kokkos::subview(
                f_d, Kokkos::ALL(), Kokkos::make_pair(0, P)), h);
        Kokkos::View<double*, DeviceSpace> out_d("o", B);
        DevComponent C = dev;
        C.F = P;   // all params come through the f view
        Kokkos::parallel_for("ktsim_eval", B, KOKKOS_LAMBDA(int s) {
            double re, im;
            eval_graph(C, gi, s, 0, 0, f_d, f_d, re, im);
            out_d(s) = sqrt(re * re + im * im);
        });
        Kokkos::fence();
        auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out_d);
        double* out = new double[B];
        for (int s = 0; s < B; ++s) out[s] = oh(s);
        nb::capsule owner(out, [](void* p) noexcept { delete[] (double*)p; });
        size_t shp[1] = {(size_t)B};
        return nb::ndarray<nb::numpy, double>(out, 1, shp, owner);
    }

    // f_sel: (B, F) uint8 → samples (B, n_out) uint8
    nb::ndarray<nb::numpy, uint8_t> sample(
        nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig> f_sel,
        uint64_t seed)
    {
        int B = (int)f_sel.shape(0);
        if ((int)f_sel.shape(1) != comp.F)
            throw std::invalid_argument("f_sel column count != F");
        int n_out = comp.n_out;

        Kokkos::View<uint8_t**, Kokkos::LayoutRight, DeviceSpace>
            f_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "f"), B,
                std::max(comp.F, 1));
        {
            Kokkos::View<const uint8_t**, Kokkos::LayoutRight,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                h(f_sel.data(), B, comp.F);
            if (comp.F > 0)
                Kokkos::deep_copy(
                    Kokkos::subview(f_d, Kokkos::ALL(),
                                    Kokkos::make_pair(0, comp.F)), h);
        }
        Kokkos::View<uint8_t**, Kokkos::LayoutRight, DeviceSpace>
            m_d("m", B, std::max(n_out, 1));

        DevComponent C = dev;
        Kokkos::parallel_for("ktsim_sample", B, KOKKOS_LAMBDA(int s) {
            uint64_t rng = seed ^ ((uint64_t)s * 0x9e3779b97f4a7c15ULL);
            rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9ULL;
            rng ^= rng >> 27; rng = rng ? rng : 1ULL;

            double re, im;
            eval_graph(C, 0, s, 0, 0, f_d, m_d, re, im);
            double prev = sqrt(re * re + im * im);

            for (int i = 0; i < C.n_out; ++i) {
                eval_graph(C, i + 1, s, i, 1, f_d, m_d, re, im);
                double p1 = sqrt(re * re + im * im);
                int bit = rng_uniform(rng) < (p1 / prev) ? 1 : 0;
                m_d(s, i) = (uint8_t)bit;
                prev = bit ? p1 : (prev - p1);
            }
        });
        Kokkos::fence();

        auto m_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_d);
        uint8_t* out = new uint8_t[(size_t)B * std::max(n_out, 1)];
        for (int s = 0; s < B; ++s)
            for (int i = 0; i < n_out; ++i)
                out[(size_t)s * n_out + i] = m_h(s, i);
        nb::capsule owner(out, [](void* p) noexcept { delete[] (uint8_t*)p; });
        size_t shp[2] = {(size_t)B, (size_t)std::max(n_out, 1)};
        return nb::ndarray<nb::numpy, uint8_t>(out, 2, shp, owner);
    }
};

NB_MODULE(kokkos_sim, mod) {
    mod.doc() = "Kokkos/CUDA runtime for tsim compiled ZX stabilizer-rank programs";
    nb::class_<KokkosTsimComponent>(mod, "Component")
        .def(nb::init<nb::list, int>(), "graphs"_a, "num_f"_a)
        .def("sample", &KokkosTsimComponent::sample, "f_sel"_a, "seed"_a)
        .def("eval_abs", &KokkosTsimComponent::eval_abs, "graph"_a, "params"_a)
        .def_prop_ro("n_out",
                     [](const KokkosTsimComponent& c) { return c.comp.n_out; });
}
