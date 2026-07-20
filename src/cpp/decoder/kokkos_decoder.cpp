#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bp_decoders.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

static bool kokkos_initialised = false;

void ensure_kokkos() {
    if (!kokkos_initialised) {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
        std::atexit([]{ if (Kokkos::is_initialized()) Kokkos::finalize(); });
        kokkos_initialised = true;
    }
}

// Build EdgeTable from dense numpy uint8 H (nc×nb) and float32 priors (nb,)
asy::EdgeTable edge_table_from_numpy(
    nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> H_arr,
    nb::ndarray<float,   nb::ndim<1>, nb::c_contig> p_arr)
{
    int nc = (int)H_arr.shape(0);
    int nb = (int)H_arr.shape(1);
    const uint8_t* Hp = H_arr.data();
    const float*   pp = p_arr.data();

    std::vector<int>   row_idx, col_idx;
    std::vector<float> p_err(pp, pp + nb);
    for (int r = 0; r < nc; ++r)
        for (int c = 0; c < nb; ++c)
            if (Hp[r * nb + c]) { row_idx.push_back(r); col_idx.push_back(c); }

    return asy::EdgeTable::from_csc(nc, nb, row_idx, col_idx, p_err);
}

// OSD sub-batch size: enough teams to fill the GPU, capped so the bit-packed
// workspace (dominated by Hw: nc × ceil(nb/64) uint64 words per shot) stays
// within a fixed memory budget even for the largest codes.
int osd_chunk_size(int max_batch, int nc, int nb) {
    long long W = (nb + 63) / 64;
    long long per_shot = (long long)nc * W * 8;
    long long budget = 2LL << 30;   // 2 GiB
    int by_mem = (int)std::max(1LL, budget / std::max(1LL, per_shot));
    return std::min({1024, max_batch, by_mem});
}

// Convert (B×nb) uint8 predictions to numpy array via heap-owned buffer.
nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>
vec_to_ndarray(std::vector<uint8_t> data, int B, int nb) {
    auto* buf = new uint8_t[data.size()];
    std::copy(data.begin(), data.end(), buf);
    auto cap = nb::capsule(buf, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
    return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(buf, {(size_t)B, (size_t)nb}, cap);
}

nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>
vec1_to_ndarray(std::vector<uint8_t> data) {
    auto* buf = new uint8_t[data.size()];
    std::copy(data.begin(), data.end(), buf);
    auto cap = nb::capsule(buf, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
    return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(buf, {data.size()}, cap);
}

// Convert (B×nb) int32 cluster ids to numpy array via heap-owned buffer.
nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>
vec_to_ndarray_i32(std::vector<int32_t> data, int B, int nb) {
    auto* buf = new int32_t[data.size()];
    std::copy(data.begin(), data.end(), buf);
    auto cap = nb::capsule(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
    return nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(buf, {(size_t)B, (size_t)nb}, cap);
}

// ─── BpOsd0Decoder ────────────────────────────────────────────────────────────
struct BpOsd0Decoder {
    asy::EdgeTable    et;
    asy::BpWorkspace  ws;
    asy::OsdWorkspace osd_ws;
    Kokkos::View<uint8_t**, DeviceSpace> H_dev;
    int max_bp_iter_;
    int max_batch_;

    BpOsd0Decoder(
        nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> H,
        nb::ndarray<float,   nb::ndim<1>, nb::c_contig> priors,
        int max_iter = 30,
        int max_batch = 4096)
        : max_bp_iter_(max_iter), max_batch_(max_batch)
    {
        ensure_kokkos();
        et     = edge_table_from_numpy(H, priors);
        ws     = asy::BpWorkspace(max_batch, et.num_checks, et.num_bits, et.num_edges);
        int osd_b = osd_chunk_size(max_batch, et.num_checks, et.num_bits);
        osd_ws = asy::OsdWorkspace(osd_b, et.num_checks, et.num_bits, max_batch);
        H_dev  = asy::build_H_dense(et);
    }

    nb::tuple decode_batch(nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> syn_arr) {
        int B  = (int)syn_arr.shape(0);
        int nc = (int)syn_arr.shape(1);
        if (nc != et.num_checks)
            throw std::invalid_argument("syndrome columns != num_checks");
        if (B > max_batch_)
            throw std::invalid_argument("batch size exceeds max_batch");

        std::vector<uint8_t> syn(syn_arr.data(), syn_arr.data() + (size_t)B * nc);
        auto res = asy::bp_osd0_decode_batch(et, H_dev, syn, B, max_bp_iter_, ws, osd_ws);

        return nb::make_tuple(
            vec_to_ndarray(std::move(res.predictions), B, et.num_bits),
            vec1_to_ndarray(std::move(res.converged)));
    }
};

// ─── RelayBpDecoder ───────────────────────────────────────────────────────────
struct RelayBpDecoder {
    asy::EdgeTable    et;
    asy::BpWorkspace  ws;
    asy::OsdWorkspace osd_ws;
    Kokkos::View<uint8_t**, DeviceSpace> H_dev;
    asy::RelayConfig  cfg;
    int max_batch_;

    RelayBpDecoder(
        nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> H,
        nb::ndarray<float,   nb::ndim<1>, nb::c_contig> priors,
        int max_batch   = 4096,
        int pre_iter    = 80,
        int num_legs    = 100,
        int leg_max_iter = 60,
        float gamma_min  = -0.24f,
        float gamma_max  =  0.66f,
        int stop_nconv   = 2,
        uint64_t seed    = 42)
        : max_batch_(max_batch)
    {
        ensure_kokkos();
        et  = edge_table_from_numpy(H, priors);
        ws  = asy::BpWorkspace(max_batch, et.num_checks, et.num_bits, et.num_edges);
        int osd_b = osd_chunk_size(max_batch, et.num_checks, et.num_bits);
        osd_ws = asy::OsdWorkspace(osd_b, et.num_checks, et.num_bits, max_batch);
        H_dev  = asy::build_H_dense(et);
        cfg    = { pre_iter, num_legs, leg_max_iter, gamma_min, gamma_max, stop_nconv, seed };
    }

    nb::tuple decode_batch(nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> syn_arr) {
        int B  = (int)syn_arr.shape(0);
        int nc = (int)syn_arr.shape(1);
        if (nc != et.num_checks) throw std::invalid_argument("syndrome columns != num_checks");
        if (B > max_batch_)     throw std::invalid_argument("batch size exceeds max_batch");

        std::vector<uint8_t> syn(syn_arr.data(), syn_arr.data() + (size_t)B * nc);
        auto res = asy::relay_bp_decode_batch(et, H_dev, syn, B, cfg, ws, osd_ws);

        return nb::make_tuple(
            vec_to_ndarray(std::move(res.predictions), B, et.num_bits),
            vec1_to_ndarray(std::move(res.converged)));
    }
};

// ─── BpLsdDecoder ─────────────────────────────────────────────────────────────
struct BpLsdDecoder {
    asy::EdgeTable    et;
    asy::BpWorkspace  ws;
    asy::OsdWorkspace osd_ws;
    Kokkos::View<uint8_t**, DeviceSpace> H_dev;
    asy::LsdConfig    lsd_cfg;
    int max_bp_iter_;
    int max_batch_;

    BpLsdDecoder(
        nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> H,
        nb::ndarray<float,   nb::ndim<1>, nb::c_contig> priors,
        int max_iter = 30,
        int max_batch = 4096,
        int max_cluster_bits = 20,
        int max_cluster_iters = 10)
        : max_bp_iter_(max_iter), max_batch_(max_batch)
    {
        ensure_kokkos();
        et  = edge_table_from_numpy(H, priors);
        ws  = asy::BpWorkspace(max_batch, et.num_checks, et.num_bits, et.num_edges);
        int osd_b = osd_chunk_size(max_batch, et.num_checks, et.num_bits);
        osd_ws = asy::OsdWorkspace(osd_b, et.num_checks, et.num_bits, max_batch);
        H_dev  = asy::build_H_dense(et);
        lsd_cfg = { max_cluster_bits, max_cluster_iters };
    }

    nb::tuple decode_batch(nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> syn_arr) {
        int B  = (int)syn_arr.shape(0);
        int nc = (int)syn_arr.shape(1);
        if (nc != et.num_checks) throw std::invalid_argument("syndrome columns != num_checks");
        if (B > max_batch_)     throw std::invalid_argument("batch size exceeds max_batch");

        std::vector<uint8_t> syn(syn_arr.data(), syn_arr.data() + (size_t)B * nc);
        auto res = asy::bp_lsd_decode_batch(et, H_dev, syn, B, max_bp_iter_,
                                             lsd_cfg, ws, osd_ws);
        return nb::make_tuple(
            vec_to_ndarray(std::move(res.predictions), B, et.num_bits),
            vec1_to_ndarray(std::move(res.converged)));
    }

    // Same decode, plus per-bit local (per-shot) 1-based cluster ids (0 = no
    // cluster -- includes every bit on shots where BP converged, since LSD
    // never runs there). See bp_decoders.hpp's bp_lsd_decode_batch docstring
    // for the Python-side reduction into cluster_sizes/cluster_llrs.
    nb::tuple decode_batch_with_stats(nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig> syn_arr) {
        int B  = (int)syn_arr.shape(0);
        int nc = (int)syn_arr.shape(1);
        if (nc != et.num_checks) throw std::invalid_argument("syndrome columns != num_checks");
        if (B > max_batch_)     throw std::invalid_argument("batch size exceeds max_batch");

        std::vector<uint8_t> syn(syn_arr.data(), syn_arr.data() + (size_t)B * nc);
        std::vector<int32_t> clusters;
        auto res = asy::bp_lsd_decode_batch(et, H_dev, syn, B, max_bp_iter_,
                                             lsd_cfg, ws, osd_ws, &clusters);
        return nb::make_tuple(
            vec_to_ndarray(std::move(res.predictions), B, et.num_bits),
            vec1_to_ndarray(std::move(res.converged)),
            vec_to_ndarray_i32(std::move(clusters), B, et.num_bits));
    }
};

} // anonymous namespace

NB_MODULE(kokkos_decoder, m) {
    m.doc() = "Kokkos GPU decoders: BP+OSD-0, Relay BP, BP+LSD";

    nb::class_<BpOsd0Decoder>(m, "BpOsd0Decoder")
        .def(nb::init<
                nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig>,
                nb::ndarray<float,   nb::ndim<1>, nb::c_contig>,
                int, int>(),
             "H"_a, "priors"_a, "max_iter"_a = 30, "max_batch"_a = 4096,
             "BP + OSD-0 decoder (Kokkos/CUDA). H: (nc,nb) uint8, priors: (nb,) float32.")
        .def("decode_batch", &BpOsd0Decoder::decode_batch, "syndromes"_a,
             "Decode batch. Returns (predictions (B,nb) uint8, converged (B,) uint8).");

    nb::class_<RelayBpDecoder>(m, "RelayBpDecoder")
        .def(nb::init<
                nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig>,
                nb::ndarray<float,   nb::ndim<1>, nb::c_contig>,
                int, int, int, int, float, float, int, uint64_t>(),
             "H"_a, "priors"_a,
             "max_batch"_a   = 4096,
             "pre_iter"_a    = 80,
             "num_legs"_a    = 100,
             "leg_max_iter"_a = 60,
             "gamma_min"_a   = -0.24f,
             "gamma_max"_a   =  0.66f,
             "stop_nconv"_a  = 2,
             "seed"_a        = (uint64_t)42,
             "Relay BP decoder: pre_iter standard BP + num_legs memory-BP legs with random gamma."
             " OSD-0 fallback on remaining non-converged shots.")
        .def("decode_batch", &RelayBpDecoder::decode_batch, "syndromes"_a,
             "Decode batch. Returns (predictions (B,nb) uint8, converged (B,) uint8).");

    nb::class_<BpLsdDecoder>(m, "BpLsdDecoder")
        .def(nb::init<
                nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig>,
                nb::ndarray<float,   nb::ndim<1>, nb::c_contig>,
                int, int, int, int>(),
             "H"_a, "priors"_a,
             "max_iter"_a          = 30,
             "max_batch"_a         = 4096,
             "max_cluster_bits"_a  = 20,
             "max_cluster_iters"_a = 10,
             "BP + LSD decoder. GPU BP then CPU cluster BFS + brute-force (≤ max_cluster_bits)"
             " or OSD-0 for larger clusters.")
        .def("decode_batch", &BpLsdDecoder::decode_batch, "syndromes"_a,
             "Decode batch. Returns (predictions (B,nb) uint8, converged (B,) uint8).")
        .def("decode_batch_with_stats", &BpLsdDecoder::decode_batch_with_stats, "syndromes"_a,
             "Decode batch with per-bit cluster ids. Returns (predictions (B,nb) uint8,"
             " converged (B,) uint8, clusters (B,nb) int32 -- 0 = no cluster).");
}
