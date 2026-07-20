#include "scheduler.hpp"
#include <stdexcept>
#include <random>
#include <algorithm>

namespace asy {

CssProblem make_css_problem(
    int nbits, int ncx, int ncz,
    const std::vector<uint8_t>& Hx_dense,
    const std::vector<uint8_t>& Hz_dense,
    const std::vector<uint8_t>& obs_x,
    const std::vector<uint8_t>& obs_z,
    float p,
    int tick_slack)
{
    CssProblem prob;
    prob.nbits = nbits;
    prob.ncx = ncx;
    prob.ncz = ncz;
    prob.p = p;
    prob.obs_x = obs_x;
    prob.obs_z = obs_z;

    prob.supp.assign(ncx + ncz, {});
    std::vector<int> xr, xc, zr, zc;
    int max_w = 0, deg_total;
    std::vector<int> gate_deg(nbits, 0);
    for (int c = 0; c < ncx; ++c) {
        for (int q = 0; q < nbits; ++q)
            if (Hx_dense[(size_t)c * nbits + q]) {
                prob.supp[c].push_back(q);
                xr.push_back(c); xc.push_back(q);
                ++gate_deg[q];
            }
        max_w = std::max(max_w, (int)prob.supp[c].size());
    }
    for (int c = 0; c < ncz; ++c) {
        for (int q = 0; q < nbits; ++q)
            if (Hz_dense[(size_t)c * nbits + q]) {
                prob.supp[ncx + c].push_back(q);
                zr.push_back(c); zc.push_back(q);
                ++gate_deg[q];
            }
        max_w = std::max(max_w, (int)prob.supp[ncx + c].size());
    }
    prob.num_ticks = max_w + tick_slack;

    // BP channel priors: marginal single-qubit flip probability — each
    // DEPOLARIZE2 the qubit touches contributes 8p/15 (X- or Z-component
    // depending on block), the before-round DEPOLARIZE1 contributes 2p/3.
    std::vector<float> prior(nbits);
    for (int q = 0; q < nbits; ++q) {
        float pq = 2.0f * p / 3.0f + (float)gate_deg[q] * 8.0f * p / 15.0f;
        prior[q] = std::min(pq, 0.5f);
    }
    prob.et_x = EdgeTable::from_csc(ncx, nbits, xr, xc, prior);
    prob.et_z = EdgeTable::from_csc(ncz, nbits, zr, zc, prior);
    deg_total = 0; (void)deg_total;
    return prob;
}

// Pauli-frame Monte Carlo of one extraction round + GPU BP decode.
//
// Frame bits: (xq, zq) on data, (xa, za) on the active ancilla block.
// CNOT propagation (control → target):  X_c → X_c X_t,  Z_t → Z_c Z_t.
//   X phase: ancilla controls data.  Hooks: an X on the ancilla spreads
//     onto every data qubit whose CNOT comes later; Z on data climbs onto
//     the ancilla (that *is* the syndrome readout, za accumulates it).
//   Z phase: data controls ancilla.  Hooks: a Z on the ancilla spreads
//     back onto later data qubits; X on data climbs onto the ancilla.
// DEPOLARIZE2(p) after each CNOT draws one of 15 two-qubit Paulis w.p. p/15.
float evaluate_css_schedule(
    const CssProblem& prob,
    const std::vector<int>& schedule,
    int nshots,
    int max_bp_iter,
    int batch_size,
    uint64_t seed)
{
    const int nb = prob.nbits, ncx = prob.ncx, ncz = prob.ncz;
    const int nt = prob.num_ticks;
    if ((int)schedule.size() != (ncx + ncz) * nt)
        throw std::runtime_error("evaluate_css_schedule: schedule length must be "
                                 "(ncx + ncz) * num_ticks");
    const float p = prob.p;
    const float p15 = p / 15.0f;

    // Per-phase, per-tick CNOT lists (check, qubit)
    auto tick_list = [&](int row0, int nrows) {
        std::vector<std::vector<std::pair<int,int>>> ticks(nt);
        for (int c = 0; c < nrows; ++c)
            for (int t = 0; t < nt; ++t) {
                int q = schedule[(size_t)(row0 + c) * nt + t];
                if (q >= 0 && q < nb) ticks[t].push_back({c, q});
            }
        return ticks;
    };
    auto x_ticks = tick_list(0, ncx);
    auto z_ticks = tick_list(ncx, ncz);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    // Two-qubit depolarizing: index 1..15 → (P_a, P_b) with P ∈ {I,X,Z,Y}
    // encoded as (x bit, z bit) pairs.
    auto depolarize2 = [&](uint8_t& xa, uint8_t& za, uint8_t& xb, uint8_t& zb) {
        if (uni(rng) >= p) return;
        int k = 1 + (int)(uni(rng) * 15.0f);
        if (k > 15) k = 15;
        xa ^= (k >> 3) & 1; za ^= (k >> 2) & 1;
        xb ^= (k >> 1) & 1; zb ^= k & 1;
    };

    std::vector<uint8_t> xq(nb), zq(nb), xa(std::max(ncx, ncz)), za(std::max(ncx, ncz));
    std::vector<uint8_t> syn_x((size_t)nshots * ncx);
    std::vector<uint8_t> syn_z((size_t)nshots * ncz);
    std::vector<uint8_t> true_ox(nshots), true_oz(nshots);

    for (int s = 0; s < nshots; ++s) {
        std::fill(xq.begin(), xq.end(), 0);
        std::fill(zq.begin(), zq.end(), 0);

        // before-round data depolarizing
        for (int q = 0; q < nb; ++q)
            if (uni(rng) < p) {
                int k = 1 + (int)(uni(rng) * 3.0f); if (k > 3) k = 3;
                xq[q] ^= (k >> 1) & 1; zq[q] ^= k & 1;
            }

        // ── X phase: CNOT ancilla → data ──
        std::fill(xa.begin(), xa.begin() + ncx, 0);
        std::fill(za.begin(), za.begin() + ncx, 0);
        for (int t = 0; t < nt; ++t)
            for (auto& [c, q] : x_ticks[t]) {
                xq[q] ^= xa[c];           // X_c → X_c X_t (ancilla hook onto data)
                za[c] ^= zq[q];           // Z_t → Z_c Z_t (syndrome accumulation)
                depolarize2(xa[c], za[c], xq[q], zq[q]);
            }
        uint8_t* sx = &syn_x[(size_t)s * ncx];
        for (int c = 0; c < ncx; ++c)
            sx[c] = za[c] ^ (uni(rng) < p ? 1 : 0);   // measurement flip

        // ── Z phase: CNOT data → ancilla ──
        std::fill(xa.begin(), xa.begin() + ncz, 0);
        std::fill(za.begin(), za.begin() + ncz, 0);
        for (int t = 0; t < nt; ++t)
            for (auto& [c, q] : z_ticks[t]) {
                xa[c] ^= xq[q];           // X_c → X_c X_t (syndrome accumulation)
                zq[q] ^= za[c];           // Z_t → Z_c Z_t (ancilla hook onto data)
                depolarize2(xq[q], zq[q], xa[c], za[c]);
            }
        uint8_t* sz = &syn_z[(size_t)s * ncz];
        for (int c = 0; c < ncz; ++c)
            sz[c] = xa[c] ^ (uni(rng) < p ? 1 : 0);

        // true observable flips (final data readout assumed perfect)
        uint8_t ox = 0, oz = 0;
        for (int q = 0; q < nb; ++q) {
            ox ^= (zq[q] & prob.obs_x[q]);   // Z errors flip logical X
            oz ^= (xq[q] & prob.obs_z[q]);   // X errors flip logical Z
        }
        true_ox[s] = ox; true_oz[s] = oz;
    }

    // ── decode both blocks with GPU BP, count either-observable failures ──
    std::vector<uint8_t> wrong(nshots, 0);
    auto decode_block = [&](const EdgeTable& et, int nc,
                            std::vector<uint8_t>& syn,
                            std::vector<uint8_t>& truth,
                            const std::vector<uint8_t>& obs_mask) {
        BpWorkspace ws(std::min(batch_size, nshots), nc, nb, et.num_edges);
        for (int start = 0; start < nshots; start += batch_size) {
            int B = std::min(batch_size, nshots - start);
            std::vector<uint8_t> batch(syn.begin() + (size_t)start * nc,
                                       syn.begin() + (size_t)(start + B) * nc);
            BpResult res = bp_decode_batch(et, batch, B, max_bp_iter, ws);
            for (int s = 0; s < B; ++s) {
                uint8_t pred = 0;
                for (int q = 0; q < nb; ++q)
                    pred ^= (res.predictions[(size_t)s * nb + q] & obs_mask[q]);
                if (pred != truth[start + s]) wrong[start + s] = 1;
            }
        }
    };
    decode_block(prob.et_x, ncx, syn_x, true_ox, prob.obs_x);
    decode_block(prob.et_z, ncz, syn_z, true_oz, prob.obs_z);

    int n_errors = 0;
    for (int s = 0; s < nshots; ++s) n_errors += wrong[s];
    return (float)n_errors / (float)nshots;
}

} // namespace asy
