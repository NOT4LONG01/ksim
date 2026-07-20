#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdlib>

#include "scheduler.hpp"

namespace nb = nanobind;
using NpU8   = nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>;
using NpU8v  = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>;

static bool s_kokkos_inited = false;
static void ensure_kokkos() {
    if (!s_kokkos_inited) {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
        std::atexit([]{ if (Kokkos::is_initialized()) Kokkos::finalize(); });
        s_kokkos_inited = true;
    }
}

static asy::CssProblem make_problem(
    NpU8 Hx_np, NpU8 Hz_np, NpU8v obs_x_np, NpU8v obs_z_np,
    float p, int tick_slack)
{
    ensure_kokkos();
    int ncx = (int)Hx_np.shape(0), nb_ = (int)Hx_np.shape(1);
    int ncz = (int)Hz_np.shape(0);
    if ((int)Hz_np.shape(1) != nb_)
        throw std::runtime_error("Hx and Hz must have the same number of columns");
    if ((int)obs_x_np.shape(0) != nb_ || (int)obs_z_np.shape(0) != nb_)
        throw std::runtime_error("obs masks must have H-column length");

    std::vector<uint8_t> Hx((size_t)ncx * nb_), Hz((size_t)ncz * nb_);
    for (int r = 0; r < ncx; ++r)
        for (int c = 0; c < nb_; ++c) Hx[(size_t)r * nb_ + c] = Hx_np(r, c);
    for (int r = 0; r < ncz; ++r)
        for (int c = 0; c < nb_; ++c) Hz[(size_t)r * nb_ + c] = Hz_np(r, c);

    std::vector<uint8_t> ox(nb_), oz(nb_);
    bool any_x = false, any_z = false;
    for (int j = 0; j < nb_; ++j) {
        ox[j] = obs_x_np(j) ? 1 : 0; any_x |= (bool)ox[j];
        oz[j] = obs_z_np(j) ? 1 : 0; any_z |= (bool)oz[j];
    }
    if (!any_x || !any_z)
        throw std::runtime_error("obs masks must be non-zero logical supports");

    return asy::make_css_problem(nb_, ncx, ncz, Hx, Hz, ox, oz, p, tick_slack);
}

// ─── MctsScheduler ────────────────────────────────────────────────────────────
// Joint schedule search over both check blocks.  Returned flat schedule:
// index = row * num_ticks + tick → data qubit (-1 idle); rows [0, ncx) are
// X checks, [ncx, ncx+ncz) are Z checks.  num_ticks resolves to
// (max check weight + tick_slack); read it back via num_ticks().
struct MctsScheduler {
    int _nshots, _max_bp_iter, _tick_slack;
    uint64_t _seed;
    float _best_score = -1.0f;
    int _num_ticks = 0;

    MctsScheduler(int nshots = 4096, int max_bp_iter = 30,
                  int tick_slack = 2, uint64_t seed = 42)
        : _nshots(nshots), _max_bp_iter(max_bp_iter),
          _tick_slack(tick_slack), _seed(seed)
    { ensure_kokkos(); }

    std::vector<int> schedule(NpU8 Hx, NpU8 Hz, NpU8v obs_x, NpU8v obs_z,
                              float p, int iters = 500) {
        auto prob = make_problem(Hx, Hz, obs_x, obs_z, p, _tick_slack);
        _num_ticks = prob.num_ticks;
        asy::MctsTree tree(prob, _nshots, _max_bp_iter, _seed);
        auto best = tree.run(iters);
        _best_score = tree.best_score();
        return best;
    }

    float best_score() const { return _best_score; }
    int num_ticks() const { return _num_ticks; }
};

// ─── evaluate_schedule ────────────────────────────────────────────────────────
static float py_evaluate_schedule(
    NpU8 Hx, NpU8 Hz,
    nb::list schedule_list,
    int num_ticks,
    NpU8v obs_x, NpU8v obs_z,
    float p,
    int nshots = 4096,
    int max_bp_iter = 30,
    int batch_size = 4096,
    uint64_t seed = 42)
{
    auto prob = make_problem(Hx, Hz, obs_x, obs_z, p,
                             /*tick_slack=*/0);
    if (num_ticks < prob.num_ticks)
        throw std::runtime_error("num_ticks below max check weight");
    prob.num_ticks = num_ticks;
    std::vector<int> sched;
    for (auto item : schedule_list) sched.push_back(nb::cast<int>(item));
    return asy::evaluate_css_schedule(prob, sched, nshots, max_bp_iter,
                                      batch_size, seed);
}

NB_MODULE(kokkos_mcts, m) {
    m.doc() = "Kokkos MCTS schedule optimizer (joint circuit-level objective)";

    nb::class_<MctsScheduler>(m, "MctsScheduler")
        .def(nb::init<int, int, int, uint64_t>(),
             nb::arg("nshots") = 4096, nb::arg("max_bp_iter") = 30,
             nb::arg("tick_slack") = 2, nb::arg("seed") = 42)
        .def("schedule", &MctsScheduler::schedule,
             nb::arg("Hx"), nb::arg("Hz"), nb::arg("obs_x"), nb::arg("obs_z"),
             nb::arg("p"), nb::arg("iters") = 500,
             "Joint MCTS over both blocks; returns flat schedule "
             "(row * num_ticks + tick → qubit, rows: X checks then Z checks).")
        .def("best_score", &MctsScheduler::best_score,
             "LER of the best schedule found by the last schedule() call.")
        .def("num_ticks", &MctsScheduler::num_ticks,
             "Tick count used by the last schedule() call.");

    m.def("evaluate_schedule", &py_evaluate_schedule,
          nb::arg("Hx"), nb::arg("Hz"), nb::arg("schedule"),
          nb::arg("num_ticks"), nb::arg("obs_x"), nb::arg("obs_z"),
          nb::arg("p"), nb::arg("nshots") = 4096,
          nb::arg("max_bp_iter") = 30, nb::arg("batch_size") = 4096,
          nb::arg("seed") = 42,
          "Joint circuit-level LER of a syndrome-extraction schedule "
          "(Pauli-frame Monte Carlo + GPU BP per block).");
}
