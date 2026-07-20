#include "scheduler.hpp"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace asy {

MctsTree::MctsTree(const CssProblem& prob, int nshots, int max_bp_iter,
                   uint64_t seed)
    : _prob(prob), _nrows(prob.ncx + prob.ncz), _nt(prob.num_ticks),
      _nshots(nshots), _max_bp_iter(max_bp_iter), _seed(seed), _rng(seed)
{
    int max_deg = 0;
    for (auto& s : _prob.supp) max_deg = std::max(max_deg, (int)s.size());
    if (_nt < max_deg)
        throw std::runtime_error("MctsTree: num_ticks < max check weight — "
                                 "schedule cannot place every CNOT");
    // Root at the natural-order schedule (circuit.py's no-schedule baseline):
    // the search explores around it, and best-so-far can never end up worse
    // than the baseline under the search seed.
    _root = std::make_unique<MctsNode>();
    _root->schedule = natural_schedule();
    _root->parent   = nullptr;
}

Schedule MctsTree::natural_schedule() const {
    Schedule nat((size_t)_nrows * _nt, -1);
    for (int r = 0; r < _nrows; ++r) {
        std::vector<int> qs = _prob.supp[r];
        std::sort(qs.begin(), qs.end());
        for (int t = 0; t < (int)qs.size(); ++t)
            nat[(size_t)r * _nt + t] = qs[t];
    }
    return nat;
}

Schedule MctsTree::run(int iters) {
    for (int i = 0; i < iters; ++i) {
        MctsNode* node = select(_root.get());
        if (node->visits > 0)
            node = expand(node);
        float score = rollout(node);
        backprop(node, score);
        if (score < _best_score) {
            _best_score = score;
            _best_sched = node->schedule;
        }
    }
    return _best_sched;
}

MctsNode* MctsTree::select(MctsNode* node) {
    while (!node->children.empty()) {
        MctsNode* best = nullptr;
        double best_uct = -1e18;
        for (auto& child : node->children) {
            double u = child->uct();
            if (u > best_uct) { best_uct = u; best = child.get(); }
        }
        node = best;
    }
    return node;
}

MctsNode* MctsTree::expand(MctsNode* node) {
    auto neighbours = neighbour_schedules(node->schedule);
    for (auto& ns : neighbours) {
        auto child = std::make_unique<MctsNode>();
        child->schedule = ns;
        child->parent   = node;
        node->children.push_back(std::move(child));
    }
    if (node->children.empty()) return node;
    std::uniform_int_distribution<int> d(0, (int)node->children.size() - 1);
    return node->children[d(_rng)].get();
}

float MctsTree::rollout(MctsNode* node) {
    // GPU evaluation; fixed seed ⇒ common random numbers across candidates
    return evaluate_css_schedule(_prob, node->schedule, _nshots, _max_bp_iter,
                                 /*batch_size=*/4096, _seed);
}

void MctsTree::backprop(MctsNode* node, float score) {
    float val = 1.0f - score;     // lower LER = higher value
    while (node) {
        node->visits++;
        node->value += val;
        node = node->parent;
    }
}

// Mutations preserving the every-edge-exactly-once invariant:
//   A: swap the ticks of two of a row's assigned edges
//   B: move one assigned edge to a currently idle tick of its row
std::vector<Schedule> MctsTree::neighbour_schedules(const Schedule& s) {
    std::vector<Schedule> result;
    result.reserve(4);
    std::uniform_int_distribution<int> dr(0, _nrows - 1);

    for (int k = 0; k < 4; ++k) {
        int r = dr(_rng);
        std::vector<int> assigned, idle;
        for (int t = 0; t < _nt; ++t)
            (s[(size_t)r * _nt + t] >= 0 ? assigned : idle).push_back(t);
        if (assigned.size() < 2 && idle.empty()) continue;

        Schedule ns = s;
        bool do_swap = assigned.size() >= 2 &&
                       (idle.empty() || (_rng() & 1));
        if (do_swap) {
            std::shuffle(assigned.begin(), assigned.end(), _rng);
            std::swap(ns[(size_t)r * _nt + assigned[0]],
                      ns[(size_t)r * _nt + assigned[1]]);
        } else {
            std::uniform_int_distribution<int> da(0, (int)assigned.size() - 1);
            std::uniform_int_distribution<int> di(0, (int)idle.size() - 1);
            int from = assigned[da(_rng)], to = idle[di(_rng)];
            std::swap(ns[(size_t)r * _nt + from], ns[(size_t)r * _nt + to]);
        }
        result.push_back(std::move(ns));
    }
    return result;
}

std::string MctsTree::best_schedule_json() const {
    std::ostringstream ss;
    ss << "[\n";
    for (int r = 0; r < _nrows; ++r) {
        ss << "  [";
        for (int t = 0; t < _nt; ++t) {
            ss << _best_sched[(size_t)r * _nt + t];
            if (t + 1 < _nt) ss << ", ";
        }
        ss << "]";
        if (r + 1 < _nrows) ss << ",";
        ss << "\n";
    }
    ss << "]\n";
    return ss.str();
}

} // namespace asy
