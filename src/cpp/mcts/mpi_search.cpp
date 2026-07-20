#include "scheduler.hpp"
#include <cstring>
#include <stdexcept>

namespace asy {

MpiSearchResult mpi_mcts_search(
    MctsTree& local_tree,
    int steps,
    int iters_per_step,
    MPI_Comm comm)
{
    int rank = 0, world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    Schedule best_local;
    float best_local_ler = 1.0f;

    // Encode (ler, rank) as double for MPI_Minloc
    struct FloatIntPair { float val; int rank; };

    for (int step = 0; step < steps; ++step) {
        Schedule sched = local_tree.run(iters_per_step);
        float ler = local_tree.best_score();
        if (ler < best_local_ler) {
            best_local_ler = ler;
            best_local = sched;
        }

        // Share the globally best LER across ranks
        FloatIntPair local_pair{best_local_ler, rank};
        FloatIntPair global_pair{1.0f, 0};
        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_FLOAT_INT,
                      MPI_MINLOC, comm);

        // Broadcast the winning schedule from its rank
        int sched_size = (int)sched.size();
        MPI_Bcast(&sched_size, 1, MPI_INT, global_pair.rank, comm);

        if (global_pair.rank == rank) {
            // This rank broadcasts its schedule
            MPI_Bcast(best_local.data(), sched_size, MPI_INT, rank, comm);
        } else {
            // Others receive the global best and update if better
            Schedule global_sched(sched_size);
            MPI_Bcast(global_sched.data(), sched_size, MPI_INT,
                      global_pair.rank, comm);
            if (global_pair.val < best_local_ler) {
                best_local_ler = global_pair.val;
                best_local = global_sched;
            }
        }
    }

    // Gather best across all ranks (final result)
    FloatIntPair local_pair{best_local_ler, rank};
    FloatIntPair global_pair{1.0f, 0};
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_FLOAT_INT,
                  MPI_MINLOC, comm);

    int sched_size = (int)best_local.size();
    MPI_Bcast(&sched_size, 1, MPI_INT, global_pair.rank, comm);

    Schedule final_sched(sched_size);
    if (global_pair.rank == rank) {
        final_sched = best_local;
    }
    MPI_Bcast(final_sched.data(), sched_size, MPI_INT, global_pair.rank, comm);

    return MpiSearchResult{final_sched, global_pair.val, global_pair.rank};
}

} // namespace asy
