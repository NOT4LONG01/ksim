cmake_minimum_required(VERSION 3.22)
project(kokkos_extensions LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# One CMakeLists builds all three Kokkos extensions; shared discovery + the
# nvcc_wrapper flag fixups live here once instead of duplicated per module.
find_package(Kokkos REQUIRED)
find_package(MPI    REQUIRED COMPONENTS CXX)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# nanobind injects -Os/-ffunction-sections/-fdata-sections, which nvcc_wrapper
# rejects; strip them globally, and per-target (below) from nanobind's options.
string(REPLACE " -Os" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -ffunction-sections" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -fdata-sections" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_compile_options(-diag-suppress=1675)

# Strip -Os from a target's nanobind COMPILE_OPTIONS, link Kokkos, install the .so.
function(finalize_kokkos_module tgt)
    get_target_property(_nb_opts ${tgt} COMPILE_OPTIONS)
    if(_nb_opts)
        list(REMOVE_ITEM _nb_opts -Os)
        set_target_properties(${tgt} PROPERTIES COMPILE_OPTIONS "${_nb_opts}")
    endif()
    target_link_libraries(${tgt} PRIVATE Kokkos::kokkos)
    install(TARGETS ${tgt} LIBRARY DESTINATION .)
endfunction()

# ── kokkos_decoder — BP+OSD / Relay-BP / BP+LSD GPU decoders ──────────────────
nanobind_add_module(kokkos_decoder
    decoder/kokkos_decoder.cpp
    decoder/bp_infra.cpp
    decoder/bp_osd.cpp
    decoder/relay_bp.cpp
    decoder/bp_lsd.cpp)
target_include_directories(kokkos_decoder PRIVATE decoder)
finalize_kokkos_module(kokkos_decoder)

# ── kokkos_sim — ZX stabilizer-rank runtime (the main sampler) ────────────────
nanobind_add_module(kokkos_sim NB_STATIC ksim/kokkos_sim.cpp)
finalize_kokkos_module(kokkos_sim)

# ── kokkos_mcts — MCTS schedule optimizer (reuses decoder BP; needs MPI) ──────
nanobind_add_module(kokkos_mcts
    mcts/kokkos_mcts.cpp
    mcts/mcts.cpp
    mcts/schedule_eval.cpp
    mcts/mpi_search.cpp
    decoder/bp_infra.cpp
    decoder/bp_osd.cpp)
target_include_directories(kokkos_mcts PRIVATE mcts decoder)
finalize_kokkos_module(kokkos_mcts)
target_link_libraries(kokkos_mcts PRIVATE MPI::MPI_CXX)
