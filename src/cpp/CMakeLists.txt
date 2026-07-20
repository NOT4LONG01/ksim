cmake_minimum_required(VERSION 3.22)
project(ksim_extension LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# nanobind injects -Os/-ffunction-sections/-fdata-sections, which nvcc_wrapper
# rejects on CUDA builds; strip them globally and per-target.
string(REPLACE " -Os" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -ffunction-sections" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -fdata-sections" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_compile_options(-diag-suppress=1675)

# kokkos_sim — ZX stabilizer-rank runtime (CUDA on GPU builds, OpenMP on CPU).
nanobind_add_module(kokkos_sim NB_STATIC kokkos_sim.cpp)
get_target_property(_nb_opts kokkos_sim COMPILE_OPTIONS)
if(_nb_opts)
    list(REMOVE_ITEM _nb_opts -Os)
    set_target_properties(kokkos_sim PROPERTIES COMPILE_OPTIONS "${_nb_opts}")
endif()
target_link_libraries(kokkos_sim PRIVATE Kokkos::kokkos)
install(TARGETS kokkos_sim LIBRARY DESTINATION .)
