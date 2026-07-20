# Building the Kokkos extensions

`ksim.compile` and the numpy reference `ksim.sample_flat` are pure Python and
need no build at all. The three extensions — `kokkos_sim` (GPU sampling),
`kokkos_decoder` (GPU LDPC decoding), `kokkos_mcts` (MCTS schedule search) —
are one CMake build over `src/cpp/`, against a Kokkos install. All three
`.so` install as top-level modules into `src/python/`, so an editable install
of this repo puts them on the importer's path.

**Always build all three together** (`cmake --build build` with no `--target`).
Each links Kokkos statically, so mixing `.so` files built against different
Kokkos installs inside one process corrupts the shared runtime state — see the
rebuild note in [`architecture.md`](architecture.md#cpp).

## CPU-only (OpenMP + Serial, plain g++)

What a workstation or a CPU cluster gets. Kokkos 4.1.00 installs to `lib64/`,
so `Kokkos_DIR` points there:

```bash
module load cmake intel/mpi     # MPI is what kokkos_mcts configures against
NANOBIND_DIR=$(python -c "import nanobind; print(nanobind.__file__.replace('/__init__.py',''))")
cmake -S src/cpp -B build \
  -DKokkos_DIR=$HOME/kokkos/install/lib64/cmake/Kokkos \
  -DCMAKE_PREFIX_PATH=$NANOBIND_DIR \
  -DCMAKE_CXX_COMPILER=g++ \
  -DPython_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/src/python"
cmake --build build -j$(nproc)
cmake --install build          # copies kokkos_{sim,decoder,mcts}.*.so into src/python/
pip install -e .               # re-pick up the new .so files
```

## CUDA (GPU cluster)

Reference hardware: **NVIDIA A100-SXM4-80GB (SM80/Ampere)**, Kokkos built with
`CUDA=ON, ARCH_AMPERE80=ON, OPENMP=ON`. Module names are site-specific:

```bash
module load cuda/12 openmpi cmake gcc
NANOBIND_DIR=$(python -c "import nanobind; print(nanobind.__file__.replace('/__init__.py',''))")
cmake -S src/cpp -B build \
  -DKokkos_DIR=/opt/kokkos/install/lib/cmake/Kokkos \
  -DCMAKE_PREFIX_PATH=$NANOBIND_DIR \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/nvcc_wrapper_shim \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/src/python"
cmake --build build -j$(nproc)
cmake --install build
pip install -e .
```

**`nvcc_wrapper_shim`** (`/usr/local/bin/nvcc_wrapper_shim`) is not in the repo
and has vanished from containers before. If the build fails with make
`Error 127` or `nvcc fatal : 's': expected a number`, recreate it as a bash
wrapper that rewrites host-only flags (`-Os`, `-ffunction-sections`,
`-fdata-sections`, `-fno-stack-protector`, `-fvisibility=*`) into
`-Xcompiler <flag>` form and execs `/usr/local/bin/nvcc_wrapper`.

**The whole Kokkos install can vanish, not just the shim** (seen twice on one
container). If `find_package(Kokkos REQUIRED)` fails because `Kokkos_DIR`
doesn't exist anywhere, reinstall from source:

```bash
git clone --branch 5.1.1 https://github.com/kokkos/kokkos.git
cd kokkos
# Kokkos 4.5.01 does not compile against CUDA 13's changed driver API
# (cudaGraphAddDependencies, cudaMemAdvise/cudaMemPrefetchAsync signature
# changes) -- use 5.1.1+ if CUDA >= 13. nvcc_wrapper's default_arch is "sm_70"
# pre-5.1 and CUDA 13 dropped Volta (`nvcc fatal: Unsupported gpu architecture
# 'sm_70'`) -- patch it:
sed -i 's/^default_arch="sm_70"$/default_arch="sm_80"/' bin/nvcc_wrapper
cmake -S . -B build -DCMAKE_CXX_COMPILER=$(pwd)/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=/opt/kokkos/install \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build -j$(nproc) && cmake --install build
# Then recreate nvcc_wrapper_shim execing /opt/kokkos/install/bin/nvcc_wrapper.
```

Also needs `python3.12-dev` (or the matching version — CMake's `FindPython`
needs `Development.Module`) and OpenMPI (`libopenmpi-dev`/`openmpi-bin`, for
`kokkos_mcts`'s `mpi_search.cpp`) if missing.

## Smoke test

```bash
python -c "import kokkos_sim, kokkos_decoder, kokkos_mcts; print('extensions OK')"
OMP_PROC_BIND=false pytest -q
python example/sample_demo.py     # compile → sample → det/obs, GPU or numpy reference
```
