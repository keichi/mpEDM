# mpEDM [![](https://github.com/keichi/mpEDM/workflows/build/badge.svg)](https://github.com/keichi/mpEDM/actions?query=workflow%3Abuild)

GPU-accelerated implementation of Empirical Dynamic Modeling

## Requirements

- CMake 3.10+
- C++ compiler that supports C++11 (Intel C++ Compiler is strongly recommended
  for building the CPU backend)
- HDF5 (parallel build if building the multi-node benchmarks)
- (optional) MPI if building the multi-node benchmarks
- (optional) ArrayFire 3.6.2+ if building the GPU backend

## Installation

1. Install dependencies:
    - Install [HDF5](https://www.hdfgroup.org/).
    - Install [ArrayFire](https://arrayfire.com/) if building the GPU backend.

2. Clone the source code:
    ```
    $ git clone --recursive https://github.com/keichi/mpEDM.git
    ```

3. Run cmake:
    ```
    $ cd mpEDM
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    ```
    - `-DCMAKE_BUILD_TYPE` can be either `Release`, `RelWithDebInfo`, `Debug`
      or `MinSizeRel`.
    - Add `-DCMAKE_CXX_COMPILER=/path/to/c++` to select the C++ compiler to use.
    - Add `-DCMAKE_CXX_FLAGS="..."` to customize the C++ compiler flags.
    - Add `-DHDF5_DIR=/path/to/hdf5` if HDF5 is not installed in a standard
      path.
    - Add `-DArrayFire_DIR=/path/to/arrayfire` if ArrayFire is not installed
      in a standard path.

4. Run make:
    ```
    $ make
    ```
