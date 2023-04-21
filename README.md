# mpEDM [![](https://github.com/keichi/mpEDM/workflows/build/badge.svg)](https://github.com/keichi/mpEDM/actions?query=workflow%3Abuild)

This project has been superseded by [kEDM](https://github.com/keichi/kEDM) and
will no longer be maintained. kEDM has generally higher performance than mpEDM,
and provides Python bindings for ease of use.

----

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

## Literature

For a detailed description of the algorithm and performance measurements using
up to 2,044 GPUs, please refer to the following paper:

Wassapon Watanakeesuntorn, Keichi Takahashi, Kohei Ichikawa, Joseph Park,
George Sugihara, Ryousei Takano, Jason Haga, Gerald M. Pao, "Massively
Parallel Causal Inference of Whole Brain Dynamics at Single Neuron Resolution",
_26th International Conference on Parallel and Distributed Systems (ICPADS
2020)_, Dec. 2020. [10.1109/ICPADS51040.2020.00035](https://doi.org/10.1109/ICPADS51040.2020.00035)
