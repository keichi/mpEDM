# cuEDM [![](https://github.com/keichi/cuEDM/workflows/build/badge.svg)](https://github.com/keichi/cuEDM/actions?query=workflow%3Abuild)

GPU-accelerated implementation of Empirical Dynamic Modeling

## Requirements

- CMake 3.10+
- C++ compiler that supports C++11 (Intel C++ Compiler is strongly recommended
  for building the CPU backend)
- ArrayFire 3.6.2+ for building the GPU backend

## Installation

1. Install dependencies:
    - Install [ArrayFire](https://arrayfire.com/) if building the GPU backend.
    - Install [HDF5](https://www.hdfgroup.org/) if building the HDF5 reader.

2. Clone the source code:
    ```
    $ git clone --recursive git@github.com:keichi/cuEDM.git
    ```

3. Run cmake:
    ```
    $ cd cuEDM
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    ```
    - Add `-DArrayFire_DIR=/path/to/arrayfire` option to enable the GPU backend.
    - Add `-DHDF5_DIR=/path/to/hdf5` option to enable the HDF5 reader.

4. Run make:
    ```
    $ make
    ```
