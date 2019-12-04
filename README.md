# cuEDM [![](https://github.com/keichi/cuEDM/workflows/build/badge.svg)](https://github.com/keichi/cuEDM/actions?query=workflow%3Abuild)

GPU-accelerated implementation of Empirical Dynamic Modeling

## Requirements

- CMake 3.10+
- C++ compiler with C++11 support
- ArrayFire 3.6.2+ for the GPU backend
- Intel C++ Compiler is preferred for the CPU backend

## Installation

```
$ git clone --recursive git@github.com:keichi/cuEDM.git
$ cd cuEDM
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
