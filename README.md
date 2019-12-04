# cuEDM [![](https://github.com/keichi/cuEDM/workflows/build/badge.svg)](https://github.com/keichi/cuEDM/actions?query=workflow%3Abuild)

CPU: Intel C compiler achieves highest performance.
GPU: ArrayFire 3.6.2 or higher is required for building the GPU backend.

```
$ git clone --recursive git@github.com:keichi/cuEDM.git
$ cd cuEDM
$ mkdir build
$ cd build

$ module load compiler/intel/2018/1.038
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
