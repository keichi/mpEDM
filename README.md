# cuEDM

```
$ git clone --recursive git@github.com:keichi/cuEDM.git
$ cd cuEDM

$ module load compiler/gcc/7
$ module load compiler/intel/2018/1.038
$ icc -Wall -O3 -std=c++11 -march=native -fopenmp -o cuEDM main.cpp
```
