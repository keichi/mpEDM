#ifndef __SIMPLEX_CPU_H__
#define __SIMPLEX_CPU_H__

#include "dataset.h"
#include "nearest_neighbors_cpu.h"
#include "simplex.h"

class SimplexCPU : Simplex
{
public:
    SimplexCPU(int tau, int k, int Tp, bool verbose)
        : Simplex(tau, k, Tp, verbose), knn(tau, k, verbose)
    {
    }
    ~SimplexCPU(){};

    float predict(const Timeseries &library, const Timeseries &prediction,
                  int E);

protected:
    NearestNeighborsCPU knn;

    // Compute Pearson correlation coefficient between two Timeseries
    float corrcoef(const Timeseries &ts1, const Timeseries &ts2);
};

#endif
