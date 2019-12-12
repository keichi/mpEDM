#ifndef __SIMPLEX_CPU_H__
#define __SIMPLEX_CPU_H__

#include "dataset.h"
#include "nearest_neighbors_cpu.h"
#include "simplex.h"

class SimplexCPU : public Simplex
{
public:
    SimplexCPU(int tau, int Tp, bool verbose)
        : Simplex(tau, Tp, verbose), knn(new NearestNeighborsCPU(tau, verbose))
    {
    }
    ~SimplexCPU(){};

    float predict(const Timeseries &library, const Timeseries &prediction,
                  int E);

protected:
    std::unique_ptr<NearestNeighbors> knn;

    // Compute Pearson correlation coefficient between two Timeseries
    float corrcoef(const Timeseries &ts1, const Timeseries &ts2);
};

#endif
