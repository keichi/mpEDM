#ifndef __NEAREST_NEIGHBORS_CPU_H__
#define __NEAREST_NEIGHBORS_CPU_H__

#include "nearest_neighbors.h"
#include "lut.h"

class NearestNeighborsCPU : public NearestNeighbors
{
public:
    NearestNeighborsCPU(int E_max, int tau, int k, bool verbose);

    void compute_lut(LUT &out, const Timeseries &ts, int E);

protected:
    LUT cache;
};

#endif
