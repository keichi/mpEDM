#ifndef __NEAREST_NEIGHBORS_CPU_H__
#define __NEAREST_NEIGHBORS_CPU_H__

#include "lut.h"
#include "nearest_neighbors.h"

class NearestNeighborsCPU : public NearestNeighbors
{
public:
    NearestNeighborsCPU(uint32_t tau, bool verbose);

    void compute_lut(LUT &out, const Timeseries &library,
                     const Timeseries &target, uint32_t E, uint32_t top_k);

protected:
    LUT cache;
};

#endif
