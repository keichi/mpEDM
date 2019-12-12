#ifndef __NEAREST_NEIGHBORS_GPU_H__
#define __NEAREST_NEIGHBORS_GPU_H__

#include "dataset.h"
#include "lut.h"
#include "nearest_neighbors.h"

class NearestNeighborsGPU : public NearestNeighbors
{
public:
    NearestNeighborsGPU(int tau, int k, bool verbose);

    void compute_lut(LUT &out, const Timeseries &library,
                     const Timeseries &predictee, int E);

protected:
};

#endif
