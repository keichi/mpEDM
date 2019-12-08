#ifndef __NEAREST_NEIGHBORS_GPU_H__
#define __NEAREST_NEIGHBORS_GPU_H__

#include "dataset.h"
#include "nearest_neighbors.h"
#include "lut.h"

class NearestNeighborsGPU : public NearestNeighbors
{
public:
    NearestNeighborsGPU(int E_max, int tau, int k, bool verbose);

    void compute_lut(LUT &out, const Timeseries &ts, int E);

protected:
};

#endif
