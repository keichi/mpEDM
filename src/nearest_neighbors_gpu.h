#ifndef __NEAREST_NEIGHBORS_GPU_H__
#define __NEAREST_NEIGHBORS_GPU_H__

#include "data_frame.h"
#include "lut.h"
#include "nearest_neighbors.h"

class NearestNeighborsGPU : public NearestNeighbors
{
public:
    NearestNeighborsGPU(uint32_t tau, uint32_t Tp, bool verbose);

    void compute_lut(LUT &out, const Series &library, const Series &target,
                     uint32_t E, uint32_t top_k) override;

protected:
};

#endif
