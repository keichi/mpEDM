#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__

#include "dataset.h"
#include "lut.h"

class NearestNeighbors
{
public:
    NearestNeighbors(int E_max, int tau, int k, bool verbose);
    virtual ~NearestNeighbors();

    virtual void run(const Dataset &ds);
    virtual void compute_lut(LUT &out, const Timeseries &ts, int E) = 0;

protected:
    // Maximum embedding dimension (number of columns)
    const int E_max;
    // Lag
    const int tau;
    // Number of neighbors to find
    const int top_k;
    // Enable verbose logging
    const bool verbose;
};

#endif
