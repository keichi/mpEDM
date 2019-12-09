#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__

#include "dataset.h"
#include "lut.h"

class NearestNeighbors
{
public:
    NearestNeighbors(int tau, int k, bool verbose)
        : tau(tau), top_k(k), verbose(verbose)
    {
    }
    virtual ~NearestNeighbors(){};

    virtual void compute_lut(LUT &out, const Timeseries &library,
                             const Timeseries &predictee, int E) = 0;

protected:
    // Lag
    const int tau;
    // Number of neighbors to find
    const int top_k;
    // Enable verbose logging
    const bool verbose;
};

#endif
