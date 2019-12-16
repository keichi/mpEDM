#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__

#include "dataset.h"
#include "lut.h"

class NearestNeighbors
{
public:
    NearestNeighbors(uint32_t tau, bool verbose) : tau(tau), verbose(verbose) {}
    virtual ~NearestNeighbors(){};

    virtual void compute_lut(LUT &out, const Timeseries &library,
                             const Timeseries &target, uint32_t E)
    {
        compute_lut(out, library, target, E, E + 1);
    }

    virtual void compute_lut(LUT &out, const Timeseries &library,
                             const Timeseries &target, uint32_t E,
                             uint32_t top_k) = 0;

protected:
    // Lag
    const uint32_t tau;
    // Enable verbose logging
    const bool verbose;
};

#endif
