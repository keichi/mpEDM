#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__

#include <cstdint>

#include "dataframe.h"
#include "lut.h"

class NearestNeighbors
{
public:
    NearestNeighbors(uint32_t tau, uint32_t Tp, bool verbose)
        : tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~NearestNeighbors(){};

    virtual void compute_lut(LUT &out, const Series &library,
                             const Series &target, uint32_t E)
    {
        compute_lut(out, library, target, E, E + 1);
    }

    virtual void compute_lut(LUT &out, const Series &library,
                             const Series &target, uint32_t E,
                             uint32_t top_k) = 0;

protected:
    // Lag
    const uint32_t tau;
    // Steps to predict in future
    const uint32_t Tp;
    // Enable verbose logging
    const bool verbose;
};

#endif
