#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__

#include "dataset.h"
#include "lut.h"

class NearestNeighbors
{
public:
    NearestNeighbors(int tau, bool verbose) : tau(tau), verbose(verbose) {}
    virtual ~NearestNeighbors(){};

    virtual void compute_lut(LUT &out, const Timeseries &library,
                             const Timeseries &predictee, int E)
    {
        compute_lut(out, library, predictee, E, E + 1);
    }

    virtual void compute_lut(LUT &out, const Timeseries &library,
                             const Timeseries &predictee, int E, int top_k) = 0;

protected:
    // Lag
    const int tau;
    // Enable verbose logging
    const bool verbose;
};

#endif
