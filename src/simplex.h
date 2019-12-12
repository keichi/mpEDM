#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#include "dataset.h"
#include "lut.h"

class Simplex
{
public:
    Simplex(int tau, int Tp, bool verbose) : tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~Simplex(){};

    virtual float predict(const LUT &lut, const Timeseries &library,
                          const Timeseries &predictee, int E) = 0;

protected:
    // Lag
    const int tau;
    // How many steps to predict in future
    const int Tp;
    // Enable verbose logging
    const bool verbose;
    // Minimum weight
    const float min_weight = 1e-6f;
};

#endif
