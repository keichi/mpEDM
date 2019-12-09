#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#include "dataset.h"

class Simplex
{
public:
    Simplex(int tau, int k, int Tp, bool verbose)
        : tau(tau), top_k(k), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~Simplex(){};

    virtual float predict(const Timeseries &library,
                          const Timeseries &predictee, int E) = 0;

protected:
    // Lag
    const int tau;
    // Number of neighbors to find
    const int top_k;
    // How many steps to predict in future
    const int Tp;
    // Enable verbose logging
    const bool verbose;
};

#endif
