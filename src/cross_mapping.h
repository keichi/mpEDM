#ifndef __CROSS_MAPPING_H__
#define __CROSS_MAPPING_H__

#include <cstdint>

#include "dataset.h"

class CrossMapping
{
public:
    CrossMapping(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : E_max(E_max), tau(tau), Tp(Tp), verbose(verbose)
    {
    }

    virtual void predict(const Timeseries &library,
                         const Timeseries &target) = 0;

protected:
    uint32_t E_max;
    uint32_t tau;
    uint32_t Tp;
    bool verbose;
};

#endif
