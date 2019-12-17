#ifndef __CROSS_MAPPING_H__
#define __CROSS_MAPPING_H__

#include <cstdint>

class CrossMapping
{
public:
    CrossMapping(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : E_max(E_max), tau(tau), Tp(Tp), verbose(verbose)
    {
    }

protected:
    uint32_t E_max;
    uint32_t tau;
    uint32_t Tp;
    bool verbose;
};

#endif
