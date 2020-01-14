#ifndef __CROSS_MAPPING_H__
#define __CROSS_MAPPING_H__

#include <cstdint>

#include "dataframe.h"

class CrossMapping
{
public:
    CrossMapping(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : max_E(max_E), tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~CrossMapping() {}

    virtual void run(std::vector<float> &rhos, const DataFrame &df,
                     const std::vector<uint32_t> &optimal_E) = 0;

protected:
    uint32_t max_E;
    uint32_t tau;
    uint32_t Tp;
    bool verbose;
};

#endif
