#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#include "data_frame.h"
#include "lut.h"

class Simplex
{
public:
    Simplex(uint32_t tau, uint32_t Tp, bool verbose)
        : tau(tau), Tp(Tp), verbose(verbose)
    {
    }
    virtual ~Simplex(){};

    // Predict timeseries using Simplex projection. `prediction` is the
    // predicted rsult. The actual values are stored into `buffer`. `lut`
    // needs to be pre-computed using NearestNeighbors and normalized.
    virtual Series predict(std::vector<float> &buffer, const LUT &lut,
                           const Series &target, uint32_t E) = 0;

    // Shift and trim the target timeseries so that its time index matches the
    // predicted timeseries.
    virtual Series shift_target(const Series &target, uint32_t E)
    {
        const auto shift = (E - 1) * tau + Tp;

        return target.slice(shift);
    }

protected:
    // Lag
    const uint32_t tau;
    // How many steps to predict in future
    const uint32_t Tp;
    // Enable verbose logging
    const bool verbose;
};

#endif
