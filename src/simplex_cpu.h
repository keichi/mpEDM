#ifndef __SIMPLEX_CPU_H__
#define __SIMPLEX_CPU_H__

#include "data_frame.h"
#include "lut.h"
#include "simplex.h"

class SimplexCPU : public Simplex
{
public:
    SimplexCPU(uint32_t tau, uint32_t Tp, bool verbose)
        : Simplex(tau, Tp, verbose)
    {
    }
    ~SimplexCPU(){};

    Series predict(std::vector<float> &buffer, const LUT &lut,
                   const Series &target, uint32_t E) override;

protected:
};

#endif
