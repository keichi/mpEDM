#ifndef __SIMPLEX_CPU_H__
#define __SIMPLEX_CPU_H__

#include "dataset.h"
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

    void predict(Timeseries &prediction, const LUT &lut,
                 const Timeseries &target, uint32_t E);

protected:
    // Predicted result
    std::vector<float> _prediction;
};

#endif
