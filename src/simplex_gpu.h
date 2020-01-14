#ifndef __SIMPLEX_GPU_H__
#define __SIMPLEX_GPU_H__

#include "dataframe.h"
#include "lut.h"
#include "simplex.h"

class SimplexGPU : public Simplex
{
public:
    SimplexGPU(uint32_t tau, uint32_t Tp, bool verbose)
        : Simplex(tau, Tp, verbose)
    {
    }
    ~SimplexGPU(){};

    void predict(Series &prediction, std::vector<float> &buffer, const LUT &lut,
                 const Series &target, uint32_t E) override;

protected:
};

#endif
