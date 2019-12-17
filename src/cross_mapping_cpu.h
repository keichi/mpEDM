#ifndef __CROSS_MAPPING_CPU_H__
#define __CROSS_MAPPING_CPU_H__

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"

class CrossMappingCPU : public CrossMapping
{
public:
    CrossMappingCPU(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : CrossMapping(E_max, tau, Tp, verbose),
          knn(new NearestNeighborsCPU(tau, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), luts(E_max)
    {
    }

    void predict(const Timeseries &library, const Timeseries &target);

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;
};

#endif
