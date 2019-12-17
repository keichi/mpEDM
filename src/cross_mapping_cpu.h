#ifndef __CROSS_MAPPING_CPU_H__
#define __CROSS_MAPPING_CPU_H__

#include <memory>

#include <omp.h>

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"

class CrossMappingCPU : public CrossMapping
{
public:
    CrossMappingCPU(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : CrossMapping(E_max, tau, Tp, verbose),
          knn(new NearestNeighborsCPU(tau, verbose)), luts(E_max)
    {
        for (auto i = 0; i < omp_get_max_threads(); i++)
        {
            simplex.push_back(SimplexCPU(tau, Tp, verbose));
        }
    }

    void predict(std::vector<float> &rhos, const Timeseries &library,
                 const std::vector<Timeseries> &targets,
                 const std::vector<uint32_t> &optimal_E) override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::vector<SimplexCPU> simplex;
    std::vector<LUT> luts;
};

#endif
