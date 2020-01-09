#ifndef __CROSS_MAPPING_GPU_H__
#define __CROSS_MAPPING_GPU_H__

#include <memory>

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"
#include "simplex_cpu.h"

class CrossMappingGPU : public CrossMapping
{
public:
    CrossMappingGPU(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : CrossMapping(E_max, tau, Tp, verbose),
          knn(new NearestNeighborsGPU(tau, Tp, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), luts(E_max)
    {
    }

    void run(std::vector<float> &rhos, const Dataset &ds,
             const std::vector<uint32_t> &optimal_E) override;

    void predict(std::vector<float> &rhos, const Timeseries &library,
                 const std::vector<Timeseries> &targets,
                 const std::vector<uint32_t> &optimal_E);

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;
};

#endif
