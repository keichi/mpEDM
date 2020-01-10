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
    CrossMappingGPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose);

    void run(std::vector<float> &rhos, const Dataset &ds,
             const std::vector<uint32_t> &optimal_E) override;

    void predict(std::vector<float> &rhos, const Timeseries &library,
                 const std::vector<Timeseries> &targets,
                 const std::vector<uint32_t> &optimal_E);

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;
    uint32_t n_devs;
};

#endif
