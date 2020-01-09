#ifndef __CROSS_MAPPING_GPU_H__
#define __CROSS_MAPPING_GPU_H__

#include <arrayfire.h>
#include <memory>

#include <arrayfire.h>

#include "cross_mapping.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"
#include "simplex_cpu.h"

class CrossMappingGPU : public CrossMapping
{
public:
    CrossMappingGPU(uint32_t E_max, uint32_t tau, uint32_t Tp, bool verbose)
        : CrossMapping(E_max, tau, Tp, verbose),
          knn(new NearestNeighborsGPU(tau, Tp, verbose)), luts(E_max)
    {
        n_devs = af::getDeviceCount();
    }

    void run(std::vector<float> &rhos, const Dataset &ds,
             const std::vector<uint32_t> &optimal_E) override;

    void predict(std::vector<float> &rhos, const Dataset &ds, af::array data,
                 const uint32_t index, const std::vector<uint32_t> &optimal_E);

    void simplex(af::array &prediction, const LUT &lut, const af::array &target,
                 uint32_t E);

    void shift_target(af::array &shifted_target, const af::array &target,
                      uint32_t E);

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::vector<LUT> luts;
    uint32_t n_devs;
};

#endif
