#ifndef __EMBEDDING_DIM_GPU_H__
#define __EMBEDDING_DIM_GPU_H__

#include <arrayfire.h>
#include <memory>

#include "embedding_dim.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"
#include "simplex_cpu.h"

class EmbeddingDimGPU : public EmbeddingDim
{
public:
    EmbeddingDimGPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : EmbeddingDim(max_E, tau, Tp, verbose),
          knn(new NearestNeighborsGPU(tau, Tp, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), rhos(max_E)
    {
        n_devs = af::getDeviceCount();

        luts.resize(n_devs);
        buffers.resize(n_devs);
    }

    uint32_t run(const Timeseries &ts) override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;
    std::vector<float> rhos;
    std::vector<std::vector<float>> buffers;
    uint32_t n_devs;
};

#endif
