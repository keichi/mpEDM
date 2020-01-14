#ifndef __EMBEDDING_DIM_GPU_H__
#define __EMBEDDING_DIM_GPU_H__

#include <memory>

#include "embedding_dim.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"
#include "simplex_cpu.h"

class EmbeddingDimGPU : public EmbeddingDim
{
public:
    EmbeddingDimGPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose);

    uint32_t run(const Series &ts) override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    std::vector<LUT> luts;
    std::vector<float> rhos;
    std::vector<std::vector<float>> buffers;
    uint32_t n_devs;
};

#endif
