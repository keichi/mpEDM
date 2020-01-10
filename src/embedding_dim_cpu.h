#ifndef __EMBEDDING_DIM_CPU_H__
#define __EMBEDDING_DIM_CPU_H__

#include <memory>

#include "embedding_dim.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"

class EmbeddingDimCPU : public EmbeddingDim
{
public:
    EmbeddingDimCPU(uint32_t max_E, uint32_t tau, uint32_t Tp, bool verbose)
        : EmbeddingDim(max_E, tau, Tp, verbose),
          knn(new NearestNeighborsCPU(tau, Tp, verbose)),
          simplex(new SimplexCPU(tau, Tp, verbose)), rhos(max_E)
    {
    }

    uint32_t run(const Timeseries &ts) override;

protected:
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;
    LUT lut;
    std::vector<float> rhos;
    std::vector<float> buffer;
};

#endif
