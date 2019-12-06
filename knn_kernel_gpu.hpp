#ifndef __KNN_KERNEL_GPU_HPP__
#define __KNN_KERNEL_GPU_HPP__

#include <limits>

#include <arrayfire.h>

#include "dataset.hpp"
#include "knn_kernel.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernelGPU : public KNNKernel
{
public:
    KNNKernelGPU(int E_max, int tau, int k, bool verbose) : KNNKernel(E_max, tau, k, verbose) {}

    void compute_lut(LUT &out, const float *const col, int E, int n)
    {
        out.resize(n_rows, top_k);

        af::array idx;
        af::array dist;

        std::vector<float> block_host(E * n_rows);

        // Perform embedding
        for (int i = 0; i < E; i++) {
            for (int j = 0; j < n; j++) {
                block_host[i * n_rows + j] = col[i * tau + j];
            }
            for (int j = n; j < n_rows; j++) {
                block_host[i * n_rows + j] = std::numeric_limits<float>::infinity();
            }
        }

        af::array block(n_rows, E, block_host.data());

        af::nearestNeighbour(idx, dist, block, block, 1, top_k, AF_SSD);
        dist = af::sqrt(dist);

        dist.host(out.distances.data());
        idx.host(out.indices.data());
    }

protected:
};

#endif
