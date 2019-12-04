#ifndef __KNN_KERNEL_GPU_HPP__
#define __KNN_KERNEL_GPU_HPP__

#include <arrayfire.h>

#include "dataset.hpp"
#include "knn_kernel.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernelGPU : public KNNKernel
{
public:
    KNNKernelGPU(int E_max, int tau, int k) : KNNKernel(E_max, tau, k) {}

    void compute_lut(LUT &out, const float *const col, int E, int n)
    {
        out.resize(n, top_k);

        af::array idx;
        af::array dist;

        std::vector<float> block_host(E * n);

        // Perform embedding
        for (int i = 0; i < E; i++) {
            for (int j = 0; j < n; j++) {
                block_host[i * n + j] = col[i * tau + j];
            }
        }

        af::array block(n, E, block_host.data());

        af::nearestNeighbour(idx, dist, block, block, 1, top_k, AF_SSD);
        dist = af::sqrt(dist);

        dist.host(out.distances.data());
        idx.host(out.indices.data());
    }

protected:
};

#endif
