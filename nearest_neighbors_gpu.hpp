#ifndef __KNN_KERNEL_GPU_HPP__
#define __KNN_KERNEL_GPU_HPP__

#include <limits>

#include <arrayfire.h>

#include "dataset.hpp"
#include "nearest_neighbors.hpp"
#include "lut.hpp"
#include "timer.hpp"

class NearestNeighborsGPU : public NearestNeighbors
{
public:
    NearestNeighborsGPU(int E_max, int tau, int k, bool verbose)
        : NearestNeighbors(E_max, tau, k, verbose)
    {
        af::setBackend(AF_BACKEND_CUDA);
        af::info();
    }

    void compute_lut(LUT &out, const Timeseries &ts, int E)
    {
        auto n = ts.size() - (E - 1) * tau;

        out.resize(ts.size(), top_k);

        af::array idx;
        af::array dist;

        // We actually only need n rows, but we allocate n_rows rows. Fixed
        // array size allows ArrayFire to recycle previously allocated buffers
        // and greatly reduces memory allocations on the GPU.
        std::vector<float> block_host(E * ts.size());

        // Perform embedding
        for (auto i = 0; i < E; i++) {

            // Populate the first n with input data
            for (auto j = 0; j < n; j++) {
                auto p = ts.data();
                block_host[i * ts.size() + j] = p[i * tau + j];
            }

            // Populate the rest with dummy data
            // We put infinity so that they are be ignored in the sorting
            for (auto j = n; j < ts.size(); j++) {
                block_host[i * ts.size() + j] =
                    std::numeric_limits<float>::infinity();
            }
        }

        af::array block(ts.size(), E, block_host.data());

        af::nearestNeighbour(idx, dist, block, block, 1, top_k, AF_SSD);
        dist = af::sqrt(dist);

        dist.host(out.distances.data());
        idx.host(out.indices.data());

        out.resize(n, top_k);
    }

protected:
};

#endif
