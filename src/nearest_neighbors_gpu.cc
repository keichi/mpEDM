#include <limits>

#include <arrayfire.h>

#include "nearest_neighbors_gpu.h"

NearestNeighborsGPU::NearestNeighborsGPU(int tau, bool verbose)
    : NearestNeighbors(tau, verbose)
{
    af::info();
}

void NearestNeighborsGPU::compute_lut(LUT &out, const Timeseries &library,
                                      const Timeseries &predictee, int E,
                                      int top_k)
{
    const auto n_library = library.size() - (E - 1) * tau;
    const auto n_predictee = predictee.size() - (E - 1) * tau;
    const auto p_library = library.data();
    const auto p_predictee = predictee.data();

    out.resize(predictee.size(), top_k);

    af::array idx;
    af::array dist;

    // We actually only need n_library rows, but we allocate library.size()
    // rows. Fixed array size allows ArrayFire to recycle previously allocated
    // buffers and greatly reduces memory allocations on the GPU.
    std::vector<float> library_block_host(E * library.size());
    // Same with library
    std::vector<float> predictee_block_host(E * predictee.size());

    // Perform embedding
    for (auto i = 0; i < E; i++) {
        // Populate the first n with input data
        for (auto j = 0; j < n_library; j++) {
            library_block_host[i * library.size() + j] = p_library[i * tau + j];
        }

        // Populate the rest with dummy data
        // We put infinity so that they are be ignored in the sorting
        for (auto j = n_library; j < library.size(); j++) {
            library_block_host[i * library.size() + j] =
                std::numeric_limits<float>::infinity();
        }

        // Same with library
        for (auto j = 0; j < n_predictee; j++) {
            predictee_block_host[i * predictee.size() + j] =
                p_predictee[i * tau + j];
        }

        for (auto j = n_predictee; j < predictee.size(); j++) {
            predictee_block_host[i * predictee.size() + j] =
                std::numeric_limits<float>::infinity();
        }
    }

    // Copy embedded blocks to GPU
    af::array library_block(library.size(), E, library_block_host.data());
    af::array predictee_block(predictee.size(), E, predictee_block_host.data());

    // Compute k-nearest neighbors
    af::nearestNeighbour(idx, dist, predictee_block, library_block, 1, top_k,
                         AF_SSD);
    // Compute L2 norms from SSDs
    dist = af::sqrt(dist);

    // Copy distances and indices to CPU
    dist.host(out.distances.data());
    idx.host(out.indices.data());

    // Trim the last (E-1)*tau invalid rows
    out.resize(n_predictee, top_k);
}
