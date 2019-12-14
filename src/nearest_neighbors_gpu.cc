#include <limits>

#include <arrayfire.h>

#include "nearest_neighbors_gpu.h"

NearestNeighborsGPU::NearestNeighborsGPU(uint32_t tau, bool verbose)
    : NearestNeighbors(tau, verbose)
{
    af::info();
}

void NearestNeighborsGPU::compute_lut(LUT &out, const Timeseries &library,
                                      const Timeseries &target, uint32_t E,
                                      uint32_t top_k)
{
    const auto n_library = library.size() - (E - 1) * tau;
    const auto n_target = target.size() - (E - 1) * tau;
    const auto p_library = library.data();
    const auto p_target = target.data();

    out.resize(target.size(), top_k);

    af::array idx;
    af::array dist;

    // We actually only need n_library rows, but we allocate library.size()
    // rows. Fixed array size allows ArrayFire to recycle previously allocated
    // buffers and greatly reduces memory allocations on the GPU.
    std::vector<float> library_block_host(E * library.size());
    // Same with library
    std::vector<float> target_block_host(E * target.size());

    // Perform embedding
    for (auto i = 0ul; i < E; i++) {
        // Populate the first n with input data
        for (auto j = 0ul; j < n_library; j++) {
            library_block_host[i * library.size() + j] = p_library[i * tau + j];
        }

        // Populate the rest with dummy data
        // We put infinity so that they are be ignored in the sorting
        for (auto j = n_library; j < library.size(); j++) {
            library_block_host[i * library.size() + j] =
                std::numeric_limits<float>::infinity();
        }

        // Same with library
        for (auto j = 0ul; j < n_target; j++) {
            target_block_host[i * target.size() + j] = p_target[i * tau + j];
        }

        for (auto j = n_target; j < target.size(); j++) {
            target_block_host[i * target.size() + j] =
                std::numeric_limits<float>::infinity();
        }
    }

    // Copy embedded blocks to GPU
    af::array library_block(library.size(), E, library_block_host.data());
    af::array target_block(target.size(), E, target_block_host.data());

    // Compute k-nearest neighbors
    af::nearestNeighbour(idx, dist, target_block, library_block, 1, top_k,
                         AF_SSD);
    // Compute L2 norms from SSDs
    dist = af::sqrt(dist);

    // Copy distances and indices to CPU
    dist.host(out.distances.data());
    idx.host(out.indices.data());

    // Trim the last (E-1)*tau invalid rows
    out.resize(n_target, top_k);
}
