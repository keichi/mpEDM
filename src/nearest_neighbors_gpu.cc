#include <limits>
#include <vector>

#include <arrayfire.h>

#include "nearest_neighbors_gpu.h"

NearestNeighborsGPU::NearestNeighborsGPU(uint32_t tau, uint32_t Tp,
                                         bool verbose)
    : NearestNeighbors(tau, Tp, verbose)
{
    if (verbose) {
        af::info();
    }
}

void NearestNeighborsGPU::compute_lut(LUT &out, const Series &library,
                                      const Series &target, uint32_t E,
                                      uint32_t top_k)
{
    const auto shift = (E - 1) * tau + Tp;

    const auto n_library = library.size() - shift;
    const auto n_target = target.size() - shift + Tp;
    const auto p_library = library.data();
    const auto p_target = target.data();

    af::array idx;
    af::array dist;

    // We only need library.size() - (E - 1) * tau rows, but we allocate
    // library.size() rows. Using the same array size across all E allows
    // ArrayFire to recycle previously allocated buffers and greatly reduces
    // memory allocations on the GPU.
    std::vector<float> library_block_host(E * library.size());
    // Same with library
    std::vector<float> target_block_host(E * target.size());

    // Perform embedding
    // TODO Use OpenMP?
    for (auto i = 0u; i < E; i++) {
        // Populate the first n with input data
        for (auto j = 0u; j < n_library; j++) {
            library_block_host[i * library.size() + j] = p_library[i * tau + j];
        }

        // Populate the rest with dummy data
        // We put infinity so that they are be ignored in the sorting
        for (auto j = n_library; j < library.size(); j++) {
            library_block_host[i * library.size() + j] =
                std::numeric_limits<float>::infinity();
        }

        // Same with library
        for (auto j = 0u; j < n_target; j++) {
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
    af::nearestNeighbour(idx, dist, target_block, library_block, 1, top_k + 1,
                         AF_SSD);
    // Compute L2 norms from SSDs
    dist = af::sqrt(dist);

    std::vector<uint32_t> idx_host(target.size() * (top_k + 1));
    std::vector<float> dist_host(target.size() * (top_k + 1));

    // Copy distances and indices to CPU
    idx.host(idx_host.data());
    dist.host(dist_host.data());

    out.resize(n_target, top_k);

    // Remove degenerate neighbors
    // Shift indices
    // TODO Use OpenMP?
    for (auto i = 0u; i < n_target; i++) {
        auto offset = 0u;

        for (auto j = 0u; j < top_k; j++) {
            // Remove degenerate neighbor. Note that we assume there is only
            // one degenerate neighbor for each library row.
            if (p_library + idx_host[i * (top_k + 1) + j] == p_target + i) {
                offset = 1u;
            }

            out.distances[i * top_k + j] =
                dist_host[i * (top_k + 1) + j + offset];
            out.indices[i * top_k + j] =
                idx_host[i * (top_k + 1) + j + offset] + shift;
        }
    }
}
