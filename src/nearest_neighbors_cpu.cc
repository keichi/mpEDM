#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "nearest_neighbors_cpu.h"

NearestNeighborsCPU::NearestNeighborsCPU(uint32_t tau, bool verbose)
    : NearestNeighbors(tau, verbose)
{
}

// clang-format off
void NearestNeighborsCPU::compute_lut(LUT &out, const Timeseries &library,
                                      const Timeseries &target, uint32_t E,
                                      uint32_t top_k)
{
    const auto n_library = library.size() - (E - 1) * tau;
    const auto n_target = target.size() - (E - 1) * tau;
    const auto p_library = library.data();
    const auto p_target = target.data();

    cache.resize(n_target, n_library);

    // Compute distances between all library and target points
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        std::vector<float> ssd(n_library);

        for (auto k = 0u; k < E; k++) {
            #pragma omp simd
            #pragma code_align 32
            for (auto j = 0u; j < n_library; j++) {
                // Perform embedding on-the-fly
                auto diff = p_target[i + k * tau] - p_library[j + k * tau];
                ssd[j] += diff * diff;
            }
        }

        #pragma omp simd
        for (auto j = 0u; j < n_library; j++) {
            cache.distances[i * n_library + j] = ssd[j];
            cache.indices[i * n_library + j] = j;
        }
    }

    // Ignore degenerate neighbors
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        for (auto j = 0u; j < n_library; j++) {
            if (p_target + i != p_library + j) {
                continue;
            }

            cache.distances[i * n_library + j] =
                std::numeric_limits<float>::infinity();
        }
    }

    // Sort indices
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        std::partial_sort(cache.indices.begin() + i * n_library,
                          cache.indices.begin() + i * n_library + top_k,
                          cache.indices.begin() + (i + 1) * n_library,
                          [&](uint32_t a, uint32_t b) -> uint32_t {
                              return cache.distances[i * n_library + a] <
                                     cache.distances[i * n_library + b];
                          });
    }

    // Allocate buffer in LUT
    out.resize(n_target, top_k);

    // Compute L2 norms from SSDs and reorder them to match the indices
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        #pragma omp simd
        #pragma nounroll
        for (auto j = 0u; j < top_k; j++) {
            auto idx = cache.indices[i * n_library + j];
            out.distances[i * top_k + j] =
                std::sqrt(cache.distances[i * n_library + idx]);
            out.indices[i * top_k + j] = idx;
        }
    }
}
// clang-format on
