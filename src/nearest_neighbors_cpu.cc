#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "nearest_neighbors_cpu.h"

NearestNeighborsCPU::NearestNeighborsCPU(int tau, bool verbose)
    : NearestNeighbors(tau, verbose)
{
}

// clang-format off
void NearestNeighborsCPU::compute_lut(LUT &out, const Timeseries &library,
                                      const Timeseries &predictee, int E, int top_k)
{
    const auto n_library = library.size() - (E - 1) * tau;
    const auto n_predictee = predictee.size() - (E - 1) * tau;
    const auto p_library = library.data();
    const auto p_predictee = predictee.data();

    cache.resize(n_predictee, n_predictee);

    // Compute distances between all library and predictee points
    #pragma omp parallel for
    for (auto i = 0; i < n_predictee; i++) {
        std::vector<float> ssd(n_library);

        for (auto k = 0; k < E; k++) {
            #pragma omp simd
            #pragma code_align 32
            for (auto j = 0; j < n_library; j++) {
                // Perform embedding on-the-fly
                auto diff = p_predictee[i + k * tau] - p_library[j + k * tau];
                ssd[j] += diff * diff;
            }
        }

        #pragma omp simd
        for (auto j = 0; j < n_library; j++) {
            cache.distances[i * n_predictee + j] = ssd[j];
            cache.indices[i * n_predictee + j] = j;
        }
    }

    // Sort indices
    #pragma omp parallel for
    for (auto i = 0; i < n_predictee; i++) {
        std::partial_sort(cache.indices.begin() + i * n_predictee,
                          cache.indices.begin() + i * n_predictee + top_k,
                          cache.indices.begin() + (i + 1) * n_predictee,
                          [&](int a, int b) -> int {
                              return cache.distances[i * n_predictee + a] <
                                     cache.distances[i * n_predictee + b];
                          });
    }

    // Allocate buffer in LUT
    out.resize(n_predictee, top_k);

    // Compute L2 norms from SSDs and reorder them to match the indices
    #pragma omp parallel for
    for (auto i = 0; i < n_predictee; i++) {
        #pragma omp simd
        #pragma nounroll
        for (auto j = 0; j < top_k; j++) {
            auto idx = cache.indices[i * n_predictee + j];
            out.distances[i * top_k + j] =
                std::sqrt(cache.distances[i * n_predictee + idx]);
            out.indices[i * top_k + j] = idx;
        }
    }
}
// clang-format on
