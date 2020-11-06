#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#include "nearest_neighbors_cpu.h"

NearestNeighborsCPU::NearestNeighborsCPU(uint32_t tau, uint32_t Tp,
                                         bool verbose)
    : NearestNeighbors(tau, Tp, verbose)
{
}

// clang-format off
void NearestNeighborsCPU::compute_lut(LUT &out, const Series &library,
                                      const Series &target, uint32_t E,
                                      uint32_t top_k)
{
    const auto shift = (E - 1) * tau + Tp;

    const auto n_library = library.size() - shift;
    const auto n_target = target.size() - shift + Tp;
    const auto p_library = library.data();
    const auto p_target = target.data();

    // Allocate temporary buffer for distance matrix
    distances.resize(n_target * n_library);

    timer_distances.start();

    // Compute distances between all library and target points
    #pragma omp parallel
    {
        LIKWID_MARKER_START("calc_distances");
    }

    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        std::vector<float> ssd(n_library);

        for (auto k = 0u; k < E; k++) {
            const float tmp = p_target[i + k * tau];

            #pragma omp simd
            for (auto j = 0u; j < n_library; j++) {
                // Perform embedding on-the-fly
                auto diff = tmp - p_library[j + k * tau];
                ssd[j] += diff * diff;
            }
        }

        #pragma omp simd
        for (auto j = 0u; j < n_library; j++) {
            distances[i * n_library + j] = ssd[j];
        }
    }

    #pragma omp parallel
    {
        LIKWID_MARKER_STOP("calc_distances");
    }

    // Ignore degenerate neighbors
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        for (auto j = 0u; j < n_library; j++) {
            if (p_target + i == p_library + j) {
                distances[i * n_library + j] =
                    std::numeric_limits<float>::infinity();
            }
        }
    }

    timer_distances.stop();

    // Allocate buffer in LUT
    out.resize(n_target, top_k);

    timer_sorting.start();

    // Sort indices
    #pragma omp parallel
    {
        LIKWID_MARKER_START("partial_sort");

        #pragma omp for
        for (auto i = 0u; i < n_target; i++) {
            std::partial_sort_copy(Counter<uint32_t>(0),
                                   Counter<uint32_t>(n_library),
                                   out.indices.begin() + i * top_k,
                                   out.indices.begin() + (i + 1) * top_k,
                                   [&](uint32_t a, uint32_t b) -> uint32_t {
                                       return distances[i * n_library + a] <
                                              distances[i * n_library + b];
                                   });
        }

        LIKWID_MARKER_STOP("partial_sort");
    }

    timer_sorting.stop();

    // Compute L2 norms from SSDs and reorder them to match the indices
    // Shift indices
    #pragma omp parallel for
    for (auto i = 0u; i < n_target; i++) {
        for (auto j = 0u; j < top_k; j++) {
            auto idx = out.indices[i * top_k + j];
            out.distances[i * top_k + j] =
                std::sqrt(distances[i * n_library + idx]);
            out.indices[i * top_k + j] = idx + shift;

        }
    }
}
// clang-format on
