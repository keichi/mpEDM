#ifndef __KNN_KERNEL_CPU_HPP__
#define __KNN_KERNEL_CPU_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "nearest_neighbors.h"
#include "lut.h"
#include "timer.h"

class NearestNeighborsCPU : public NearestNeighbors
{
public:
    NearestNeighborsCPU(int E_max, int tau, int k, bool verbose)
        : NearestNeighbors(E_max, tau, k, verbose)
    {
    }

    void compute_lut(LUT &out, const Timeseries &ts, int E)
    {
        auto n = ts.size() - (E - 1) * tau;

        cache.resize(n, n);

        // Compute distances between all points
        #pragma omp parallel for
        for (auto i = 0; i < n; i++) {
            std::vector<float> norms(n);

            for (auto k = 0; k < E; k++) {
                auto p = ts.data();

                #pragma omp simd
                #pragma code_align 32
                for (auto j = 0; j < n; j++) {
                    // Perform embedding on-the-fly
                    float diff = p[i + k * tau] - p[j + k * tau];
                    norms[j] += diff * diff;
                }
            }

            #pragma omp simd
            for (auto j = 0; j < n; j++) {
                cache.distances[i * n + j] = norms[j];
                cache.indices[i * n + j] = j;
            }
        }

        // Sort indices
        #pragma omp parallel for
        for (auto i = 0; i < n; i++) {
            std::partial_sort(cache.indices.begin() + i * n,
                              cache.indices.begin() + i * n + top_k,
                              cache.indices.begin() + (i + 1) * n,
                              [&](int a, int b) -> int {
                                  return cache.distances[i * n + a] <
                                         cache.distances[i * n + b];
                              });
        }

        out.resize(n, top_k);

        #pragma omp parallel for
        for (auto i = 0; i < n; i++) {
            #pragma omp simd
            #pragma nounroll
            for (auto j = 0; j < top_k; j++) {
                auto idx = cache.indices[i * n + j];
                out.distances[i * top_k + j] =
                    std::sqrt(cache.distances[i * n + idx]);
                out.indices[i * top_k + j] = idx;
            }
        }
    }

protected:
    LUT cache;
};

#endif
