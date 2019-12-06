#ifndef __KNN_KERNEL_CPU_HPP__
#define __KNN_KERNEL_CPU_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "knn_kernel.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernelCPU : public KNNKernel
{
public:
    KNNKernelCPU(int E_max, int tau, int k, bool verbose) : KNNKernel(E_max, tau, k, verbose) {}

    void compute_lut(LUT &out, const float *const col, int E, int n)
    {
        cache.resize(n, n);

        Timer timer;
        timer.start();

        // Compute distances between all points
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            std::vector<float> norms(n);

            for (int k = 0; k < E; k++) {
                #pragma omp simd
                #pragma code_align 32
                for (int j = 0; j < n; j++) {
                    // Perform embedding on-the-fly
                    float diff = col[i + k * tau] - col[j + k * tau];
                    norms[j] += diff * diff;
                }
            }

            #pragma omp simd
            for (int j = 0; j < n; j++) {
                cache.distances[i * n + j] = norms[j];
                cache.indices[i * n + j] = j;
            }
        }

        for (int i = 0; i < n; i++) {
            cache.distances[i * n + i] = std::numeric_limits<float>::max();
        }

        timer.stop();
        // std::cout << "Distance calculated in " << timer.elapsed() << " [ms]"
        //           << std::endl;
        timer.reset();

        timer.start();

        // Sort indices
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            std::partial_sort(cache.indices.begin() + i * n,
                              cache.indices.begin() + i * n + top_k,
                              cache.indices.begin() + (i + 1) * n,
                              [&](int a, int b) -> int {
                                  return cache.distances[i * n + a] <
                                         cache.distances[i * n + b];
                              });
        }

        timer.stop();
        // std::cout << "Sorted in " << timer.elapsed() << " [ms]" << std::endl;

        out.resize(n, top_k);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {

            #pragma omp simd
            #pragma nounroll
            for (int j = 0; j < top_k; j++) {
                int idx = cache.indices[i * n + j];
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
