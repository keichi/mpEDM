#ifndef __BLOCK_HPP__
#define __BLOCK_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "dataset.hpp"
#include "lut.hpp"
#include "timer.hpp"

class Block
{
public:
    Block(const Dataset &ds, int col_idx, int E, int tau)
        : col(ds.cols[col_idx].data()), E(E), tau(tau),
          n(ds.n_rows - (E - 1) * tau)
    {
    }

    void print() const
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < E; j++) {
                std::cout << col[(E - j - 1) * tau + i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void compute_lut(LUT &out, int top_k, LUT &cache) const
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
            for (int j = 0; j < top_k; j++) {
                out.distances[i * n + j] =
                    std::sqrt(cache.distances[i * n + j]);
            }
            std::copy(cache.indices.begin() + i * n,
                      cache.indices.begin() + i * n + top_k,
                      out.indices.begin() + i * top_k);
        }
    }

protected:
    // Pointer to the timeseries we are working on
    // Note that we do NOT own memory, Dataset holds it
    const float *col;
    // Embedding dimension (number of columns)
    int E;
    // Lag
    int tau;
    // Number of rows
    int n;
};

#endif
