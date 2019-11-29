#ifndef __BLOCK_HPP__
#define __BLOCK_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "dataset.hpp"
#include "timer.hpp"

class LUT
{
public:
    // distances[i * n + j]: Eucledian distance between point i and j
    std::vector<float> distances;
    // indices[i * n + j]: Index of the j-th closest point from point i
    std::vector<int> indices;
    int n;

    void print_distance_matrix() const
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << distances[i * n + j] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void print() const
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = indices[i * n + j];
                std::cout << idx << " (" << distances[i * n + idx] << "), ";
            }
            std::cout << std::endl;
        }
    }
};

class Block
{
public:
    // Pointer to the timeseries we are working on
    // Note that we do NOT own memory, Dataset holds it
    const float *col;
    // Embedding dimension (number of columns)
    int E;
    // Lag
    int tau;
    // Number of rows
    int n;

    Block(const Dataset &ds, int col_idx, int E, int tau) : E(E), tau(tau)
    {
        col = ds.cols[col_idx].data();
        n = ds.n_rows - (E - 1) * tau;
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
        const int N = n;

        cache.n = n;
        cache.distances.resize(n * n);
        cache.indices.resize(n * n);

        Timer timer;
        timer.start();

        // Compute distances between all points
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            std::vector<float> norms(n);

            for (int k = 0; k < E; k++) {
                for (int j = 0; j < N; j++) {
                    // Perform embedding on-the-fly
                    float diff = col[i + k * tau] - col[j + k * tau];
                    norms[j] += diff * diff;
                }
            }

            #pragma ivdep
            for (int j = 0; j < N; j++) {
                cache.distances[i * N + j] = std::sqrt(norms[j]);
                cache.indices[i * N + j] = j;
            }
        }

        #pragma ivdep
        for (int i = 0; i < N; i++) {
            cache.distances[i * N + i] = std::numeric_limits<float>::max();
        }

        timer.stop();
        // std::cout << "Distance calculated in " << timer.elapsed() << " [ms]"
        //           << std::endl;
        timer.reset();

        timer.start();

        // Sort indices
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            std::partial_sort(
                cache.indices.begin() + i * n,
                cache.indices.begin() + i * n + top_k,
                cache.indices.begin() + (i + 1) * n,
                [&](int a, int b) -> int {
                    return cache.distances[i * n + a] < cache.distances[i * n + b];
                });
        }

        timer.stop();
        // std::cout << "Sorted in " << timer.elapsed() << " [ms]" << std::endl;

        out.n = n;
        out.distances.resize(top_k * n);
        out.indices.resize(top_k * n);

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            std::copy(cache.distances.begin() + i * n,
                      cache.distances.begin() + i * n + top_k,
                      out.distances.begin() + i * top_k
                      );
            std::copy(cache.indices.begin() + i * n,
                      cache.indices.begin() + i * n + top_k,
                      out.indices.begin() + i * top_k
                      );
        }

    }
};

#endif
