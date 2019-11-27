#ifndef __BLOCK_HPP__
#define __BLOCK_HPP__

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "dataset.hpp"

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

    void compute_lut(LUT &lut) const
    {
        lut.n = n;
        lut.distances.resize(n * n);
        lut.indices.resize(n * n);

        // Compute distances between all points
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float norm = 0.0f;

                for (int k = 0; k < E; k++) {
                    // Perform embedding on-the-fly
                    float diff = col[i + k * tau] - col[j + k * tau];
                    norm += diff * diff;
                }

                lut.distances[i * n + j] = std::sqrt(norm);
                lut.indices[i * n + j] = j;
            }
        }

        for (int i = 0; i < n; i++) {
            lut.distances[i * n + i] = std::numeric_limits<float>::max();
        }

        // Sort indices
        for (int i = 0; i < n; i++) {
            std::sort(
                lut.indices.begin() + i * n, lut.indices.begin() + (i + 1) * n,
                [&](int a, int b) -> int {
                    return lut.distances[i * n + a] < lut.distances[i * n + b];
                });
        }
    }
};

#endif
