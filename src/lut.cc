#include <algorithm>
#include <cmath>
#include <iostream>

#include "lut.h"

void LUT::resize(int nr, int nc)
{
    _n_rows = nr;
    _n_cols = nc;
    distances.resize(nr * nc);
    indices.resize(nr * nc);
}

void LUT::print_distance_matrix() const
{
    for (auto i = 0; i < _n_rows; i++) {
        for (auto j = 0; j < _n_cols; j++) {
            std::cout << distance(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

void LUT::print() const
{
    for (auto i = 0; i < _n_rows; i++) {
        for (auto j = 0; j < _n_cols; j++) {
            std::cout << index(i, j) << " (" << distance(i, j) << "), ";
        }
        std::cout << std::endl;
    }
}

// Convert distances to exponential scale, normalize and handle zeros
void LUT::normalize()
{
    for (auto i = 0; i < _n_rows; i++) {
        auto sum_weights = 0.0f;
        const auto min_dist =
            *min_element(distances.begin() + i * _n_cols,
                         distances.begin() + (i + 1) * _n_cols);

        for (auto j = 0; j < _n_cols; j++) {
            const auto dist = distances[i * _n_cols + j];
            const auto weighted_dist =
                min_dist > 0.0f ? std::exp(-dist / min_dist) : 1.0f;
            const auto weight = std::max(weighted_dist, min_weight);

            distances[i * _n_cols + j] = weight;
            sum_weights += weight;
        }

        for (auto j = 0; j < _n_cols; j++) {
            distances[i * _n_cols + j] /= sum_weights;
        }
    }
}
