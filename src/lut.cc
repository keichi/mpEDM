#include <algorithm>
#include <cmath>
#include <iostream>

#include "lut.h"

void LUT::resize(uint32_t nr, uint32_t nc)
{
    _n_rows = nr;
    _n_columns = nc;
    distances.resize(nr * nc);
    indices.resize(nr * nc);
}

// cppcheck-suppress unusedFunction
void LUT::print_distances() const
{
    for (auto i = 0u; i < _n_rows; i++) {
        for (auto j = 0u; j < _n_columns; j++) {
            std::cout << distances[i * _n_columns + j] << ", ";
        }
        std::cout << std::endl;
    }
}

// cppcheck-suppress unusedFunction
void LUT::print_indices() const
{
    for (auto i = 0u; i < _n_rows; i++) {
        for (auto j = 0u; j < _n_columns; j++) {
            std::cout << indices[i * _n_columns + j] << ", ";
        }
        std::cout << std::endl;
    }
}

// Convert distances to exponential scale, normalize and handle zeros
void LUT::normalize()
{
    for (auto i = 0u; i < _n_rows; i++) {
        auto sum_weights = 0.0f;
        const auto min_dist =
            *min_element(distances.begin() + i * _n_columns,
                         distances.begin() + (i + 1) * _n_columns);

        for (auto j = 0u; j < _n_columns; j++) {
            const auto dist = distances[i * _n_columns + j];
            auto weighted_dist = 0.0f;

            if (min_dist > 0.0f) {
                weighted_dist = std::exp(-dist / min_dist);
            } else {
                weighted_dist = dist > 0.0f ? 0.0f : 1.0f;
            }

            const auto weight = std::max(weighted_dist, min_weight);

            distances[i * _n_columns + j] = weight;
            sum_weights += weight;
        }

        for (auto j = 0u; j < _n_columns; j++) {
            distances[i * _n_columns + j] /= sum_weights;
        }
    }
}
