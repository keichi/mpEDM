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
    for (int i = 0; i < _n_rows; i++) {
        for (int j = 0; j < _n_cols; j++) {
            std::cout << distances[i * _n_cols + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void LUT::print() const
{
    for (int i = 0; i < _n_rows; i++) {
        for (int j = 0; j < _n_cols; j++) {
            int idx = indices[i * _n_cols + j];
            float dist = distances[i * _n_cols + j];
            std::cout << idx << " (" << dist << "), ";
        }
        std::cout << std::endl;
    }
}
