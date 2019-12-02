#ifndef __LUT_HPP__
#define __LUT_HPP__

#include <iostream>
#include <vector>

class LUT
{
public:
    // distances[i * n + j]: Eucledian distance between point i and j
    std::vector<float> distances;
    // indices[i * n + j]: Index of the j-th closest point from point i
    std::vector<int> indices;

    void resize(int nr, int nc)
    {
        n_rows = nr;
        n_cols = nc;
        distances.resize(nr * nc);
        indices.resize(nr * nc);
    }

    void print_distance_matrix() const
    {
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                std::cout << distances[i * n_rows + j] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void print() const
    {
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                int idx = indices[i * n_rows + j];
                std::cout << idx << " (" << distances[i * n_rows + idx]
                          << "), ";
            }
            std::cout << std::endl;
        }
    }

protected:
    int n_rows;
    int n_cols;
};

#endif
