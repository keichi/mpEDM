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

#endif
