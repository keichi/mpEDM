#ifndef __LUT_HPP__
#define __LUT_HPP__

#include <vector>

class LUT
{
public:
    // Eucledian distance between point i and j
    std::vector<float> distances;
    // Index of the j-th closest point from point i
    std::vector<int> indices;

    float distance(int i, int j) const
    {
        return distances[i * n_cols + j];
    }
    int index(int i, int j) const {
        return indices[i * n_cols + j];
    }

    void resize(int nr, int nc);
    void print_distance_matrix() const;
    void print() const;

protected:
    int n_rows;
    int n_cols;
};

#endif
