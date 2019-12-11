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

    float distance(int i, int j) const { return distances[i * _n_cols + j]; }
    int index(int i, int j) const { return indices[i * _n_cols + j]; }
    float n_rows() const { return _n_rows; }
    int n_cols() const { return _n_cols; }

    void resize(int nr, int nc);
    void print_distance_matrix() const;
    void print() const;

protected:
    int _n_rows;
    int _n_cols;
};

#endif
