#ifndef __LUT_HPP__
#define __LUT_HPP__

#include <vector>

class LUT
{
public:
    LUT() : _n_rows(0), _n_cols(0) {}
    LUT(int n_rows, int n_cols)
        : distances(n_rows * n_cols), indices(n_rows * n_cols), _n_rows(n_rows),
          _n_cols(n_cols)
    {
    }
    LUT(int n_rows, int n_cols, const std::vector<float> &distances,
        const std::vector<int> &indices)
        : distances(distances), indices(indices), _n_rows(n_rows),
          _n_cols(n_cols)
    {
    }

    // Eucledian distance between point i and j
    std::vector<float> distances;
    // Index of the j-th closest point from point i
    std::vector<int> indices;

    float distance(int i, int j) const { return distances[i * _n_cols + j]; }
    int index(int i, int j) const { return indices[i * _n_cols + j]; }
    float n_rows() const { return _n_rows; }
    int n_cols() const { return _n_cols; }

    void resize(int nr, int nc);
    void print_distances() const;
    void print_indices() const;
    void normalize();

protected:
    int _n_rows;
    int _n_cols;

    // Minimum weight
    const float min_weight = 1e-6f;
};

#endif
