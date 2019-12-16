#ifndef __LUT_HPP__
#define __LUT_HPP__

#include <vector>

class LUT
{
public:
    LUT() : _n_rows(0), _n_cols(0) {}
    LUT(size_t n_rows, size_t n_cols)
        : distances(n_rows * n_cols), indices(n_rows * n_cols), _n_rows(n_rows),
          _n_cols(n_cols)
    {
    }
    LUT(size_t n_rows, size_t n_cols, const std::vector<float> &distances,
        const std::vector<size_t> &indices)
        : distances(distances), indices(indices), _n_rows(n_rows),
          _n_cols(n_cols)
    {
    }

    // Eucledian distance between point i and j
    std::vector<float> distances;
    // Index of the j-th closest point from point i
    std::vector<size_t> indices;

    float distance(size_t i, size_t j) const
    {
        return distances[i * _n_cols + j];
    }
    size_t index(size_t i, size_t j) const { return indices[i * _n_cols + j]; }
    size_t n_rows() const { return _n_rows; }
    size_t n_cols() const { return _n_cols; }

    void resize(size_t nr, size_t nc);
    void print_distances() const;
    void print_indices() const;
    void normalize();

protected:
    size_t _n_rows;
    size_t _n_cols;

    // Minimum weight
    const float min_weight = 1e-6f;
};

#endif
