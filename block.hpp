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
    std::vector<std::vector<float>> distances;
    std::vector<std::vector<int>> indices;
};

class Block
{
public:
    std::vector<const float *> cols;
    int E;
    int tau;
    int n;

    Block(const Dataset &ds, int col_idx, int E, int tau) : E(E), tau(tau)
    {
        cols.resize(E);

        for (int i = 0; i < E; i++) {
            cols[i] = ds.cols[col_idx].data() + (E - i - 1) * tau;
        }

        n = ds.n_rows - (E - 1) * tau;
    }

    void print() const
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < E; j++) {
                std::cout << cols[j][i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void compute_lut(LUT &lut) const
    {
        lut.distances.resize(n, std::vector<float>(n));
        lut.indices.resize(n, std::vector<int>(n));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float norm = 0.0f;

                for (int k = 0; k < E; k++) {
                    float diff = cols[k][i] - cols[k][j];
                    norm += diff * diff;
                }

                lut.distances[i][j] = std::sqrt(norm);
                lut.indices[i][j] = j;
            }
        }

        for (int i = 0; i < n; i++) {
            lut.distances[i][i] = std::numeric_limits<float>::max();
        }

        for (int i = 0; i < n; i++) {
            std::sort(lut.indices[i].begin(), lut.indices[i].end(),
                      [&](int x, int y) -> int {
                          return lut.distances[i][x] < lut.distances[i][y];
                      });
        }
    }
};

#endif
